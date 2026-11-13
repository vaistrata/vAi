#ifndef VAI_NODES_TOPK_SAMPLING_H
#define VAI_NODES_TOPK_SAMPLING_H

// Token sampling from a logits distribution: TopKSampling.
#include "node-common.h"

namespace vai::node {


/*=============================================================================
  TopKSampling - row-wise top-k sampling, one kernel run as two passes
  Input : logits [.., N]
  Output: out0   [..]   (one raw uint32 sampled index per row, in 4-byte slots)

  Every axis ahead of N is a row, the way a Linear takes its leading axes, and they
  come back on the output where they went in.

  Keeps the k largest logits of each row, applies a temperature softmax over
  them, and samples by inverse CDF with a host-provided uniform refreshed
  every forward (setUniform()).

  Two dispatches of one kernel: G workgroups each select the top-k of one
  ceil(N/G)-logit slice — exact while the slice stays register-resident — then
  one workgroup merges the G*k candidates and samples. Small rows (G == 1)
  skip the second dispatch.

  Geometry (workgroup size, register list length), the reduction variant and G
  are selected per (shape, k, device) in topkTuning(); the workgroup size is a
  spec constant, so the pipeline is requested in prepare() once the shape is
  known, and a tuning change re-requests it from the pool (no-op while the
  tuning stays put).
=============================================================================*/
class TopKSampling : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;
    eva::Buffer randomBuffer;   // host-visible uniform [0,1), refreshed per forward

public:
    // wgSize threads hold perThread logits each in registers, so a pass-1
    // slice of at most wgSize*perThread elements stays exact.
    struct PipeParams {
        uint32_t   wgSize    = 0;
        uint32_t   perThread = 0;      // register list length in use (kernel L_MAX caps it)
        uint32_t   sgSize    = 0;      // forced subgroup size (0 = don't force; Intel)
        bool       sgReduce  = false;  // subgroupMax/Min two-level reduction instead of the shared tree

        uint32_t   numGroups = 0;      // pass-1 workgroups per row (dispatch + pass-2 N, not a spec constant)
        vai::Dtype iDtype    = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        uint32_t k = 40;
        float temperature = 1.0f;
    } config_;

    TopKSampling();
    TopKSampling(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

    // Set the uniform random used for this forward's sampling (call before run()).
    void setUniform(float u) { *(float*)randomBuffer.map() = u; }

    // Kernel caps (topk-sampling.comp L_MAX / K_MAX). perThread may only be
    // swept DOWN: the register budget is charged for L_MAX whatever perThread
    // says, so a longer list needs a new compile-time variant.
    static constexpr uint32_t maxPerThread = 8;
    static constexpr uint32_t maxK         = 128;

private:
    PipeParams pipeParams;
};


// Geometry / reduction / split count from device queries + shape.
inline TopKSampling::PipeParams topkTuning(uint32_t M, uint32_t N, uint32_t k, Dtype iDtype, eva::Device device)
{
    TopKSampling::PipeParams t;

    t.wgSize = 512;   // power of two; smaller only shrinks the slice and the pass-2 pool

    // Subgroup reduction where available, shared tree otherwise (log2(wgSize)+2
    // barriers per round instead of 2). Pinned to the max subgroup size — the
    // level-2 loop runs wgSize / sg² times — but only while the workgroup stays
    // inside the 16 subgroups Vulkan guarantees (maxComputeWorkgroupSubgroups,
    // which eva cannot query); narrower devices leave the width to the driver,
    // which the gl_NumSubgroups-strided level-2 loop handles at any width.
    const uint32_t sg = device.maxSubgroupSize();
    t.sgReduce = device.supportsSubgroupArithmetic();
    t.sgSize   = (t.sgReduce && sg >= t.wgSize / 16u) ? sg : 0u;

    // perThread: the shortest register list that is still exact (slice² >= N*k,
    // slice = wgSize * perThread) and opens no new occupancy wave — a shorter
    // list means more workgroups, and each carries a whole k-round merge.
    // An unknown core count has no wave to reason about, so it keeps maxPerThread.
    const uint32_t cores = device.computeBlockCount();

    t.perThread = TopKSampling::maxPerThread;
    if (cores)
    {
        auto wavesAt = [&](uint32_t p) { return CEIL_DIV(CEIL_DIV(N, t.wgSize * p) * M, cores); };  // over G*M

        const uint32_t minSlice = uint32_t(std::ceil(std::sqrt(double(N) * double(k))));
        const uint32_t shortest = std::clamp(CEIL_DIV(minSlice, t.wgSize), 1u, TopKSampling::maxPerThread);
        const uint32_t budget   = wavesAt(TopKSampling::maxPerThread);
        for (uint32_t p = shortest; p < TopKSampling::maxPerThread; ++p)   // non-increasing: first fit is shortest
            if (wavesAt(p) <= budget) { t.perThread = p; break; }
    }

    const uint32_t slice = t.wgSize * t.perThread;

    // Pass-1 workgroups:
    //  - gExact: a register-resident slice keeps the per-slice top-k exact.
    //  - gPool : the G*k candidates must fit one workgroup's merge. Binds only
    //    for k near maxK on huge vocabularies, where exactness degrades to the
    //    kernel's documented Poisson argument.
    //  - gFill : round up to the occupancy wave already being paid for, never
    //    past it — a new wave buys read parallelism for k more serial rounds.
    const uint32_t gExact = CEIL_DIV(N, slice);
    const uint32_t gPool  = std::max(1u, slice / k);
    const uint32_t waves  = cores ? CEIL_DIV(gExact * M, cores) : 0u;
    const uint32_t gFill  = cores ? std::max(1u, waves * cores / M) : 1u;

    t.numGroups = std::min(std::max(gExact, gFill), gPool);
    t.numGroups = std::min(t.numGroups, device.maxComputeWorkGroupCount()[0]);
    t.iDtype    = iDtype;
    return t;
}


inline TopKSampling::TopKSampling() : TopKSampling(Config()) {}

inline TopKSampling::TopKSampling(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0",  NodeSlot::input);
    addSlot("cands", NodeSlot::temporary);   // pass-1 partial top-k, assigned when numGroups > 1

    VAI_ASSERT(config_.k >= 1 && config_.k <= maxK);

    randomBuffer = device().createBuffer({
        .size = sizeof(float),
        .usage = eva::BUFFER_USAGE::UNIFORM_BUFFER,
        .reqMemProps = eva::MEMORY_PROPERTY::HOST_VISIBLE | eva::MEMORY_PROPERTY::HOST_COHERENT
    });
    *(float*)randomBuffer.map() = 0.0f;
}

inline void TopKSampling::prepare()
{
    const Tensor& inTen = (*this)["in0"];
    VAI_ASSERT(inTen.validShape());
    VAI_ASSERT(inTen.shape().size() >= 2);

    uint32_t M = outerSize(inTen.shape());
    uint32_t N = inTen.shape().back();
    uint32_t k = std::min(config_.k, N);

    // run() dispatches rows on grid y — silently broken past its limit.
    VAI_ASSERT(M <= device().maxComputeWorkGroupCount()[1]);

    const PipeParams p = topkTuning(M, N, k, inTen.dtype(), device());
    VAI_ASSERT(p.perThread <= maxPerThread);
    VAI_ASSERT((p.wgSize & (p.wgSize - 1u)) == 0u);   // shared-tree fallback halves it
    // sVal + sIdx + topV + topI
    VAI_ASSERT(2u * p.wgSize * 4u + 2u * maxK * 4u <= device().maxComputeSharedMemorySize());

    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        // clang-format off
        pipe = pipelinePool().request("topk-sampling", {
            eva::constant_id<0>(p.wgSize),                      // wgSize (= local_size_x)
            eva::constant_id<1>(p.perThread),                   // perThread (register list)
            eva::constant_id<2>(p.sgReduce ? 1u : 0u),          // sgReduce
            eva::constant_id<SPEC_ID_TYPE_I>(p.iDtype)          // typeI (logits)
        }, { .requiredSubgroupSize = p.sgSize });   // 0 = not forced
        // clang-format on
        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }

    // Pass-1 candidates, only when a split happens: [M, G*k] of (valueBits,
    // index) pairs (uvec2 = two 4-byte slots; no integer Dtype yet, fp32 stands in).
    if (pipeParams.numGroups > 1)
        (*this)["cands"] = Tensor({M, pipeParams.numGroups * k, 2}, Dtype::Fp32);

    // One raw uint32 index per row, under the axes the rows came from.
    std::vector<uint32_t> outShape = inTen.shape();
    outShape.pop_back();
    (*this)["out0"] = Tensor(outShape, Dtype::Fp32);
}

inline void TopKSampling::run(eva::CommandBuffer cmdBuff)
{
    auto inBuf  = (*this)["in0"].buffer();
    auto outBuf = (*this)["out0"].buffer();

    uint32_t M = outerSize((*this)["in0"].shape());
    uint32_t N = (*this)["in0"].shape().back();
    uint32_t k = std::min(config_.k, N);

    // Single set for both passes: logits, out indices, u, pass-1 candidates.
    // With numGroups == 1 the candidate binding is unused; outBuf stands in to
    // keep the set fully bound (same trick as Attention's partials).
    bool split = pipeParams.numGroups > 1;
    auto candBuf = split ? (*this)["cands"].buffer() : outBuf;

    descSet.write({ inBuf, outBuf, randomBuffer, candBuf });

    // K is the clamped k, as in prepare(): pass 2 sees N = G*k candidates, and
    // an unclamped k > N would only spend extra rounds on empty-slice sentinels.
    struct {
        uint32_t M, N, K;
        float temperature;
        uint32_t srcCands;
    } constants = { M, N, k, config_.temperature, 0 };

    // Pass 1: G workgroups per row, each selecting its slice's top-k. With
    // G == 1 this is already the final pass — it samples and writes out0.
    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch(pipeParams.numGroups, M, 1);

    if (split)
    {
        // Pass 2: one workgroup merges the G*k candidates and samples.
        constants.N = pipeParams.numGroups * k;
        constants.srcCands = 1;
        cmdBuff
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / candBuf / eva::SYNC_SCOPE::COMPUTE_READ)
            .setPushConstants(0, sizeof(constants), &constants)
            .dispatch(1, M, 1);
    }

    cmdBuff.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}


}   // namespace vai::node

#endif // VAI_NODES_TOPK_SAMPLING_H
