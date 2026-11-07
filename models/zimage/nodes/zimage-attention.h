#ifndef ZIMAGE_ATTENTION_H
#define ZIMAGE_ATTENTION_H

#include "vai/nodes/node-common.h"
#include "vai/nodes/attention.h"   // ggmlCoopmatAllowedAttention
#include <cmath>

namespace zimage {

/*=============================================================================
  FullAttentionNode - bidirectional multi-head attention over one sequence
  Input : Q [H, N, Dh] f16, K [H, N, Dh] f16 (QkNormRopeNode's layout),
          V [N, H*Dh] (token-major, the projection output as is)
  Output: out0 [N, H*Dh] = softmax(Q K^T / sqrt(Dh)) V, token-major

  A DiT runs a few thousand rows of prefill-only attention per layer, so
  this node's kernels put many rows on one shared K/V copy (prefetched a
  tile ahead), with no cache, append or split-K to carry:
    coopmat  16 rows per subgroup, numSubgroups subgroups per workgroup,
             16x16x16 f16 cooperative matrices (zimage-attention.comp).
             Measured on an A40 (DiT layer, N = 4128, 30 heads): 4.8 ms with
             8 subgroups, 5.8 ms with 4, vs. 9.0 ms for vai::node::Attention
             run cacheless under an all-bidirectional mask (its coopmat
             kernel plus the permute its head-major V needs); 32 rows per
             subgroup spilled and ran 3x slower.
    scalar   32 rows per 128 threads on plain FMAs, for devices without
             (trusted) cooperative matrices (zimage-attention-scalar.comp).
             Its tile is scalarKeysPerTile: with 32 keys the per-thread
             scores and prefetch spill registers on an Adreno 830 and run
             ~140x slower (past the driver's submit timeout at DiT lengths).
    int8     the scalar frame with the Q·K dots on packed int8 dot products
             (K quantized per key while staged, the query per row) and V
             staged as fp16 (zimage-attention-i8.comp), where
             int8Supported(). On a cool Galaxy S25 Ultra (N = 1056, 30
             heads) 34 ms vs. 83 ms scalar; the 512x512 forward's rel L2
             against fp32 goes 3.70e-2 -> 3.91e-2.
  coopmatSupported() picks first; allowCoopmat = false and int8QK = false
  turn the kernels off.
=============================================================================*/
class FullAttentionNode : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams {
        bool coopmat = false;
        bool int8 = false;
        uint32_t keysPerTile = 0;
        vai::Dtype vDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        uint32_t numHeads;
        uint32_t headDim;                       // multiple of 16
        uint32_t numSubgroups = 8;              // coopmat: query rows per workgroup = 16 * numSubgroups
        uint32_t keysPerTile = 32;              // coopmat; multiple of 16, N must be a multiple of it
        uint32_t scalarKeysPerTile = 16;        // scalar; N must be a multiple of it
        bool allowCoopmat = true;
        bool int8QK = true;                     // scalar devices: zimage-attention-i8 where supported
        vai::Dtype oDtype = vai::Dtype::None;   // None: oDefaultDtype
    } config_;

    FullAttentionNode(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

    // Whether the coopmat kernel can run with this config on `device`.
    static bool coopmatSupported(const Config& config, eva::Device device);
    // Whether the int8 kernel can run with this config on `device`.
    static bool int8Supported(const Config& config, eva::Device device);
    bool usingCoopmat() const { return pipeParams.coopmat; }
    bool usingInt8() const { return pipeParams.int8; }

private:
    PipeParams pipeParams;
    uint32_t rowsPerWg = 0;
};


inline bool FullAttentionNode::coopmatSupported(const Config& c, eva::Device device)
{
    if (!device.supportsCooperativeMatrix() || !vai::node::ggmlCoopmatAllowedAttention(device))
        return false;

    bool shapeOk = false;   // 16x16x16, f16 A/B, f32 accumulator, subgroup scope
    for (const auto& p : device.cooperativeMatrixProperties())
        if (p.M == 16 && p.N == 16 && p.K == 16 &&
            p.aType == eva::COMPONENT_TYPE::FLOAT16 && p.bType == eva::COMPONENT_TYPE::FLOAT16 &&
            p.cType == eva::COMPONENT_TYPE::FLOAT32 && p.resultType == eva::COMPONENT_TYPE::FLOAT32 &&
            p.scope == eva::SCOPE::SUBGROUP)
            shapeOk = true;

    const uint32_t sg = device.subgroupSize();
    if (!shapeOk || (sg != 16 && sg != 32 && sg != 64) || c.keysPerTile % (sg / 16) != 0)
        return false;

    // Shared arrays of zimage-attention.comp.
    const size_t rows = 16 * c.numSubgroups;
    const size_t shared = 2 * size_t(c.keysPerTile) * (c.headDim / 4 + 2) * 8   // s_k, s_v
                        + rows * (c.keysPerTile + 4) * 4                         // s_s
                        + rows * (c.keysPerTile + 8) * 2                         // s_p
                        + rows * 4;                                              // s_row
    return shared <= device.maxComputeSharedMemorySize();
}

inline bool FullAttentionNode::int8Supported(const Config& c, eva::Device device)
{
    // zimage-attention-i8.comp: compile-time head dim and tile, one subgroup
    // butterfly over the 32 invocations that stage a key.
    return device.isDot4x8PackedMixedAccelerated()
        && c.headDim == 128 && c.scalarKeysPerTile == 16 && device.subgroupSize() >= 32;
}

inline FullAttentionNode::FullAttentionNode(Config config)
: config_(config)
{
    addSlot("out0", vai::NodeSlot::output);
    addSlot("Q", vai::NodeSlot::input);
    addSlot("K", vai::NodeSlot::input);
    addSlot("V", vai::NodeSlot::input);

    VAI_ASSERT(config_.headDim % 16 == 0 && config_.keysPerTile % 16 == 0);
    config_.oDtype = (config_.oDtype == vai::Dtype::None) ? oDefaultDtype() : config_.oDtype;
}

inline void FullAttentionNode::prepare()
{
    const uint32_t H = config_.numHeads, Dh = config_.headDim;
    const vai::Tensor& q = (*this)["Q"];
    const vai::Tensor& k = (*this)["K"];
    const vai::Tensor& v = (*this)["V"];

    VAI_ASSERT(q.dtype() == vai::Dtype::Fp16 && k.dtype() == vai::Dtype::Fp16);
    VAI_ASSERT(q.isShapeOf(H, -1, Dh));
    const uint32_t N = q.shape()[1];
    VAI_ASSERT(k.isShapeOf(H, N, Dh));
    VAI_ASSERT(v.isShapeOf(N, H * Dh));
    const bool coopmat = config_.allowCoopmat && coopmatSupported(config_, device());
    const PipeParams p = {
        .coopmat = coopmat,
        .int8 = !coopmat && config_.int8QK && int8Supported(config_, device()),
        .keysPerTile = coopmat ? config_.keysPerTile : config_.scalarKeysPerTile,
        .vDtype = v.dtype(),
    };
    VAI_ASSERT(N % 16 == 0 && N % p.keysPerTile == 0);
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        const uint32_t sg = p.coopmat ? device().subgroupSize() : 0;
        const uint32_t wgSize = p.coopmat ? config_.numSubgroups * sg : 128;
        rowsPerWg = p.coopmat ? 16 * config_.numSubgroups : wgSize / 4;

        pipe = pipelinePool().request(ZIMAGE_SHADER_OUTPUT_DIR,
            p.coopmat ? "zimage-attention" : p.int8 ? "zimage-attention-i8" : "zimage-attention-scalar", {
                eva::constant_id<0>(wgSize),
                eva::constant_id<1>(sg),                               // sgSize (coopmat; scalar ignores)
                eva::constant_id<2>(Dh),
                eva::constant_id<3>(p.keysPerTile),                    // Bc
                eva::constant_id<4>(p.vDtype),                         // typeV
                eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype)       // typeO
            },
            { .requiredSubgroupSize = sg });                           // 0 = not forced
        if (!descSet)
            descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
    }

    (*this)["out0"] = vai::Tensor({N, H * Dh}, config_.oDtype);
}

inline void FullAttentionNode::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    descSet.write({ outBuf, (*this)["Q"].buffer(), (*this)["K"].buffer(), (*this)["V"].buffer() });

    struct {
        uint32_t N, H;
        float scaleLog2;
    } constants = { (*this)["Q"].shape()[1], config_.numHeads,
                    float(1.4426950408889634 / std::sqrt(double(config_.headDim))) };

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch((constants.N + rowsPerWg - 1) / rowsPerWg, constants.H)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

} // namespace zimage

#endif // ZIMAGE_ATTENTION_H
