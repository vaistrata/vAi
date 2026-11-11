#ifndef VAI_NODES_LAYER_NORM_H
#define VAI_NODES_LAYER_NORM_H

#include "node-common.h"

namespace vai::node {

/*=============================================================================
  LayerNorm - layer normalization over the feature axis, single kernel
  Input : activations [..., D]
  Output: normalized  [..., D]
  scale / shift   [D] (or [1, D] as an input); parameters, or inputs computed
                 upstream under Config::coeffsAsInputs (e.g. DiT's adaLN).
                 shift applies when present: filled by the loader, or wired.

  O[i, j] = (I[i, j] - mean(I[i, :])) * rsqrt(var(I[i, :]) + eps) * scale[j] + shift[j]

  One workgroup per row (layer-norm); the row statistics come from a
  shifted 1-pass reduction (Chan–Golub–LeVeque) accumulated in fp32
  regardless of I/O dtypes. Two spec-selected variants share the kernel:

  - reduction: subgroupAdd two-level when the device has subgroup arithmetic
    (subgroup size pinned to max — Intel varies it otherwise), shared-memory
    tree fallback everywhere else.
  - loads: vec4 when I and O are fp32 and featSize % 4 == 0, scalar otherwise.

  Geometry / reduction / vec4 are selected per (featSize, device) in
  lnTuning(); the workgroup size depends on featSize, so the pipeline is
  requested in prepare() once the shape is known, and a tuning change
  re-requests it from the pool (no-op while the tuning stays put).
=============================================================================*/
class LayerNorm : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams {
        uint32_t   wgSize   = 0;      // threads per row, ~4 elements each
        uint32_t   sgSize   = 0;      // forced subgroup size (0 = don't force; Intel)
        bool       sgReduce = false;  // subgroupAdd two-level reduction instead of the shared tree
        bool       aligned  = false;  // vec4 I/O loads (fp32, featSize % 4 == 0)
        vai::Dtype iDtype   = vai::Dtype::None;
        vai::Dtype wDtype = vai::Dtype::None;
        bool hasShift = false;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        float eps = 1e-5f;
        bool coeffsAsInputs = false;   // scale / shift are inputs, not parameters
        bool acceptResidual = false;   // input "residual", added to the output
    } config_;

    LayerNorm(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

    // The loader filled it, or under coeffsAsInputs it is wired.
    bool hasShift() const { return (*this)["shift"].validShape(); }

private:
    PipeParams pipeParams;
};


// Workgroup geometry / kernel variants from device queries + shape:
// wgSize = pow2ceil(featSize / 4) in [sg, min(16 * sg, 1024)], subgroupAdd
// when supported (subgroup size pinned — Intel varies it otherwise).
inline LayerNorm::PipeParams lnTuning(uint32_t featSize, Dtype iDtype, Dtype wDtype, Dtype oDtype, eva::Device device)
{
    const uint32_t sg = device.maxSubgroupSize();

    // TODO (eva): maxComputeWorkGroupInvocations limit query — until then
    //             cap at 1024 (common desktop value; spec guarantees 128)
    const uint32_t cap = std::min(16u * sg, 1024u);
    uint32_t wg = sg;
    while (wg * 4u < featSize && wg < cap) wg *= 2u;        // pow2ceil(featSize / 4), [sg, cap]

    LayerNorm::PipeParams t;
    t.wgSize   = wg;
    t.sgReduce = device.supportsSubgroupArithmetic();
    t.aligned  = (iDtype == Dtype::Fp32 && oDtype == Dtype::Fp32 && featSize % 4u == 0u);
    t.sgSize   = t.sgReduce ? sg : 0u;
    t.iDtype   = iDtype;
    t.wDtype = wDtype;
    return t;
}


inline LayerNorm::LayerNorm(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
    const NodeSlot::Type wSlot = config_.coeffsAsInputs ? NodeSlot::input : NodeSlot::persistent;
    addSlot("scale", wSlot);
    addSlot("shift", wSlot);
    if (config_.acceptResidual)
        addSlot("residual", NodeSlot::input);
}

inline void LayerNorm::prepare()
{
    const Dtype oDtype = (oDtype_ == Dtype::None) ? oDefaultDtype() : oDtype_;
    const Tensor& inTen = (*this)["in0"];
    const Tensor& scaleTen = (*this)["scale"];
    const Tensor* shiftTen = hasShift() ? &(*this)["shift"] : nullptr;

    uint32_t batchSize = outerSize(inTen.shape());
    uint32_t featSize  = inTen.shape().back();

    VAI_ASSERT(inTen.shape().size() >= 2);
    // run() folds rows over grid x/y; x*y covers any int batch on spec-minimum
    // (65535 per axis) devices, so this is defense against a broken query.
    VAI_ASSERT(uint64_t(batchSize) <= uint64_t(device().maxComputeWorkGroupCount()[0])
                                      * device().maxComputeWorkGroupCount()[1]);
    // scale / shift share typeW: the loader's dtype, or the producer's under coeffsAsInputs.
    const Dtype wDtype = scaleTen.dtype();
    for (const Tensor* t : {&scaleTen, shiftTen})
    {
        if (!t)
            continue;
        VAI_ASSERT(t->dtype() == wDtype);
        VAI_ASSERT(t->shape().back() == featSize && t->numElements() == featSize);
    }

    const Dtype iDtype = inTen.dtype();
    if (config_.acceptResidual)
    {
        const Tensor& rTen = (*this)["residual"];
        VAI_ASSERT(rTen.shape() == inTen.shape());
        VAI_ASSERT(rTen.dtype() == iDtype);
    }

    PipeParams p = lnTuning(featSize, iDtype, wDtype, oDtype, device());
    p.hasShift = shiftTen != nullptr;
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        pipe =
            pipelinePool().request("layer-norm",
                                   {
                                       eva::constant_id<0>(p.wgSize),                           // wgSize
                                       eva::constant_id<1>(p.sgReduce ? 1u : 0u),               // sgReduce
                                       eva::constant_id<2>(p.hasShift ? 1u : 0u),               // hasShift
                                       eva::constant_id<3>(config_.acceptResidual ? 1u : 0u),   // hasResidual
                                       eva::constant_id<SPEC_ID_TYPE_O>(oDtype),                // typeO
                                       eva::constant_id<SPEC_ID_TYPE_I>(iDtype),                // typeI
                                       eva::constant_id<SPEC_ID_TYPE_W>(wDtype),                // typeW (scale & shift)
                                       eva::constant_id<SPEC_ID_ALIGNED>(p.aligned ? 1u : 0u)   // aligned (vec4 loads)
                                   },
                                   {.requiredSubgroupSize = p.sgSize});
        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }

    (*this)["out0"] = Tensor(inTen.shape(), oDtype);
}

inline void LayerNorm::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf = (*this)["in0"].buffer();
    auto scaleBuf = (*this)["scale"].buffer();
    auto shiftBuf = pipeParams.hasShift ? (*this)["shift"].buffer() : eva::Buffer();
    auto rBuf = config_.acceptResidual ? (*this)["residual"].buffer() : inBuf;

    descSet.write({outBuf, inBuf, scaleBuf, shiftBuf, rBuf});

    const auto& shape = (*this)["in0"].shape();
    uint32_t batchSize = outerSize(shape);
    uint32_t featSize  = shape.back();

    // One workgroup per row, rows folded over grid x/y — x alone caps at 65535
    // on some devices (Intel), below a long prefill's row count. gy stays 1
    // (gx = batchSize, today's geometry) until the batch outgrows the x limit.
    const uint32_t limX = device().maxComputeWorkGroupCount()[0];
    const uint32_t gy   = std::max(1u, CEIL_DIV(batchSize, limX));
    const uint32_t gx   = CEIL_DIV(batchSize, gy);

    struct {
        uint32_t batchSize, featSize;
        float eps;
        uint32_t gridX;
    } constants = { batchSize, featSize, config_.eps, gx };

    cmdBuff
        .bindPipeline(pipe)
        .setPushConstants(0, sizeof(constants), &constants)
        .bindDescSets({descSet})
        .dispatch(gx, gy)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace vai::node

#endif // VAI_NODES_LAYER_NORM_H
