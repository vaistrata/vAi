#ifndef VAI_NODES_GROUP_NORM_H
#define VAI_NODES_GROUP_NORM_H

#include "node-common.h"

namespace vai::node {


/*=============================================================================
  GroupNorm - group normalization over a CHW feature map
  Input:  in0 [..., C, H, W], optional residual [..., C, H, W]
  Output: epilogue(scale[c] * (in0 - μ[g(c)]) * rsqrt(σ²[g(c)] + eps) + shift[c]) + residual

  The channel axis is third from the end; the axes before it are independent
  samples. scale / shift [C] are parameters, or inputs under coeffsAsInputs
  (e.g. AdaGN's timestep modulation); shift applies when present, filled by
  the loader or wired. acceptResidual opens the "residual" input (input
  dtype), added after the epilogue or, with ResidualOrder::BeforeEpilogue,
  before it. The channels split into `groups` consecutive groups; each group's statistics
  are computed at run time over its C/groups channels and all of their spatial
  positions (unlike BatchNorm, which reads μ/σ² trained offline). One
  workgroup per group (group-norm), statistics from a shifted 1-pass
  reduction accumulated in fp32 regardless of I/O dtypes — the reduction is
  subgroupAdd two-level when the device has subgroup arithmetic, a
  shared-memory tree otherwise.

  Both the workgroup size and the shape itself are baked into the pipeline, so
  it is requested in prepare() once the shape is known and re-requested when
  any of that changes (a no-op while it stays put) — same arrangement as
  LayerNorm. The shape is in there because the kernel's per-element
  channel index is j / HW: a push-constant HW costs an integer division per
  element, a specialization-constant one is a shift or a multiply-high. That
  makes the pipeline key per (dtype, group geometry), which is a handful of
  entries for a model — a ResNet-18 stage or a UNet1D level each contribute
  one, and every block inside it reuses that.
=============================================================================*/
class GroupNorm : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct GnTuningParams
    {
        uint32_t wgSize = 0;         // threads per workgroup (= local_size_x)
        uint32_t sgSize = 0;         // forced subgroup size (0 = don't force; Intel)
        uint32_t numSubgroups = 0;   // wgSize / sgSize, exact because sgSize is pinned (0: shared tree)
        uint32_t chPerGroup = 0;     // C / groups
        uint32_t HW = 0;             // spatial elements per channel
        bool sgReduce = false;       // subgroupAdd two-level reduction instead of the shared tree
        bool aligned = false;        // vec4 I/O: fp32 in and out, and HW % 4 == 0
        bool cached = false;         // aligned, and one quad per thread covers the group
        vai::Dtype iDtype = vai::Dtype::None;
        bool hasShift = false;

        bool operator==(const GnTuningParams&) const = default;
    };

    struct Config
    {
        uint32_t groups;
        float eps = 1e-5f;
        bool coeffsAsInputs = false;   // scale / shift are inputs, not parameters
        bool acceptResidual = false;   // input "residual"
        vai::ResidualOrder residualOrder = vai::ResidualOrder::AfterEpilogue;
        vai::Epilogue epilogue = vai::Epilogue::None;
    } config_;

    GroupNorm(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

    // The loader filled it, or under coeffsAsInputs it is wired.
    bool hasShift() const { return (*this)["shift"].validShape(); }

private:
    GnTuningParams gnParams;   // cached selection; pipeline re-requested on change
    uint32_t samples_ = 1;     // the input's axes before C, multiplied
};


// Workgroup geometry / kernel variant from device queries + shape:
// wgSize = pow2ceil(groupSize / 4) in [sg, min(16 * sg, 1024)], subgroupAdd
// when supported (subgroup size pinned — Intel varies it otherwise). wgSize
// stays a power of two, which the shared-tree reduction's halving loop needs:
// sg is one by the Vulkan spec, and doubling keeps it one.
inline GroupNorm::GnTuningParams gnTuning(uint32_t chPerGroup, uint32_t HW, Dtype iDtype, Dtype oDtype,
                                          eva::Device device)
{
    const uint32_t groupSize = chPerGroup * HW;
    const uint32_t sg = device.maxSubgroupSize();

    // TODO (eva): maxComputeWorkGroupInvocations limit query — until then
    //             cap at 1024 (common desktop value; spec guarantees 128)
    const uint32_t cap = std::min(16u * sg, 1024u);
    uint32_t wg = sg;
    while (wg * 4u < groupSize && wg < cap)
        wg *= 2u;   // pow2ceil(groupSize / 4), [sg, cap]

    GroupNorm::GnTuningParams t;
    t.wgSize = wg;
    t.sgReduce = device.supportsSubgroupArithmetic();
    t.sgSize = t.sgReduce ? sg : 0u;
    t.numSubgroups = t.sgReduce ? wg / sg : 0u;
    t.chPerGroup = chPerGroup;
    t.HW = HW;
    // HW % 4 == 0 is both conditions at once: it makes F a multiple of 4 (so
    // base = g * F stays quad-aligned) and keeps a quad inside one channel,
    // which is what lets a quad share one scale / shift lookup.
    t.aligned = (iDtype == Dtype::Fp32 && oDtype == Dtype::Fp32 && HW % 4u == 0u);
    // One quad per thread means the kernel can hold the group in registers
    // across the reduction and read the input once. wgSize targets ~4 elements
    // per thread, so this is the common case, not a corner of it.
    t.cached = t.aligned && (groupSize / 4u <= wg);
    t.iDtype = iDtype;
    return t;
}


inline GroupNorm::GroupNorm(Config config)
: config_(config)
{
    VAI_ASSERT(config_.groups != 0u);
    VAI_ASSERT(config_.epilogue < Epilogue::Max);

    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
    const NodeSlot::Type wSlot = config_.coeffsAsInputs ? NodeSlot::input : NodeSlot::persistent;
    addSlot("scale", wSlot);
    addSlot("shift", wSlot);
    if (config_.acceptResidual)
        addSlot("residual", NodeSlot::input);
}

inline void GroupNorm::prepare()
{
    const Dtype oDtype = (oDtype_ == Dtype::None) ? oDefaultDtype() : oDtype_;
    const Tensor& inTen = (*this)["in0"];
    const Tensor& scaleTen = (*this)["scale"];
    const Tensor* shiftTen = hasShift() ? &(*this)["shift"] : nullptr;

    const auto& shape = inTen.shape();
    const size_t rank = shape.size();
    VAI_ASSERT(rank >= 3u);
    const uint32_t C = shape[rank - 3];
    const uint32_t HW = shape[rank - 2] * shape[rank - 1];
    samples_ = 1u;
    for (size_t axis = 0; axis + 3 < rank; ++axis)
        samples_ *= shape[axis];
    VAI_ASSERT(C % config_.groups == 0u);

    const Dtype iDtype = inTen.dtype();
    // scale / shift share typeW: the loader's dtype, or the producer's under coeffsAsInputs.
    const Dtype wDtype = scaleTen.dtype();
    VAI_ASSERT(scaleTen.isShapeOf(C));
    VAI_ASSERT(!shiftTen || shiftTen->dtype() == wDtype);
    VAI_ASSERT(!shiftTen || shiftTen->isShapeOf(C));
    if (config_.acceptResidual)
    {
        const Tensor& rTen = (*this)["residual"];
        VAI_ASSERT(rTen.shape() == shape);
        VAI_ASSERT(rTen.dtype() == iDtype);
    }

    const uint32_t chPerGroup = C / config_.groups;

    GnTuningParams t = gnTuning(chPerGroup, HW, iDtype, oDtype, device());
    t.hasShift = shiftTen != nullptr;
    if (!pipe || !(t == gnParams))
    {
        gnParams = t;

        // The pool keys the pipeline on the shader name, these constants and
        // the config, so nothing here has to be repeated in a tag.
        // clang-format off
        pipe = pipelinePool().request("group-norm", {
            eva::constant_id<0>(t.wgSize),                          // wgSize
            eva::constant_id<1>(t.sgReduce ? 1u : 0u),              // sgReduce
            eva::constant_id<2>(t.hasShift ? 1u : 0u),              // hasShift
            eva::constant_id<3>(t.chPerGroup),                      // chPerGroup
            eva::constant_id<4>(t.HW),                              // HW
            eva::constant_id<5>(t.numSubgroups),                    // numSubgroups
            eva::constant_id<6>(t.cached ? 1u : 0u),                // cached
            eva::constant_id<7>(config_.groups),                    // groups
            eva::constant_id<8>(config_.acceptResidual ? 1u : 0u),  // hasResidual
            eva::constant_id<9>((uint32_t)config_.residualOrder),   // residualOrder
            eva::constant_id<SPEC_ID_EPILOGUE>((uint32_t)config_.epilogue),
            eva::constant_id<SPEC_ID_TYPE_O>(oDtype),       // typeO
            eva::constant_id<SPEC_ID_TYPE_I>(iDtype),               // typeI
            eva::constant_id<SPEC_ID_TYPE_W>(wDtype),               // typeW (scale & shift)
            eva::constant_id<SPEC_ID_ALIGNED>(t.aligned ? 1u : 0u)  // vec4 I/O
        }, { .requiredSubgroupSize = t.sgSize });
        // clang-format on

        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }

    (*this)["out0"] = Tensor(inTen.shape(), oDtype);
}

inline void GroupNorm::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf = (*this)["in0"].buffer();
    auto scaleBuf = (*this)["scale"].buffer();
    auto shiftBuf = gnParams.hasShift ? (*this)["shift"].buffer() : eva::Buffer();
    auto rBuf = config_.acceptResidual ? (*this)["residual"].buffer() : inBuf;

    descSet.write({ outBuf, inBuf, scaleBuf, shiftBuf, rBuf });

    // The shape is a specialization constant, so eps is all that is left here.
    struct
    {
        float eps;
    } constants = { config_.eps };

    cmdBuff.bindPipeline(pipe)
        .bindDescSets({ descSet })
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch(samples_ * config_.groups)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace vai::node

#endif   // VAI_NODES_GROUP_NORM_H
