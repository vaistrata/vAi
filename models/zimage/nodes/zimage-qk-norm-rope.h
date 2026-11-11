#ifndef ZIMAGE_QK_NORM_ROPE_H
#define ZIMAGE_QK_NORM_ROPE_H

#include "vai/nodes/node-common.h"

namespace zimage {

/*=============================================================================
  QkNormRopeNode - per-head RMS norm + interleaved-pair RoPE, head-major store
  Input : in0 [N, H*Dh] (token-major projection), cs [N, Dh/2, 2] (cos, sin)
  Params: gamma [Dh] (when hasNorm)
  Output: out0 [H, N, Dh] = rope(rmsnorm(in0[n, h]) * γ, cs[n])

  The rotation pairs ADJACENT head dims (2k, 2k+1) as one complex number, the
  layout Z-Image's view_as_complex RoPE uses; vai::node::Rope rotates the
  half-split pairs instead. The per-pair angles come in through cs, built on
  the host from each token's 3-axis position ids (ropeTable() in model.h),
  so one node serves the image, caption and unified sequences alike.

  The split to heads, the QK norm, the rotation and the [N, H, Dh] ->
  [H, N, Dh] permute that FullAttentionNode wants are one kernel.
=============================================================================*/
class QkNormRopeNode : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        uint32_t numHeads;
        uint32_t headDim;           // Dh / 2 must be a power of two
        bool hasNorm = true;
        float eps = 1e-5f;
        vai::Dtype wDtype = vai::Dtype::None;   // None: wDefaultDtype
        vai::Dtype oDtype = vai::Dtype::None;   // None: oDefaultDtype
    } config_;

    QkNormRopeNode(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};


inline QkNormRopeNode::QkNormRopeNode(Config config)
: config_(config)
{
    addSlot("out0", vai::NodeSlot::output);
    addSlot("in0", vai::NodeSlot::input);
    addSlot("cs", vai::NodeSlot::input);
    if (config_.hasNorm) addSlot("gamma", vai::NodeSlot::persistent);

    const uint32_t pairs = config_.headDim / 2;
    VAI_ASSERT(config_.headDim % 2 == 0 && (pairs & (pairs - 1)) == 0 && pairs <= 1024);

    config_.wDtype = (config_.wDtype == vai::Dtype::None) ? wDefaultDtype() : config_.wDtype;
    config_.oDtype = (config_.oDtype == vai::Dtype::None) ? oDefaultDtype() : config_.oDtype;
}

inline void QkNormRopeNode::prepare()
{
    const uint32_t H = config_.numHeads, Dh = config_.headDim;
    const vai::Tensor& inTen = (*this)["in0"];
    const vai::Tensor& csTen = (*this)["cs"];
    const vai::Dtype iDtype = inTen.dtype();

    VAI_ASSERT(inTen.isShapeOf(-1, H * Dh));
    const uint32_t N = inTen.shape()[0];
    VAI_ASSERT(csTen.dtype() == iDtype);
    VAI_ASSERT(csTen.isShapeOf(N, Dh / 2, 2));
    if (config_.hasNorm)
    {
        VAI_ASSERT((*this)["gamma"].dtype() == config_.wDtype);
        VAI_ASSERT((*this)["gamma"].isShapeOf(Dh));
    }

    const PipeParams p = { .iDtype = iDtype };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        pipe = pipelinePool().request(ZIMAGE_SHADER_OUTPUT_DIR, "zimage-qk-norm-rope", {
            eva::constant_id<0>(Dh / 2),                               // wgSize = Dh / 2
            eva::constant_id<1>(uint32_t(config_.hasNorm)),
            eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),          // typeO
            eva::constant_id<SPEC_ID_TYPE_I>(iDtype),                  // typeI (in0, cs)
            eva::constant_id<SPEC_ID_TYPE_W>(config_.wDtype)           // typeW (γ)
        });
        if (!descSet)
            descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
    }

    (*this)["out0"] = vai::Tensor({H, N, Dh}, config_.oDtype);
}

inline void QkNormRopeNode::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf  = (*this)["in0"].buffer();
    auto csBuf  = (*this)["cs"].buffer();
    auto gBuf   = config_.hasNorm ? (*this)["gamma"].buffer() : inBuf;   // stand-in when absent

    descSet.write({ outBuf, inBuf, csBuf, gBuf });

    struct {
        uint32_t N, H;
        float eps;
    } constants = { (*this)["in0"].shape()[0], config_.numHeads, config_.eps };

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch(constants.N, constants.H)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

} // namespace zimage

#endif // ZIMAGE_QK_NORM_ROPE_H
