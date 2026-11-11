#ifndef VAI_NODES_GATED_MUL_H
#define VAI_NODES_GATED_MUL_H

#include "node-common.h"

namespace vai::node {


/*=============================================================================
  GatedMul - element-wise multiply with one operand activated (GLU gating)
  Input : in0  [..., 2H]
  Output: out0 [..., H] = activation(in0[..., j]) * in0[..., H + j]

  The two halves arrive in ONE buffer: the projections feeding them are a
  single Linear over the concatenated weights, so the leading half is whatever
  that concatenation put first. It is the activated one.

  activation names the GLU variant — silu: SwiGLU, gelu: GEGLU, relu: ReGLU,
  none: bilinear.
=============================================================================*/
class GatedMul : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        vai::Epilogue activation = vai::Epilogue::Silu;
    } config_;

    GatedMul();
    GatedMul(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};


inline GatedMul::GatedMul() : GatedMul(Config()) {}

inline GatedMul::GatedMul(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
}

inline void GatedMul::prepare()
{
    const Dtype oDtype = (oDtype_ == Dtype::None) ? oDefaultDtype() : oDtype_;
    const Tensor& inTen = (*this)["in0"];

    const Dtype iDtype = inTen.dtype();

    const PipeParams p = { .iDtype = iDtype };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        pipe = pipelinePool().request("gated-mul",
                                      {
                                          eva::constant_id<SPEC_ID_TYPE_O>(oDtype),                          // typeO
                                          eva::constant_id<SPEC_ID_TYPE_I>(iDtype),                          // typeI
                                          eva::constant_id<SPEC_ID_EPILOGUE>((uint32_t)config_.activation)   // epilogue
                                      });
        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }

    VAI_ASSERT(!inTen.shape().empty());
    VAI_ASSERT(inTen.shape().back() % 2 == 0);

    std::vector<uint32_t> outShape = inTen.shape();
    outShape.back() /= 2;
    (*this)["out0"] = Tensor(outShape, oDtype);
}

inline void GatedMul::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf  = (*this)["in0"].buffer();

    descSet.write({ outBuf, inBuf });

    // One thread per output element, workgroups folded over grid x/y — x alone
    // caps at 65535 on some devices (Intel), below what a long prefill needs.
    constexpr uint32_t wgSize = 64;   // = local_size_x of gated-mul
    const uint32_t elems  = uint32_t((*this)["out0"].numElements());
    const uint32_t groups = CEIL_DIV(elems, wgSize);
    const uint32_t limX   = device().maxComputeWorkGroupCount()[0];
    const uint32_t gy     = std::max(1u, CEIL_DIV(groups, limX));
    const uint32_t gx     = CEIL_DIV(groups, gy);

    struct {
        uint32_t N;       // output elements
        uint32_t H;       // output width = half the input's
        uint32_t gridX;   // workgroups along x, so the shader can undo the fold
    } constants = { elems, (*this)["out0"].shape().back(), gx };

    cmdBuff
        .bindPipeline(pipe)
        .setPushConstants(0, sizeof(constants), &constants)
        .bindDescSets({descSet})
        .dispatch(gx, gy)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace vai::node

#endif // VAI_NODES_GATED_MUL_H
