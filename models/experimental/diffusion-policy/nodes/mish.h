#ifndef EXPERIMENTAL_DIFFUSION_POLICY_NODES_MISH_H
#define EXPERIMENTAL_DIFFUSION_POLICY_NODES_MISH_H

#include "vai/nodes/common.h"

namespace model::experimental::diffusion_policy::node {


/*=============================================================================
  Mish - element-wise Mish (torch.nn.Mish)
  Input : in0  [any shape]
  Output: out0 [same shape] = in0 · tanh(softplus(in0)); optionally in-place

  vai::ActType has no Mish, so the diffusion UNet's standalone ones (its step
  encoder's, and the Mish its condition passes through) run here; the one after
  each GroupNorm is fused into GroupNormMish. Math is fp32; softplus switches to
  the identity past 20, as torch's does. The inPlace option aliases the output
  to the input and requires matching input and output dtypes.
=============================================================================*/
class Mish : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        bool inPlace = false;
    } config_;

    Mish();
    Mish(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};


inline Mish::Mish()
: Mish(Config())
{
}

inline Mish::Mish(Config config)
: config_(config)
{
    addSlot("out0", vai::NodeSlot::output);
    addSlot("in0", vai::NodeSlot::input);
}

inline void Mish::prepare()
{
    const vai::Dtype oDtype = (oDtype_ == vai::Dtype::None) ? oDefaultDtype() : oDtype_;
    const vai::Tensor& inTen = (*this)["in0"];
    VAI_ASSERT(inTen.validShape());

    const vai::Dtype iDtype = inTen.dtype();
    if (config_.inPlace)   // in-place reads and writes the same buffer
        VAI_ASSERT(oDtype == iDtype);

    const PipeParams p = { .iDtype = iDtype };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        // clang-format off
        pipe = pipelinePool().request(EXPERIMENTAL_DIFFUSION_POLICY_SHADER_OUTPUT_DIR, "mish", {
            eva::constant_id<0>(config_.inPlace ? 1u : 0u),   // inPlace
            eva::constant_id<SPEC_ID_TYPE_O>(oDtype),         // typeO
            eva::constant_id<SPEC_ID_TYPE_I>(iDtype)          // typeI
        });
        // clang-format on
        if (!descSet)
            descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
    }

    (*this)["out0"] = config_.inPlace ? inTen : vai::Tensor(inTen.shape(), oDtype);
}

inline void Mish::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf = config_.inPlace ? eva::Buffer() : (*this)["in0"].buffer();

    descSet.write({ outBuf, inBuf });

    const uint32_t N = uint32_t((*this)["out0"].numElements());

    struct {
        uint32_t N;
    } constants = { N };

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch2(N)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace model::experimental::diffusion_policy::node

#endif // EXPERIMENTAL_DIFFUSION_POLICY_NODES_MISH_H
