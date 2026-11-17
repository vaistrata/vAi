#ifndef EXPERIMENTAL_DIFFUSION_POLICY_NODES_SINUSOIDAL_POS_EMB_H
#define EXPERIMENTAL_DIFFUSION_POLICY_NODES_SINUSOIDAL_POS_EMB_H

#include "vai/nodes/common.h"

namespace model::experimental::diffusion_policy::node {


/*=============================================================================
  SinusoidalPosEmb - a timestep's sinusoidal embedding (lerobot's
  DiffusionSinusoidalPosEmb)
  Input : in0  [B] timesteps
  Output: out0 [B, dim] = cat(sin(t · f), cos(t · f)),
          f_i = exp(-i · log(10000) / (dim/2 - 1)), i < dim/2

  The diffusion step encoder's first stage. The timestep is an input tensor
  rewritten per denoising step, so the step loop needs no re-recording. The
  argument is formed in fp32 as torch does, then range-reduced before sin / cos.
=============================================================================*/
class SinusoidalPosEmb : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        uint32_t dim;   // even, >= 4
    } config_;

    SinusoidalPosEmb(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};


inline SinusoidalPosEmb::SinusoidalPosEmb(Config config)
: config_(config)
{
    VAI_ASSERT(config_.dim >= 4u && config_.dim % 2u == 0u);
    addSlot("out0", vai::NodeSlot::output);
    addSlot("in0", vai::NodeSlot::input);
}

inline void SinusoidalPosEmb::prepare()
{
    const vai::Dtype oDtype = (oDtype_ == vai::Dtype::None) ? oDefaultDtype() : oDtype_;
    const vai::Tensor& inTen = (*this)["in0"];
    VAI_ASSERT(inTen.validShape() && inTen.shape().size() == 1u);

    const PipeParams p = { .iDtype = inTen.dtype() };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        // clang-format off
        pipe = pipelinePool().request(EXPERIMENTAL_DIFFUSION_POLICY_SHADER_OUTPUT_DIR, "sinusoidal-pos-emb", {
            eva::constant_id<SPEC_ID_TYPE_O>(oDtype),     // typeO
            eva::constant_id<SPEC_ID_TYPE_I>(p.iDtype)    // typeI
        });
        // clang-format on
        if (!descSet)
            descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
    }

    (*this)["out0"] = vai::Tensor({ inTen.shape()[0], config_.dim }, oDtype);
}

inline void SinusoidalPosEmb::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf = (*this)["in0"].buffer();

    descSet.write({ outBuf, inBuf });

    struct {
        uint32_t rows, dim;
    } constants = { (*this)["in0"].shape()[0], config_.dim };

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch2(constants.rows * constants.dim)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace model::experimental::diffusion_policy::node

#endif // EXPERIMENTAL_DIFFUSION_POLICY_NODES_SINUSOIDAL_POS_EMB_H
