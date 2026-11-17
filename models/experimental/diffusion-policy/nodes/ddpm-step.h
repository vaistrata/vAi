#ifndef EXPERIMENTAL_DIFFUSION_POLICY_NODES_DDPM_STEP_H
#define EXPERIMENTAL_DIFFUSION_POLICY_NODES_DDPM_STEP_H

#include "vai/nodes/common.h"
#include "../ddpm-scheduler.h"

namespace model::experimental::diffusion_policy::node {


/*=============================================================================
  DdpmStep - one DDPM denoising step (diffusers' DDPMScheduler.step)
  Input : in0         [...] the sample x_t
          modelOutput [...] the UNet's prediction for it: ε, or x0 (PredictionType)
          schedule    [6]   the step's DdpmSchedule row (a CyclicTableReader's)
          noise       [steps, ...] Fp32, the loop's noise; row k for step k
  Output: out0        [...] x_{t−1}

  x0   = (x − sqrt(1 − ᾱ_t) · ε) / sqrt(ᾱ_t), clamped to ±clipRange under clipSample
  out0 = c_x0 · x0 + c_xt · x + σ_t · noise[k]

  What a pi0.5-style loop folds into a residual Linear, for a sampler that is
  not one: the x0 clip and the noise sit between the model and the update.
  Every coefficient is the schedule row's, so the loop reads them per step with
  no host write; the step index k in it picks the noise row. Element-wise, so
  out0 may share x's buffer.
=============================================================================*/
class DdpmStep : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;
        vai::Dtype eDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        PredictionType predictionType = PredictionType::Epsilon;
        bool clipSample = true;
        float clipSampleRange = 1.0f;
    } config_;

    DdpmStep(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};


inline DdpmStep::DdpmStep(Config config)
: config_(config)
{
    addSlot("out0", vai::NodeSlot::output);
    addSlot("in0", vai::NodeSlot::input);
    addSlot("modelOutput", vai::NodeSlot::input);
    addSlot("schedule", vai::NodeSlot::input);
    addSlot("noise", vai::NodeSlot::input);
}

inline void DdpmStep::prepare()
{
    const vai::Dtype oDtype = (oDtype_ == vai::Dtype::None) ? oDefaultDtype() : oDtype_;
    const vai::Tensor& xTen = (*this)["in0"];
    const vai::Tensor& eTen = (*this)["modelOutput"];
    const vai::Tensor& sTen = (*this)["schedule"];
    const vai::Tensor& zTen = (*this)["noise"];
    VAI_ASSERT(xTen.validShape() && eTen.numElements() == xTen.numElements());
    VAI_ASSERT(sTen.numElements() == DdpmSchedule::rowWidth && sTen.dtype() == vai::Dtype::Fp32);
    VAI_ASSERT(zTen.dtype() == vai::Dtype::Fp32 && zTen.numElements() % xTen.numElements() == 0u);

    const PipeParams p = { .iDtype = xTen.dtype(), .eDtype = eTen.dtype() };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        // clang-format off
        pipe = pipelinePool().request(EXPERIMENTAL_DIFFUSION_POLICY_SHADER_OUTPUT_DIR, "ddpm-step", {
            eva::constant_id<0>(config_.predictionType == PredictionType::Sample ? 1u : 0u),   // predictSample
            eva::constant_id<1>(config_.clipSample ? 1u : 0u),                              // clipSample
            eva::constant_id<2>(p.eDtype),                                                  // typeE
            eva::constant_id<SPEC_ID_TYPE_O>(oDtype),                                       // typeO
            eva::constant_id<SPEC_ID_TYPE_I>(p.iDtype)                                      // typeI
        });
        // clang-format on
        if (!descSet)
            descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
    }

    (*this)["out0"] = vai::Tensor(xTen.shape(), oDtype);
}

inline void DdpmStep::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();

    descSet.write({
        outBuf,
        (*this)["in0"].buffer(),
        (*this)["modelOutput"].buffer(),
        (*this)["schedule"].buffer(),
        (*this)["noise"].buffer(),
    });

    struct {
        uint32_t N;
        float clipRange;
    } constants = { uint32_t((*this)["in0"].numElements()), config_.clipSampleRange };

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch2(constants.N)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace model::experimental::diffusion_policy::node

#endif // EXPERIMENTAL_DIFFUSION_POLICY_NODES_DDPM_STEP_H
