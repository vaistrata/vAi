/*=============================================================================
  Written by   ippx123@gmail.com
  Model        Z-Image-Turbo sampling: noise, the flow-matching Euler loop
  Modules      turboSigmas, gaussianNoise, sampleTurbo
=============================================================================*/

#ifndef ZIMAGE_SAMPLER_H
#define ZIMAGE_SAMPLER_H

#include "model.h"
#include <random>
#include <vector>

namespace zimage {

// FlowMatchEulerDiscreteScheduler with the Turbo settings: numSteps sigmas
// linspace(1, 1/numSteps) under the static shift, then a final 0; one DiT
// forward per step. 8 is the step count diffusers' ZImagePipeline example and
// ComfyUI's workflow use.
inline std::vector<float> turboSigmas(uint32_t numSteps = 8, float shift = 3.0f)
{
    std::vector<float> sigmas(numSteps + 1, 0.0f);
    for (uint32_t i = 0; i < numSteps; ++i)
    {
        const float s = (numSteps == 1) ? 1.0f
                      : 1.0f + (1.0f / numSteps - 1.0f) * float(i) / float(numSteps - 1);
        sigmas[i] = shift * s / (1.0f + (shift - 1.0f) * s);
    }
    return sigmas;
}

// A seeded standard-normal latent [inChannels, H, W]. Not torch's generator:
// the same seed gives a different image here than in diffusers.
inline std::vector<float> gaussianNoise(const ModelConfig& cfg, uint32_t H, uint32_t W, uint64_t seed)
{
    std::mt19937_64 gen(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> noise(size_t(cfg.inChannels) * H * W);
    for (float& x : noise)
        x = normal(gen);
    return noise;
}

// Runs the sampler on latent [inChannels, H, W] in place, from noise to the
// final latent. The DiT sees t = 1 - sigma and predicts -velocity. The first
// step also uploads the weights and records the graph, which later steps
// reuse. onStep(i, sigma) runs after each step.
template<class OnStep>
inline void sampleTurbo(Model& model, const vai::Tensor& capFeats, std::vector<float>& latent,
                        uint32_t H, uint32_t W, const std::vector<float>& sigmas, OnStep&& onStep)
{
    const ModelConfig& cfg = model.modelCfg;
    vai::Tensor out;
    for (size_t i = 0; i + 1 < sigmas.size(); ++i)
    {
        model(makeInputs(cfg, latent.data(), H, W, capFeats, 1.0f - sigmas[i]), {{"outPatches", &out}});
        const std::vector<float> v = unpatchify(cfg, out.getData<float>(), H, W);

        const float dt = sigmas[i + 1] - sigmas[i];
        for (size_t j = 0; j < latent.size(); ++j)
            latent[j] += dt * -v[j];
        onStep(uint32_t(i), sigmas[i]);
    }
}

} // namespace zimage

#endif // ZIMAGE_SAMPLER_H
