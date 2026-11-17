#ifndef EXPERIMENTAL_DIFFUSION_POLICY_DDPM_SCHEDULER_H
#define EXPERIMENTAL_DIFFUSION_POLICY_DDPM_SCHEDULER_H

#include "vai/engine/error.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>


namespace model::experimental::diffusion_policy {


enum class BetaSchedule { Linear, SquaredCosCapV2 };
enum class PredictionType : uint32_t { Epsilon, Sample };


// diffusers' DDPMScheduler as lerobot configures it (variance_type "fixed_small",
// timestep_spacing "leading").
struct DdpmConfig
{
    uint32_t numTrainTimesteps = 100;
    BetaSchedule betaSchedule = BetaSchedule::SquaredCosCapV2;
    float betaStart = 0.0001f;   // Linear only
    float betaEnd = 0.02f;
    bool clipSample = true;
    float clipSampleRange = 1.0f;
    PredictionType predictionType = PredictionType::Epsilon;
};


/////////////////////////////////////////////////////////////////////////////////////////
// DdpmSchedule - the per-step values of a denoising loop, step k = 0 .. steps-1
//
// timesteps[k] is the k-th timestep the loop visits, from the last down to 0
// (set_timesteps: (arange(steps) * ratio)[::-1], ratio = train / steps). rows[k]
// is what DdpmStep reads for it:
//
//   { k, sqrt(ᾱ_t), sqrt(1 − ᾱ_t), c_x0, c_xt, σ_t }
//   x0   = (x − sqrt(1 − ᾱ_t) · ε) / sqrt(ᾱ_t), clipped
//   prev = c_x0 · x0 + c_xt · x + σ_t · z
//   c_x0 = sqrt(ᾱ_prev) · β'_t / (1 − ᾱ_t),  c_xt = sqrt(α'_t) · (1 − ᾱ_prev) / (1 − ᾱ_t)
//   σ_t  = sqrt(max((1 − ᾱ_prev) / (1 − ᾱ_t) · β'_t, 1e-20)), 0 at t = 0
//
// with α'_t = ᾱ_t / ᾱ_prev, β'_t = 1 − α'_t, ᾱ_prev = ᾱ_{t − ratio} (1 past the
// start). The betas and their cumulative product are fp32, as diffusers keeps them.
/////////////////////////////////////////////////////////////////////////////////////////
struct DdpmSchedule
{
    static constexpr uint32_t rowWidth = 6;

    std::vector<float> timesteps;   // [steps]
    std::vector<float> rows;        // [steps, rowWidth]

    static DdpmSchedule make(const DdpmConfig& cfg, uint32_t steps)
    {
        const uint32_t N = cfg.numTrainTimesteps;
        VAI_ASSERT(steps >= 1u && steps <= N);

        std::vector<float> betas(N);
        if (cfg.betaSchedule == BetaSchedule::Linear)
        {
            for (uint32_t i = 0; i < N; ++i)
                betas[i] = float(cfg.betaStart + (double(cfg.betaEnd) - cfg.betaStart) * i / std::max(1u, N - 1u));
        }
        else
        {
            // betas_for_alpha_bar, cosine: computed in double, stored fp32.
            auto alphaBar = [](double t) {
                const double c = std::cos((t + 0.008) / 1.008 * M_PI / 2.0);
                return c * c;
            };
            for (uint32_t i = 0; i < N; ++i)
                betas[i] = float(std::min(1.0 - alphaBar(double(i + 1) / N) / alphaBar(double(i) / N), 0.999));
        }

        std::vector<float> alphasCumprod(N);
        float prod = 1.0f;
        for (uint32_t i = 0; i < N; ++i)
            alphasCumprod[i] = prod = prod * (1.0f - betas[i]);

        const uint32_t ratio = N / steps;
        DdpmSchedule s;
        s.timesteps.resize(steps);
        s.rows.resize(size_t(steps) * rowWidth);
        for (uint32_t k = 0; k < steps; ++k)
        {
            const int32_t t = int32_t((steps - 1u - k) * ratio);
            const int32_t prevT = t - int32_t(ratio);
            const float aT = alphasCumprod[t];
            const float aPrev = prevT >= 0 ? alphasCumprod[prevT] : 1.0f;
            const float bT = 1.0f - aT;
            const float bPrev = 1.0f - aPrev;
            const float curAlpha = aT / aPrev;
            const float curBeta = 1.0f - curAlpha;

            const float variance = std::max(bPrev / bT * curBeta, 1e-20f);
            float* row = &s.rows[size_t(k) * rowWidth];
            row[0] = float(k);
            row[1] = std::sqrt(aT);
            row[2] = std::sqrt(bT);
            row[3] = std::sqrt(aPrev) * curBeta / bT;
            row[4] = std::sqrt(curAlpha) * bPrev / bT;
            row[5] = t > 0 ? std::sqrt(variance) : 0.0f;
            s.timesteps[k] = float(t);
        }
        return s;
    }
};


}   // namespace model::experimental::diffusion_policy

#endif   // EXPERIMENTAL_DIFFUSION_POLICY_DDPM_SCHEDULER_H
