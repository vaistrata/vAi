#ifndef EXPERIMENTAL_DIFFUSION_POLICY_PROCESSING_H
#define EXPERIMENTAL_DIFFUSION_POLICY_PROCESSING_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

namespace model::experimental::diffusion_policy {


/////////////////////////////////////////////////////////////////////////////////////////
// Diffusion noise - splitmix64 through Box-Muller, as pi0.5 draws its own. A sampled
// action is only as reproducible as the noise it started from, so the seed is the
// caller's; std::normal_distribution would tie it to the standard library too.
/////////////////////////////////////////////////////////////////////////////////////////
inline std::vector<float> sampleNoise(size_t count, uint64_t seed)
{
    auto next = [state = seed]() mutable {
        state += 0x9E3779B97F4A7C15ull;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        return (z >> 11) * 0x1.0p-53;   // [0, 1), 53 significant bits
    };

    std::vector<float> noise(count);
    for (size_t i = 0; i < count; i += 2)
    {
        const double u1 = std::max(next(), 1e-12);   // log(0) is the one input Box-Muller rejects
        const double u2 = next();
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double theta = 2.0 * std::numbers::pi_v<double> * u2;

        noise[i] = float(r * std::cos(theta));
        if (i + 1 < count)
            noise[i + 1] = float(r * std::sin(theta));
    }

    return noise;
}


/////////////////////////////////////////////////////////////////////////////////////////
// MIN_MAX normalization - LeRobot maps a feature onto [-1, 1]:
//   x' = (x − min) / (max − min + 1e-8) · 2 − 1,   x = (x' + 1) / 2 · (max − min) + min
// values is [rows, dim], one feature vector per row; min / max are [dim].
/////////////////////////////////////////////////////////////////////////////////////////
inline std::vector<float> normalizeMinMax(std::vector<float> values, const std::vector<float>& min,
                                          const std::vector<float>& max)
{
    const size_t dim = min.size();
    for (size_t i = 0; i < values.size(); ++i)
    {
        const size_t d = i % dim;
        values[i] = (values[i] - min[d]) / (max[d] - min[d] + 1e-8f) * 2.0f - 1.0f;
    }
    return values;
}

inline std::vector<float> unnormalizeMinMax(std::vector<float> values, const std::vector<float>& min,
                                            const std::vector<float>& max)
{
    const size_t dim = min.size();
    for (size_t i = 0; i < values.size(); ++i)
    {
        const size_t d = i % dim;
        values[i] = (values[i] + 1.0f) / 2.0f * (max[d] - min[d]) + min[d];
    }
    return values;
}


/////////////////////////////////////////////////////////////////////////////////////////
// Executed actions - lerobot's generate_actions: of the denoised sample [horizon, dim],
// the rows start .. start + count (start = n_obs_steps − 1, the current step), still
// normalized.
/////////////////////////////////////////////////////////////////////////////////////////
inline std::vector<float> actionWindow(const float* sample, uint32_t dim, uint32_t start, uint32_t count)
{
    return std::vector<float>(sample + size_t(start) * dim, sample + size_t(start + count) * dim);
}

}   // namespace model::experimental::diffusion_policy

#endif   // EXPERIMENTAL_DIFFUSION_POLICY_PROCESSING_H
