#ifndef PRODUCTION_PI05_PROCESSING_H
#define PRODUCTION_PI05_PROCESSING_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <string>
#include <vector>

namespace model::production::pi05 {


/////////////////////////////////////////////////////////////////////////////////////////
// Flow-matching noise - splitmix64 through Box-Muller. A sampled chunk is only
// as reproducible as the noise it started from, so the seed is the caller's.
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
// Prompt - "Task: {task}, State: {bins};\nAction: ". Each state value, already in
// [-1, 1], is cut into one of 256 equal bins (numpy.digitize over the 256 left
// edges, minus one), so the language model reads the state as integers. Below -1
// numpy gives -1, and so does this.
/////////////////////////////////////////////////////////////////////////////////////////
inline std::string makePrompt(const std::string& task, const std::vector<float>& normalizedState)
{
    std::string cleaned = task;
    cleaned.erase(0, cleaned.find_first_not_of(" \t\r\n"));
    cleaned.erase(cleaned.find_last_not_of(" \t\r\n") + 1);
    std::replace(cleaned.begin(), cleaned.end(), '_', ' ');
    std::replace(cleaned.begin(), cleaned.end(), '\n', ' ');

    std::string bins;
    for (float x : normalizedState)
    {
        const int bin = std::clamp(int(std::floor((x + 1.0f) * 128.0f)), -1, 255);
        if (!bins.empty())
            bins += ' ';
        bins += std::to_string(bin);
    }

    return "Task: " + cleaned + ", State: " + bins + ";\nAction: ";
}

}   // namespace model::production::pi05

#endif   // PRODUCTION_PI05_PROCESSING_H
