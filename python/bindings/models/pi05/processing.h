#ifndef PYVAI_MODELS_PI05_PROCESSING_H
#define PYVAI_MODELS_PI05_PROCESSING_H

// Verbatim copy of makePrompt() from models/production/pi05/processing.h, bound as
// pi05_make_prompt. Its sampleNoise() is pi0's, already bound as
// pi0_sample_noise. Re-copy when the source changes.

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace pyvai::examples {
namespace {

/////////////////////////////////////////////////////////////////////////////////////////
// Prompt - "Task: {task}, State: {bins};\nAction: ". Each state value, already in
// [-1, 1], is cut into one of 256 equal bins (numpy.digitize over the 256 left
// edges, minus one), so the language model reads the state as integers. Below -1
// numpy gives -1, and so does this.
/////////////////////////////////////////////////////////////////////////////////////////
std::string
makePrompt(const std::string& task, const std::vector<float>& normalizedState)
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

}   // namespace
}   // namespace pyvai::examples

#endif // PYVAI_MODELS_PI05_PROCESSING_H
