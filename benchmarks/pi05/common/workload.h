#ifndef BENCHMARKS_PI05_COMMON_WORKLOAD_H
#define BENCHMARKS_PI05_COMMON_WORKLOAD_H

// The fixed pi0.5 workload every bench runs, and its inputs prepared for a profile.
// The observation itself (cameras, task, state, noise seed) is common/bench-inputs.h.

#include "profile.h"
#include "processing.h"
#include "vai/utils/image-processor.h"
#include "vai/utils/tokenizer.h"
#include "bench-inputs.h"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <string>
#include <vector>

namespace bench::pi05 {

struct Workload
{
    const char* size;   // folder under data/ckpts/pi05/lerobot/ and data/benchmarks/pi05/
    uint32_t cameras = uint32_t(std::size(vlaCameras));
    uint32_t imageChannels = 3;
    uint32_t imageSize = 224;
    uint32_t maxPromptTokens = 200;
    uint32_t actionChunk = 50;
    uint32_t actionDim = 32;
    uint32_t denoiseSteps = 10;

    size_t actionCount() const { return size_t(actionChunk) * actionDim; }

    // data/benchmarks/pi05/<size>/: the converted models and the reference outputs.
    std::filesystem::path dataDir() const { return std::filesystem::path(VAI_DATA_DIR) / "benchmarks/pi05" / size; }
};


inline constexpr Workload workloads[] = {
    { .size = "3B" },
};


inline const Workload* findWorkload(const char* size)
{
    for (const Workload& workload : workloads)
        if (!strcmp(workload.size, size))
            return &workload;
    return nullptr;
}


// One prediction's host inputs, in the profile's convention.
struct Inputs
{
    std::vector<float> pixels;   // cameras x image, in the profile's layout and range
    std::vector<float> state;    // normalized, actionDim wide
    std::vector<int> tokens;     // the prompt with the discretized state
    std::vector<float> noise;    // [actionChunk, actionDim]
};


inline Inputs prepareInputs(const Workload& workload, const ProfileSpec& profile)
{
    namespace processing = model::production::pi05;
    using vai::utils::Align;
    using vai::utils::ImageProcessor;

    Inputs inputs;
    const uint32_t S = workload.imageSize;
    inputs.pixels.reserve(size_t(workload.cameras) * workload.imageChannels * S * S);
    for (const char* camera : vlaCameras)
    {
        // clang-format off
        ImageProcessor image = ImageProcessor(std::string(VAI_DATA_DIR "/") + camera)
            .letterbox(S, Align::Center, 0.0f)
            .normalize(profile.imageMin, profile.imageMax);
        // clang-format on
        if (profile.imageLayout == ImageLayout::CHW)
            image.chw();
        const std::vector<float>& pixels = image;
        inputs.pixels.insert(inputs.pixels.end(), pixels.begin(), pixels.end());
    }

    inputs.state = makeVLAState(workload.actionDim);
    vai::utils::Tokenizer tokenizer(VAI_DATA_DIR "/tokenizers/paligemma/tokenizer.json");
    inputs.tokens = tokenizer.encode(processing::makePrompt(vlaTask, inputs.state), true);
    inputs.noise = processing::sampleNoise(workload.actionCount(), vlaNoiseSeed);
    return inputs;
}


// Prompts of any length for the latency rows: a prompt of N tokens is the first N tokens of
// llmParagraph in bench-inputs.h, BOS first, as the LLM benches take theirs. Only the length
// matters to the latency; the reference check keeps the observation's own prompt.
inline std::vector<int> paragraphTokens()
{
    vai::utils::Tokenizer tokenizer(VAI_DATA_DIR "/tokenizers/paligemma/tokenizer.json");
    return tokenizer.encode(llmParagraph, true);
}

}   // namespace bench::pi05

#endif   // BENCHMARKS_PI05_COMMON_WORKLOAD_H
