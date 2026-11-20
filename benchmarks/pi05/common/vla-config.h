#ifndef BENCHMARKS_PI05_COMMON_VLA_CONFIG_H
#define BENCHMARKS_PI05_COMMON_VLA_CONFIG_H

// The pi0.5 configurations the benches run, which comparison runs which, and their inputs
// prepared for a framework's image format. The observation itself (cameras, task, state, noise
// seed) is common/bench-inputs.h.

#include "processing.h"
#include "vai/utils/image-processor.h"
#include "vai/utils/tokenizer.h"
#include "bench-inputs.h"
#include "bench-utils.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace bench::pi05 {

// The same for every pi0.5 configuration: the model's, not the configuration's.
struct Pi05Model
{
    static constexpr uint32_t imageChannels = 3;
    static constexpr uint32_t imageSize = 224;
    static constexpr uint32_t actionDim = 32;
    static constexpr uint32_t maxPromptTokens = 200;
    static constexpr uint32_t tokensPerImage = 256;   // SigLIP's (224 / 14)² patches
};


// Whose policy a configuration follows, which decides the prompt.
enum class Policy {
    LeRobot,   // the Hugging Face checkpoint's own: the discretized state rides in the prompt
    OpenPI,    // openpi's, as TensorRT Edge-LLM serves it: pi05_libero leaves the state out
};


// What one prediction runs. vAi and the framework it is compared with run the same one, and
// scripts/reference.py, run with the same values, writes the actions both must match.
struct VlaConfig
{
    const char* size;   // checkpoint folder under data/ckpts/pi05/lerobot/ and data/benchmarks/pi05/
    Policy policy;
    uint32_t cameras;       // images given, the first of vlaCameras
    uint32_t actionChunk;   // actions one prediction returns
    uint32_t denoiseSteps;

    const char* policyName() const { return policy == Policy::OpenPI ? "openpi" : "lerobot"; }
    size_t actionCount() const { return size_t(actionChunk) * Pi05Model::actionDim; }
    uint32_t imageTokens() const { return cameras * Pi05Model::tokensPerImage; }

    // data/benchmarks/pi05/<size>/: the converted models and the reference outputs.
    std::filesystem::path dataDir() const { return std::filesystem::path(VAI_DATA_DIR) / "benchmarks/pi05" / size; }

    // What reference.py writes for the configuration: the actions to match, the noise they start
    // from, and the pixels the model saw ([cameras, C, S, S] in [-1, 1], after LeRobot's preprocessing).
    std::filesystem::path referencePath() const { return dataDir() / ("reference_" + suffix()); }
    std::filesystem::path noisePath() const { return dataDir() / ("noise_" + suffix()); }
    std::filesystem::path imagesPath() const { return dataDir() / ("images_" + suffix()); }

    // <cameras>_<chunk>_<steps>.bin, as reference.py names its files.
    std::string suffix() const
    {
        return std::to_string(cameras) + "_" + std::to_string(actionChunk) + "_" + std::to_string(denoiseSteps) +
               ".bin";
    }
};


namespace vlaConfigs {
// clang-format off
// Both LIBERO policies take two images. LeRobot's adds an empty third camera, but its mask keeps
// that slot out of every attention and every position: the same as the two images alone.
//
// The LeRobot LIBERO checkpoint's own policy: two images, a 50-action chunk, 10 steps.
inline constexpr VlaConfig lerobotLibero { .size = "3B", .policy = Policy::LeRobot, .cameras = 2, .actionChunk = 50, .denoiseSteps = 10 };
// openpi's pi05_libero, as TensorRT Edge-LLM serves it: two images, a 10-action chunk, 10 steps.
inline constexpr VlaConfig openpiLibero  { .size = "3B", .policy = Policy::OpenPI,  .cameras = 2, .actionChunk = 10, .denoiseSteps = 10 };
// clang-format on
}   // namespace vlaConfigs


// The configuration each framework compared with runs, by vAi's --comparison name. The first is
// the default.
struct Comparison
{
    const char* name;
    const VlaConfig* config;
};

inline constexpr Comparison comparisons[] = {
    { "migraphx", &vlaConfigs::lerobotLibero },
    { "trt", &vlaConfigs::openpiLibero },
    { "vlacpp", &vlaConfigs::lerobotLibero },
};


inline const VlaConfig* findVlaConfig(const char* comparison)
{
    for (const Comparison& c : comparisons)
        if (!strcmp(c.name, comparison))
            return c.config;
    return nullptr;
}


// How a framework takes its images; each bench names its own.
enum class ImageLayout { CHW, HWC };

struct ImageFormat
{
    ImageLayout layout;
    float min, max;   // the range pixels are normalized to
};

inline constexpr ImageFormat signedCHW{ ImageLayout::CHW, -1.0f, 1.0f };   // SigLIP's input: vAi, MIGraphX, Edge-LLM
inline constexpr ImageFormat unitHWC{ ImageLayout::HWC, 0.0f, 1.0f };      // raw RGB, which vla.cpp normalizes itself


// Pixels in signedCHW, cameras x [C, S, S], in another format.
inline std::vector<float> convertPixels(const std::vector<float>& chw, uint32_t cameras, const ImageFormat& format)
{
    const size_t C = Pi05Model::imageChannels, S = Pi05Model::imageSize;
    const bool sameRange = format.min == signedCHW.min && format.max == signedCHW.max;
    std::vector<float> pixels(chw.size());
    for (size_t b = 0; b < cameras; ++b)
        for (size_t c = 0; c < C; ++c)
            for (size_t y = 0; y < S; ++y)
                for (size_t x = 0; x < S; ++x)
                {
                    const size_t from = ((b * C + c) * S + y) * S + x;
                    const size_t to = format.layout == ImageLayout::CHW ? from : ((b * S + y) * S + x) * C + c;
                    const float value = chw[from];
                    pixels[to] = sameRange ? value : format.min + (value + 1.0f) * 0.5f * (format.max - format.min);
                }
    return pixels;
}


// One prediction's host inputs.
struct Inputs
{
    std::vector<float> pixels;     // cameras x image, in the image format
    std::vector<float> state;      // normalized, actionDim wide
    std::vector<int> tokens;       // the prompt, in the setting's policy
    std::vector<float> noise;      // [actionChunk, actionDim]
    bool imagesFromFile = false;   // reference.py's pixels; else preprocessed here
    bool noiseFromFile = false;    // reference.py's noise; else sampleNoise's
};


inline Inputs prepareInputs(const VlaConfig& config, const ImageFormat& format)
{
    namespace processing = model::production::pi05;
    using vai::utils::Align;
    using vai::utils::ImageProcessor;

    Inputs inputs;
    const uint32_t S = Pi05Model::imageSize;
    const size_t pixelCount = size_t(config.cameras) * Pi05Model::imageChannels * S * S;

    // The pixels reference.py's model saw: LeRobot resizes with its own interpolation, which
    // vAi's letterbox does not reproduce. Preprocessed here only when it has not written them.
    const std::vector<float> referencePixels = readReference(config.imagesPath().string(), pixelCount);
    inputs.imagesFromFile = !referencePixels.empty();

    if (inputs.imagesFromFile)
    {
        inputs.pixels = convertPixels(referencePixels, config.cameras, format);
    }
    else
    {
        inputs.pixels.reserve(pixelCount);

        for (uint32_t i = 0; i < config.cameras; ++i)
        {
            // clang-format off
            ImageProcessor image = ImageProcessor(std::string(VAI_DATA_DIR "/") + vlaCameras[i])
                .letterbox(S, Align::Center, 0.0f)
                .normalize(format.min, format.max);
            // clang-format on

            if (format.layout == ImageLayout::CHW)
                image.chw();

            const std::vector<float>& pixels = image;
            inputs.pixels.insert(inputs.pixels.end(), pixels.begin(), pixels.end());
        }
    }

    inputs.state = makeVLAState(Pi05Model::actionDim);
    vai::utils::Tokenizer tokenizer(VAI_DATA_DIR "/tokenizers/paligemma/tokenizer.json");
    if (config.policy == Policy::LeRobot)
        inputs.tokens = tokenizer.encode(processing::makePrompt(vlaTask, inputs.state), true);
    else
    {
        // openpi's PaligemmaTokenizer without a state: the cleaned task, then "\n" on its own.
        std::string task = vlaTask;
        task.erase(0, task.find_first_not_of(" \t\r\n"));
        task.erase(task.find_last_not_of(" \t\r\n") + 1);
        std::replace(task.begin(), task.end(), '_', ' ');
        std::replace(task.begin(), task.end(), '\n', ' ');
        inputs.tokens = tokenizer.encode(task, true);
        const std::vector<int> newline = tokenizer.encode("\n", false);
        inputs.tokens.insert(inputs.tokens.end(), newline.begin(), newline.end());
    }

    // reference.py draws its noise from torch's RNG, which sampleNoise does not reproduce:
    // only its own noise can match its reference.
    inputs.noise = readReference(config.noisePath().string(), config.actionCount());
    inputs.noiseFromFile = !inputs.noise.empty();

    if (!inputs.noiseFromFile)
        inputs.noise = processing::sampleNoise(config.actionCount(), vlaNoiseSeed);

    return inputs;
}


inline void printVlaConfig(const VlaConfig& config, const Inputs& inputs)
{
    printf(
        "config: pi0.5 %s, %s policy\n"
        "input: %u cameras x %ux%u, %zu prompt tokens, action chunk %ux%u, %u denoise steps\n",
        config.size,
        config.policyName(),
        config.cameras,
        Pi05Model::imageSize,
        Pi05Model::imageSize,
        inputs.tokens.size(),
        config.actionChunk,
        Pi05Model::actionDim,
        config.denoiseSteps);
    if (inputs.imagesFromFile)
        printf("images: %s\n", config.imagesPath().string().c_str());
    else
        printf("images: preprocessed here (no %s)\n", config.imagesPath().string().c_str());
    if (inputs.noiseFromFile)
        printf("noise: %s\n", config.noisePath().string().c_str());
    else
        printf(
            "noise: sampled, seed %llu (no %s)\n",
            (unsigned long long)vlaNoiseSeed,
            config.noisePath().string().c_str());
    fflush(stdout);
}

}   // namespace bench::pi05

#endif   // BENCHMARKS_PI05_COMMON_VLA_CONFIG_H
