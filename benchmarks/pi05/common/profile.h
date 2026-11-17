#ifndef BENCHMARKS_PI05_COMMON_PROFILE_H
#define BENCHMARKS_PI05_COMMON_PROFILE_H

// What one comparison holds both sides to. Every framework runs the same pi0.5 core
// (vision encoder, prefill, Euler denoise), but each takes its inputs and returns its
// actions by its own convention. A profile names that convention, and vAi follows the
// profile of the framework it is compared with:
//
//     vAi vs MIGraphX   native    the ONNX export of vAi's own pipeline
//     vAi vs TensorRT   openpi    Edge-LLM's OpenPI pi05 runtime
//     vAi vs vla.cpp    lerobot   vla.cpp's LeRobot pipeline

#include <cstring>
#include <filesystem>
#include <string>

namespace bench::pi05 {

enum class Profile { Native, OpenPI, LeRobot };

enum class ImageLayout { CHW, HWC };

// The type the framework takes its images in; the bench converts the FP32 pixels to it.
enum class ImageDtype { Fp32, Fp16 };

// What a timed prediction covers.
enum class TimedStages {
    Core,   // model inputs (pixels, tokens, noise) -> normalized actions
    Full,   // also the framework's own image normalization and action unnormalization
};


struct ProfileSpec
{
    Profile profile;
    const char* name;

    ImageLayout imageLayout;
    float imageMin, imageMax;   // the range pixels are normalized to
    ImageDtype imageDtype;

    TimedStages timed;

    // State and action statistics, relative to the bundle; nullptr: neutral (q01 = -1,
    // q99 = 1), so the actions are the model's own.
    const char* statsFile;
};


// clang-format off
inline constexpr ProfileSpec profiles[] = {
    { Profile::Native,  "native",  ImageLayout::CHW, -1.0f, 1.0f, ImageDtype::Fp32, TimedStages::Core, nullptr },
    { Profile::OpenPI,  "openpi",  ImageLayout::CHW, -1.0f, 1.0f, ImageDtype::Fp16, TimedStages::Core, nullptr },
    { Profile::LeRobot, "lerobot", ImageLayout::HWC,  0.0f, 1.0f, ImageDtype::Fp32, TimedStages::Full, nullptr },
};
// clang-format on


inline const ProfileSpec& profileSpec(Profile profile)
{
    for (const ProfileSpec& spec : profiles)
        if (spec.profile == profile)
            return spec;
    return profiles[0];
}


inline const ProfileSpec* findProfile(const char* name)
{
    for (const ProfileSpec& spec : profiles)
        if (!strcmp(spec.name, name))
            return &spec;
    return nullptr;
}


// data/benchmarks/pi05/<size>/<profile>/reference.bin, or, until that profile has its own,
// the export's shared data/benchmarks/pi05/<size>/reference.bin.
inline std::string referencePath(const ProfileSpec& spec, const std::filesystem::path& sizeDir)
{
    const std::filesystem::path own = sizeDir / spec.name / "reference.bin";
    if (std::filesystem::exists(own))
        return own.string();
    return (sizeDir / "reference.bin").string();
}

}   // namespace bench::pi05

#endif   // BENCHMARKS_PI05_COMMON_PROFILE_H
