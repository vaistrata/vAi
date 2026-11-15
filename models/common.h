#ifndef MODELS_COMMON_H
#define MODELS_COMMON_H

#include "vai/groups/transformer.h"
#include <stdexcept>
#include <string_view>


struct VisionTransformerConfig : public vai::TransformerConfig
{
    uint32_t imageSize = 0;   // 0: the weights pin no resolution, so a pass brings its own
    uint32_t imageChannels;
    uint32_t patchSize;

    constexpr uint32_t patchesPerSide() const { return imageSize / patchSize; }

    constexpr uint32_t numPatches() const { return patchesPerSide() * patchesPerSide(); }
};


namespace model::common {

// SigLIP So400m/14. imageSize must be the checkpoint's: its position table holds one row per patch.
constexpr VisionTransformerConfig so400m14(uint32_t imageSize = 0)
{
    return {
        vai::TransformerConfig{
            .nBlocks = 27,
            .embedDim = 1152,
            .headDim = 72,
            .numQHeads = 16,
            .numKVHeads = 16,
            .ffnHiddenDim = 4304,
            .ffnActivation = vai::Epilogue::GeluTanh,
            .normEps = 1e-6f,
        },
        imageSize,
        3,   // imageChannels
        14   // patchSize
    };
}


// Gemma. variant: "2B" or "300M". nBlocks, headDim and numKVHeads hold at every width,
// which is what lets pi0 pair its two stacks block for block on one K/V cache.
constexpr vai::TransformerConfig gemma(std::string_view variant)
{
    const bool is2B = variant == "2B";
    if (!is2B && variant != "300M")
        throw std::invalid_argument("unknown Gemma variant");
    return {
        .nBlocks = 18,
        .embedDim = is2B ? 2048u : 1024u,
        .headDim = 256,
        .numQHeads = 8,
        .numKVHeads = 1,
        .ffnHiddenDim = is2B ? 16384u : 4096u,
        .ffnActivation = vai::Epilogue::GeluTanh,
        .normEps = 1e-6f,
        .ropeCfg = vai::RopeConfig{ .base = 10000.0f },
    };
}

}   // namespace model::common


#endif   // MODELS_COMMON_H
