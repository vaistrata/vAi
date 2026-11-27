#ifndef EXPERIMENTAL_QWEN3VL_GROUPS_VISION_ENCODER_H
#define EXPERIMENTAL_QWEN3VL_GROUPS_VISION_ENCODER_H

#include "../nodes/interp-pos-embed.h"
#include "models/common.h"
#include "vai/groups/feed-forward.h"
#include "vai/groups/patch-embedding.h"
#include "vai/nodes/layer-norm.h"
#include "vai/nodes/reshape.h"
#include <array>
#include <optional>
#include <string>
#include <vector>

namespace model::experimental::qwen3vl {

struct VisionConfig
{
    VisionTransformerConfig trf;
    std::array<uint32_t, 3> deepstack;   // the blocks whose outputs feed decoder blocks 0, 1 and 2
    static constexpr uint32_t mergeSize = 2;
    static constexpr uint32_t patchesPerToken = mergeSize * mergeSize;
};

// Qwen3-VL's vision model: a ViT with 16 heads, 16 x 16 patches and the axial 2D
// RoPE at every size, and its DeepStack taps. The image size is the run's, so the
// weights pin none.
constexpr VisionConfig vm(uint32_t nBlocks, uint32_t embedDim, uint32_t ffnHiddenDim, std::array<uint32_t, 3> deepstack)
{
    return {
        .trf = {
            vai::TransformerConfig{
                .nBlocks = nBlocks,
                .embedDim = embedDim,
                .headDim = embedDim / 16,
                .numQHeads = 16,
                .numKVHeads = 16,
                .ffnHiddenDim = ffnHiddenDim,
                .ffnActType = vai::ActType::GeluTanh,
                .normEps = 1e-6f,
                .ropeCfg = vai::RopeConfig{
                    .posType = vai::RopePosType::Grid2D,
                    .base = 10000.0f,
                    .tileSize = VisionConfig::mergeSize,
                },
            },
            0,    // imageSize
            3,    // imageChannels
            16,   // patchSize
        },
        .deepstack = deepstack,
    };
}

}   // namespace model::experimental::qwen3vl
namespace model::experimental::qwen3vl::group {

// The checkpoint stacks Q, K and V into one weight, hence the Fused layout.
using VisionTransformer = vai::group::Transformer<vai::TransformerType::Classic, vai::QkvLayout::Fused>;


/////////////////////////////////////////////////////////////////////////////////////////
// Connector - the ViT's patches to fewer tokens of the decoder's width
// Input: in0 [B, 4 * N, D], patches in 2 x 2 tiles
// Output: out0 [B * N, outDim], one row per token, images one after another
//
// PatchEmbedding lays each 2 x 2 tile of patches in 4 consecutive rows, top-left,
// top-right, bottom-left, bottom-right, so they join as one 4 * D row with no grid
// size; tokenNorm and projector then take it to the decoder's width (HF's
// Qwen3VLVisionPatchMerger). Only a DeepStack connector has tokenNorm: the main
// one reads patches the ViT's finalNorm already normalized.
/////////////////////////////////////////////////////////////////////////////////////////
struct Connector : public vai::NodeGroup
{
    vai::node::Reshape toTokens;                     // [B, 4 * N, D] -> [B * N, 4 * D]
    std::optional<vai::node::LayerNorm> tokenNorm;   // [B * N, 4 * D]
    vai::group::FeedForward projector;               // -> [B * N, outDim], GELU between

    struct Config
    {
        VisionConfig base;
        uint32_t outDim;   // the decoder's width
        bool normTokens;   // a DeepStack connector's: normalize the regrouped row
    };

    Connector(Config config)
    : toTokens({ -1, config.base.patchesPerToken * config.base.trf.embedDim })
    , projector({
          .inDim = config.base.patchesPerToken * config.base.trf.embedDim,
          .hiddenDim = config.base.patchesPerToken * config.base.trf.embedDim,
          .outDim = config.outDim,
          .actType = vai::ActType::Gelu,
      })
    {
        if (config.normTokens)
            tokenNorm.emplace(vai::node::LayerNorm::Config{ .eps = config.base.trf.normEps });

        // clang-format off
        toTokens - tokenNorm - projector;
        // clang-format on

        defineSlot("in0", toTokens.slot("in0"));
        defineSlot("out0", projector.slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// VisionEncoder - Qwen3-VL's ViT, its connector and the DeepStack taps
// Input: in0 [B, C, S, S]
// Output: out0 [B * tokensPerImage, outDim], the image tokens
//         deep0.. [B * tokensPerImage, outDim], one per DeepStack tap
//
// Patches run in 2 x 2 tiles: the 2D RoPE turns each by its (row, column), which
// Rope reads off the patch index with the same tileSize, and posEmbed adds the
// learned position table resampled to the grid in that order.
/////////////////////////////////////////////////////////////////////////////////////////
struct VisionEncoder : public vai::NodeGroup
{
    struct Config
    {
        VisionConfig base;
        uint32_t imageSize;
        uint32_t outDim;
    };

    const uint32_t patchesPerSide;
    const uint32_t tokensPerSide;
    const uint32_t tokensPerImage;

    vai::group::PatchEmbedding patchEmbed;   // [B, C, S, S] -> [B, G * G, D]
    node::InterpPosEmbed posEmbed;           //              -> [B, G * G, D], in place
    VisionTransformer encoder;               //              -> [B, G * G, D]
    Connector connector;                     //              -> [B * tokensPerImage, outDim]
    std::vector<Connector> deepConnectors;   // one per DeepStack tap

    VisionEncoder(Config config)
    : patchesPerSide(config.imageSize / config.base.trf.patchSize)
    , tokensPerSide(patchesPerSide / config.base.mergeSize)
    , tokensPerImage(tokensPerSide * tokensPerSide)
    , patchEmbed({
          .imageChannels = config.base.trf.imageChannels,
          .patchSize = config.base.trf.patchSize,
          .embedDim = config.base.trf.embedDim,
          .tileSize = config.base.mergeSize,
      })
    , posEmbed({ .tileSize = config.base.mergeSize })
    , encoder({ .base = config.base.trf })
    , connector({ .base = config.base, .outDim = config.outDim, .normTokens = false })
    {
        VAI_ASSERT(config.imageSize % (config.base.mergeSize * config.base.trf.patchSize) == 0);

        deepConnectors.reserve(config.base.deepstack.size());
        for (size_t j = 0; j < config.base.deepstack.size(); ++j)
            deepConnectors.emplace_back(Connector::Config{
                .base = config.base,
                .outDim = config.outDim,
                .normTokens = true,
            });

        // clang-format off
        patchEmbed - posEmbed - encoder - connector;
        for (size_t j = 0; j < deepConnectors.size(); ++j)
            encoder.blocks[config.base.deepstack[j]] - deepConnectors[j];
        // clang-format on

        defineSlot("in0", patchEmbed.slot("in0"));
        defineSlot("out0", connector.slot("out0"));
        for (size_t j = 0; j < deepConnectors.size(); ++j)
            defineSlot("deep" + std::to_string(j), deepConnectors[j].slot("out0"));
    }

    // gridW [1]: patches per row, from which posEmbed and the 2D RoPE read each patch's
    // row and column.
    void setGridWidth(eva::Buffer buffer)
    {
        posEmbed.setGridWidth(buffer);
        for (auto& block : encoder.blocks)
        {
            block.mhAttn.ropeQ->setPosParam(buffer);
            block.mhAttn.ropeK->setPosParam(buffer);
        }
    }
};

}   // namespace model::experimental::qwen3vl::group

#endif   // EXPERIMENTAL_QWEN3VL_GROUPS_VISION_ENCODER_H
