#ifndef PRODUCTION_GEMMA3_GROUPS_VISION_ENCODER_H
#define PRODUCTION_GEMMA3_GROUPS_VISION_ENCODER_H

#include "models/common.h"
#include "vai/groups/patch-embedding.h"
#include "vai/groups/transformer.h"
#include "vai/nodes/linear.h"
#include "vai/nodes/pool2d.h"
#include "vai/nodes/reshape.h"
#include "vai/nodes/rms-norm.h"

namespace model::production::gemma3 {

struct VisionConfig
{
    static constexpr VisionTransformerConfig trf = common::so400m14(896);
    static constexpr uint32_t mergeSize = 4;
    static_assert(trf.patchesPerSide() % mergeSize == 0);
};

}   // namespace model::production::gemma3
namespace model::production::gemma3::group {

using VisionTransformer = vai::group::Transformer<vai::TransformerType::Classic>;


/////////////////////////////////////////////////////////////////////////////////////////
// Connector - the ViT's patches to fewer tokens of the decoder's width
// Input: in0 [B, G * G, D], patches in raster order
// Output: out0 [B * (G / 4)^2, outDim], one row per token, images one after another
//
// Each 4 x 4 block of patches averages to one token, 1/16 the rows; tokenNorm and
// projector then take it to the decoder's width (HF's Gemma3MultiModalProjector).
/////////////////////////////////////////////////////////////////////////////////////////
struct Connector : public vai::NodeGroup
{
    vai::node::Reshape toGrid;      // [B, G * G, D] -> [B, G, G, D]
    vai::node::AvgPool2d pool;      // -> [B, G / 4, G / 4, D]
    vai::node::Reshape toTokens;    // -> [B * (G / 4)^2, D]
    vai::node::RmsNorm tokenNorm;   // [B * (G / 4)^2, D]
    vai::node::Linear projector;    // -> [B * (G / 4)^2, outDim]

    struct Config
    {
        VisionConfig base;
        uint32_t outDim;   // the decoder's width
    };

    Connector(Config config)
    : toGrid({ -1, config.base.trf.patchesPerSide(), config.base.trf.patchesPerSide(), config.base.trf.embedDim })
    , pool({ .kernelSize = config.base.mergeSize, .stride = config.base.mergeSize, .channelsLast = true })
    , toTokens({ -1, config.base.trf.embedDim })
    , tokenNorm({ .eps = config.base.trf.normEps })
    , projector({ .inDim = config.base.trf.embedDim, .outDim = config.outDim })
    {
        // clang-format off
        toGrid - pool - toTokens - tokenNorm - projector;
        // clang-format on

        defineSlot("in0", toGrid.slot("in0"));
        defineSlot("out0", projector.slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// VisionEncoder - Gemma 3's ViT and its connector
// Input: in0 [B, C, S, S]
// Output: out0 [B * tokensPerImage, outDim], the image tokens
/////////////////////////////////////////////////////////////////////////////////////////
struct VisionEncoder : public vai::NodeGroup
{
    struct Config
    {
        VisionConfig base;
        uint32_t outDim;   // the decoder's width
    };

    const uint32_t patchesPerSide;
    const uint32_t tokensPerSide;
    const uint32_t tokensPerImage;

    vai::group::PatchEmbedding patchEmbed;   // [B, C, S, S] -> [B, G * G, D], G = patchesPerSide
    VisionTransformer encoder;               //              -> [B, G * G, D]
    Connector connector;                     //              -> [B * tokensPerImage, outDim]

    VisionEncoder(Config config)
    : patchesPerSide(config.base.trf.patchesPerSide())
    , tokensPerSide(patchesPerSide / config.base.mergeSize)
    , tokensPerImage(tokensPerSide * tokensPerSide)
    , patchEmbed({
          .imageChannels = config.base.trf.imageChannels,
          .imageSize = config.base.trf.imageSize,
          .patchSize = config.base.trf.patchSize,
          .embedDim = config.base.trf.embedDim,
      })
    , encoder({ .base = config.base.trf })
    , connector({ .base = config.base, .outDim = config.outDim })
    {
        // clang-format off
        patchEmbed - encoder - connector;
        // clang-format on

        defineSlot("in0", patchEmbed.slot("in0"));
        defineSlot("out0", connector.slot("out0"));
    }
};

}   // namespace model::production::gemma3::group

#endif   // PRODUCTION_GEMMA3_GROUPS_VISION_ENCODER_H
