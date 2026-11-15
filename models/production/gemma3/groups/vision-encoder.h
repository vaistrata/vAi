#ifndef PRODUCTION_GEMMA3_GROUPS_VISION_ENCODER_H
#define PRODUCTION_GEMMA3_GROUPS_VISION_ENCODER_H

#include "vai/groups/patch-embedding.h"
#include "vai/groups/transformer.h"
#include "models/common.h"
#include "vai/nodes/linear.h"
#include "vai/nodes/pool2d.h"
#include "vai/nodes/reshape.h"
#include "vai/nodes/rms-norm.h"

namespace model::production::gemma3::group {

using VisionTransformer = vai::group::Transformer<vai::TransformerType::Classic>;


struct VisionEncoder : vai::NodeGroup
{
    static constexpr VisionTransformerConfig config = common::so400m14(896);   // SigLIP So400m/14 at 896 pixels
    static constexpr uint32_t tokensPerSide = 16;                              // 4 x 4 patches pool into each token
    static constexpr uint32_t tokensPerImage = tokensPerSide * tokensPerSide;

    vai::group::PatchEmbedding patchEmbed;   // [B, C, S, S] -> [B, N, D], N = G * G
    VisionTransformer encoder;               //              -> [B, N, D]
    vai::node::Reshape toGrid;               //              -> [B, G, G, D]
    vai::node::AvgPool2d pool;               //              -> [B, G / 4, G / 4, D]
    vai::node::Reshape toTokens;             //              -> [B, tokensPerImage, D]
    vai::node::RmsNorm imageNorm;            //              -> [B, tokensPerImage, D]

    VisionEncoder()
    : patchEmbed({
          .imageChannels = config.imageChannels,
          .imageSize = config.imageSize,
          .patchSize = config.patchSize,
          .embedDim = config.embedDim,
          .allowBatch = true,
      })
    , encoder({ .base = config, .allowBatch = true })
    , toGrid({ -1, config.patchesPerSide(), config.patchesPerSide(), config.embedDim })
    , pool({ .kernelSize = 4, .stride = 4, .channelsLast = true })
    , toTokens({ -1, tokensPerImage, config.embedDim })
    , imageNorm({ .eps = config.normEps })
    {
        VAI_ASSERT(tokensPerSide * 4 == config.patchesPerSide());

        // clang-format off
        patchEmbed - encoder - toGrid - pool - toTokens - imageNorm;
        // clang-format on

        defineSlot("in0", patchEmbed.slot("in0"));
        defineSlot("out0", imageNorm.slot("out0"));
    }
};

}   // namespace model::production::gemma3::group

#endif   // PRODUCTION_GEMMA3_GROUPS_VISION_ENCODER_H
