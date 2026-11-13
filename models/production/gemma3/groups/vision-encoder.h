#ifndef PRODUCTION_GEMMA3_GROUPS_VISION_ENCODER_H
#define PRODUCTION_GEMMA3_GROUPS_VISION_ENCODER_H

#include "vai/groups/patch-embedding.h"
#include "vai/groups/transformer.h"
#include "models/config-base.h"
#include "vai/nodes/linear.h"
#include "vai/nodes/pool2d.h"
#include "vai/nodes/reshape.h"
#include "vai/nodes/rms-norm.h"

namespace model::production::gemma3 {

struct VisionEncoderConfig : public VisionModelConfig
{
    static constexpr uint32_t tokensPerSide = 16;
    static constexpr uint32_t tokensPerImage = 256;
    uint32_t imageTokenId = 262144;

    constexpr VisionEncoderConfig()
    : VisionModelConfig(siglip::so400m14(896))
    {
    }
};

}   // namespace model::production::gemma3
namespace model::production::gemma3::group {

using VisionTransformer = vai::group::Transformer<vai::TransformerType::Classic>;


struct VisionEncoder : vai::NodeGroup
{
    vai::group::PatchEmbedding patchEmbed;   // [B, C, S, S] -> [B, N, D], N = G * G
    VisionTransformer encoder;               //              -> [B, N, D]
    vai::node::Reshape toGrid;               //              -> [B, G, G, D]
    vai::node::AvgPool2d pool;               //              -> [B, G / 4, G / 4, D]
    vai::node::Reshape toTokens;             //              -> [B, tokensPerImage, D]
    vai::node::RmsNorm imageNorm;            //              -> [B, tokensPerImage, D]

    VisionEncoder(const VisionEncoderConfig& cfg)
    : patchEmbed({
          .imageChannels = cfg.imageChannels,
          .imageSize = cfg.imageSize,
          .patchSize = cfg.patchSize,
          .embedDim = cfg.layerCfg.embedDim,
          .allowBatch = true,
      })
    , encoder(cfg.with({ .allowBatch = true }))
    , toGrid({ -1, cfg.patchesPerSide(), cfg.patchesPerSide(), cfg.layerCfg.embedDim })
    , pool({ .kernelSize = 4, .stride = 4, .channelsLast = true })
    , toTokens({ -1, cfg.numPatches() / 16, cfg.layerCfg.embedDim })
    , imageNorm({ .eps = cfg.layerCfg.normEps })
    {
        VAI_ASSERT(cfg.tokensPerSide * 4 == cfg.patchesPerSide());
        VAI_ASSERT(cfg.tokensPerImage * 16 == cfg.numPatches());

        // clang-format off
        patchEmbed - encoder - toGrid - pool - toTokens - imageNorm;
        // clang-format on

        defineSlot("in0", patchEmbed.slot("in0"));
        defineSlot("out0", imageNorm.slot("out0"));
    }
};

}   // namespace model::production::gemma3::group

#endif   // PRODUCTION_GEMMA3_GROUPS_VISION_ENCODER_H
