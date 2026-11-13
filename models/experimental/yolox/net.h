/*=============================================================================
  Written by   phg9296@vaistrata.com
  Model        YOLOX (Small / Medium / Large / XLarge)
=============================================================================*/

#ifndef EXPERIMENTAL_YOLOX_NET_H
#define EXPERIMENTAL_YOLOX_NET_H

#include "groups/backbone.h"
#include "groups/neck.h"
#include "groups/head.h"
#include "groups/detection-decoder.h"


namespace model::experimental::yolox {


/////////////////////////////////////////////////////////////////////////////////////////
// YOLOX model variants
/////////////////////////////////////////////////////////////////////////////////////////
enum class Tier { SMALL, MEDIUM, LARGE, XLARGE };

inline static constexpr group::ModelBaseConfig getModelBaseConfig(Tier tier)
{
    switch (tier)
    {
    case Tier::SMALL:
        return { 32u, 1u };
    case Tier::MEDIUM:
        return { 48u, 2u };
    case Tier::LARGE:
        return { 64u, 3u };
    case Tier::XLARGE:
        return { 80u, 4u };
    default:
        return {};
    }
}

struct ModelConfig
{
    group::ModelBaseConfig baseCfg;
    Tier tier;
    uint32_t numClasses;
};

struct RuntimeConfig
{
    uint32_t imageSize = 640u;
    float confThreshold = 0.25f;
    float iouThreshold = 0.45f;
};

inline static constexpr ModelConfig getConfigCOCO(Tier tier)
{
    return {
        .baseCfg = getModelBaseConfig(tier),
        .tier = tier,
        .numClasses = 80u,
    };
}

inline const char* tierName(Tier tier)
{
    switch (tier)
    {
    case Tier::SMALL:
        return "small";
    case Tier::MEDIUM:
        return "medium";
    case Tier::LARGE:
        return "large";
    case Tier::XLARGE:
        return "xlarge";
    default:
        return "";
    }
}


/////////////////////////////////////////////////////////////////////////////////////////
// Net - Full YOLOX detection network
// Input : in0  [3, imageSize, imageSize]
// Output: out0 [N, 5 + numClasses], e.g. [8400, 85] for COCO at 640x640
/////////////////////////////////////////////////////////////////////////////////////////
struct Net : public vai::NeuralNet
{
    ModelConfig cfg;
    RuntimeConfig runtimeCfg;

    group::Backbone backbone;
    group::Neck neck;
    group::Head head;
    group::DetectionDecoder detector;

    Net(const vai::NeuralNetConfig& netCfg, const ModelConfig& cfg, const RuntimeConfig& runtimeCfg = {})
    : NeuralNet(netCfg)
    , cfg(cfg)
    , runtimeCfg(runtimeCfg)
    , backbone(cfg.baseCfg)
    , neck(cfg.baseCfg)
    , head({ .baseChannels = cfg.baseCfg.baseChannels, .numClasses = cfg.numClasses })
    , detector({ .imageSize = runtimeCfg.imageSize, .numClasses = cfg.numClasses })
    {
        VAI_ASSERT(runtimeCfg.imageSize >= 32u && (runtimeCfg.imageSize % 32u) == 0u);

        // clang-format off
        input("in0") - backbone;
        backbone/"out_p3" - "in_p3"/neck/"out_p3" - "in_p3"/head;
        backbone/"out_p4" - "in_p4"/neck/"out_p4" - "in_p4"/head;
        backbone/"out_p5" - "in_p5"/neck/"out_p5" - "in_p5"/head;
        // clang-format on

        for (const char* scale : { "p3", "p4", "p5" })
            for (const char* output : { "regobj", "cls" })
            {
                const std::string slot = std::string(output) + "_" + scale;
                // clang-format off
                head/slot - slot/detector;
                // clang-format on
            }
        // clang-format off
        detector - output("out0");
        // clang-format on
    }
};


}   // namespace model::experimental::yolox

#endif   // EXPERIMENTAL_YOLOX_NET_H
