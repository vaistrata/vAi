/*=============================================================================
  Written by   phg9296@vaistrata.com
  Model        YOLOX (Small / Medium / Large / XLarge)
=============================================================================*/

#ifndef EXPERIMENTAL_YOLOX_NET_H
#define EXPERIMENTAL_YOLOX_NET_H

#include "groups/backbone.h"
#include "groups/neck.h"
#include "groups/head.h"
#include "nodes/decode.h"
#include <cmath>
#include <cstdint>


namespace model::experimental::yolox {


/////////////////////////////////////////////////////////////////////////////////////////
// YOLOX model variants
/////////////////////////////////////////////////////////////////////////////////////////
enum class Tier { SMALL, MEDIUM, LARGE, XLARGE };

struct ModelScale
{
    float depth;
    float width;
};

inline static constexpr ModelScale getModelScale(Tier tier)
{
    switch (tier)
    {
    case Tier::SMALL:
        return { .depth = 0.33f, .width = 0.50f };
    case Tier::MEDIUM:
        return { .depth = 0.67f, .width = 0.75f };
    case Tier::LARGE:
        return { .depth = 1.00f, .width = 1.00f };
    case Tier::XLARGE:
        return { .depth = 1.33f, .width = 1.25f };
    default:
        return {};
    }
}

struct ModelConfig
{
    ModelScale scale;
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
        .scale = getModelScale(tier),
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
// Input : in0  [B, 3, imageSize, imageSize]
// Output: out0 [B, N, 5 + numClasses], e.g. [1, 8400, 85] for COCO at 640x640
// B is the batch size; a single image uses B=1.
/////////////////////////////////////////////////////////////////////////////////////////
struct Net : public vai::NeuralNet
{
    ModelConfig cfg;
    RuntimeConfig runtimeCfg;

    group::Backbone backbone;
    group::Neck neck;
    group::Head head;
    node::Decode detector;

    Net(const vai::NeuralNetConfig& netCfg, const ModelConfig& cfg, const RuntimeConfig& runtimeCfg = {})
    : NeuralNet(netCfg)
    , cfg(cfg)
    , runtimeCfg(runtimeCfg)
    , backbone(64.0f * cfg.scale.width, std::round(3.0f * cfg.scale.depth))
    , neck(64.0f * cfg.scale.width, std::round(3.0f * cfg.scale.depth))
    , head({ .baseChannels = static_cast<uint32_t>(64.0f * cfg.scale.width), .numClasses = cfg.numClasses })
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

        // The detections go to the host's postprocess as fp32, whatever oDefaultDtype is.
        detector.setODtype(vai::Dtype::Fp32, dtypeKey());
    }
};


}   // namespace model::experimental::yolox

#endif   // EXPERIMENTAL_YOLOX_NET_H
