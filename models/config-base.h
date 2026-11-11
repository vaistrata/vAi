/*=============================================================================
  Written by   oneway@vaistrata.com
  Configs      The base configs every model builds on, and the backbones more
               than one model reads: SigLIP and Gemma.

  A model's own checkpoints are its ModelConfig, in its net.h. Nothing the
  caller chooses is here: RuntimeConfig and the graph stay with the model.
=============================================================================*/

#ifndef MODELS_CONFIG_BASE_H
#define MODELS_CONFIG_BASE_H

#include "vai/groups/transformer.h"


// The projector from image tokens to language tokens is not part of it.
struct VisionModelConfig : public vai::TransformerConfig
{
    uint32_t imageSize = 0;      // 0: the weights pin no resolution, so a pass brings its own
    uint32_t imageChannels;
    uint32_t patchSize;

    uint32_t patchesPerSide() const
    {
        return imageSize / patchSize;
    }

    uint32_t numPatches() const
    {
        return patchesPerSide() * patchesPerSide();
    }
};


// The LM head is not part of it.
struct LanguageModelConfig : public vai::TransformerConfig
{
    uint32_t vocabSize;
    uint32_t maxTrainedContext = 0;   // 0: the weights cap no context, nothing clamps the caller
};


/////////////////////////////////////////////////////////////////////////////////////////
// SigLIP - the vision tower PaliGemma and pi0 read images with
/////////////////////////////////////////////////////////////////////////////////////////
namespace model::siglip {

// The "/14" of So400m/14: the patch projection is [1152, 3*14*14]. imageSize is
// asked for because position_embedding.weight holds one row per patch, so 224
// and 448 are different weights under one architecture.
inline VisionModelConfig so400m14(uint32_t imageSize = 0)
{
    // A base cannot be named in a designated list, so the encoder comes first, in order.
    return {
        vai::TransformerConfig{.numLayers = 27,
                               .layerCfg = {.embedDim = 1152,
                                            .headDim = 72,
                                            .numQHeads = 16,
                                            .numKVHeads = 16,
                                            .ffnHiddenDim = 4304,
                                            .ffnActivation = vai::Epilogue::GeluTanh,
                                            .normEps = 1e-6f}},
        imageSize,
        3,   // imageChannels
        14   // patchSize
    };
}

}   // namespace model::siglip


/////////////////////////////////////////////////////////////////////////////////////////
// Gemma (2B / 300M)
/////////////////////////////////////////////////////////////////////////////////////////
namespace model::gemma {

struct ModelConfig
{
    const char* variant;                    // checkpoint folder under data/gemma/
    LanguageModelConfig languageCfg;

    static ModelConfig _2B()   { return {.variant = "2B",   .languageCfg = lm(2048, 16384)}; }
    static ModelConfig _300M() { return {.variant = "300M", .languageCfg = lm(1024, 4096)}; }

private:
    static vai::RopeConfig rope()
    {
        return {
            .type = vai::RopeType::Default,
            .base = 10000.0f
        };
    }

    // numLayers, headDim and numKVHeads hold at every width, which is what lets
    // pi0 pair its two stacks layer for layer on one K/V cache.
    static LanguageModelConfig lm(uint32_t embedDim, uint32_t ffnHiddenDim)
    {
        return {
            vai::TransformerConfig{
                .numLayers = 18,
                .layerCfg = {
                    .embedDim = embedDim,
                    .headDim = 256,
                    .numQHeads = 8,
                    .numKVHeads = 1,
                    .ropeCfg = rope(),
                    .ffnHiddenDim = ffnHiddenDim,
                    .ffnActivation = vai::Epilogue::GeluTanh,
                    .normEps = 1e-6f}
            },
            256000, // vocabSize
            8192    // maxTrainedContext
        };
    }
};

}   // namespace model::gemma


#endif // MODELS_CONFIG_BASE_H
