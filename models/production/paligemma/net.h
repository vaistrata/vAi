/*=============================================================================
  Written by   oneway@vaistrata.com
  Model        PaliGemma (3B, 224 / 448)
=============================================================================*/

#ifndef PRODUCTION_PALIGEMMA_NET_H
#define PRODUCTION_PALIGEMMA_NET_H

#include "models/config-base.h"
#include "vai/nodes/token-embedding.h"
#include "vai/groups/patch-embedding.h"
#include "vai/nodes/concat.h"
#include "vai/groups/transformer.h"
#include "vai/nodes/linear.h"
#include "vai/nodes/topk-sampling.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>


namespace model::production::paligemma {


using VisionTransformer = vai::group::Transformer<vai::TransformerType::Classic>;


using LanguageTransformer = vai::group::Transformer<vai::TransformerType::Default>;


/////////////////////////////////////////////////////////////////////////////////////////
// ModelConfig - PaliGemma (3B, 224 / 448) - SigLIP So400m/14 and the Gemma 2B decoder
/////////////////////////////////////////////////////////////////////////////////////////
struct ModelConfig
{
    const char* variant;   // checkpoint folder under data/ckpts/paligemma/google/
    VisionModelConfig visionCfg;
    LanguageModelConfig languageCfg = {
        gemma::ModelConfig::_2B().languageCfg,
        257152,   // vocabSize: 256000 text, 1024 <loc*> and 128 <seg*>
        8192      // maxTrainedContext
    };

    static ModelConfig _3B_224() { return {.variant = "3B-224", .visionCfg = siglip::so400m14(224)}; }
    static ModelConfig _3B_448() { return {.variant = "3B-448", .visionCfg = siglip::so400m14(448)}; }
};


struct RuntimeConfig
{
    vai::Dtype kvCacheDtype = vai::Dtype::Fp16;
    uint32_t maxRuntimeContext = uint32_t(-1);
};


struct Net : public vai::NeuralNet
{
    ModelConfig modelCfg;
    RuntimeConfig runtimeCfg;
    uint32_t maxRuntimeContext;

    vai::group::PatchEmbedding patchEmbed;
    VisionTransformer encoder;
    vai::node::Linear projector;
    vai::node::Concat catTokens;

    vai::node::TokenEmbedding embedding;
    LanguageTransformer decoder;
    vai::node::Linear lmHead;   // Weights are shared with embedding.
    vai::node::TopKSampling topkSampling;

    eva::Buffer kvCacheCounter;

    Net(const vai::NeuralNetConfig& netCfg, const ModelConfig& modelCfg, const RuntimeConfig& runtimeCfg = {})
        : NeuralNet(netCfg), modelCfg(modelCfg), runtimeCfg(runtimeCfg),
          maxRuntimeContext(std::min(runtimeCfg.maxRuntimeContext, modelCfg.languageCfg.maxTrainedContext)),
          patchEmbed({.imageChannels = modelCfg.visionCfg.imageChannels,
                      .imageSize = modelCfg.visionCfg.imageSize,
                      .patchSize = modelCfg.visionCfg.patchSize,
                      .embedDim = modelCfg.visionCfg.layerCfg.embedDim}),
          encoder(modelCfg.visionCfg.with({.kvCacheDtype = runtimeCfg.kvCacheDtype})),
          projector({.inDim = modelCfg.visionCfg.layerCfg.embedDim, .outDim = modelCfg.languageCfg.layerCfg.embedDim}),
          catTokens({.allowPartialInputs = true}),
          embedding({.vocabSize = modelCfg.languageCfg.vocabSize,
                     .embedDim = modelCfg.languageCfg.layerCfg.embedDim,
                     .scale = std::sqrt(float(modelCfg.languageCfg.layerCfg.embedDim))}),
          decoder(modelCfg.languageCfg.with(
              {.kvCacheSize = maxRuntimeContext, .kvCacheDtype = runtimeCfg.kvCacheDtype, .sliceLastRow = true})),
          lmHead({.inDim = modelCfg.languageCfg.layerCfg.embedDim, .outDim = modelCfg.languageCfg.vocabSize})
    {
        this->runtimeCfg.maxRuntimeContext = maxRuntimeContext;

        input("inImage") - patchEmbed - encoder - projector - "in0" / catTokens;
        input("inTokenIds") - embedding - "in1" / catTokens;
        catTokens - decoder - lmHead - topkSampling - output("nextTokenId");

        kvCacheCounter = vai::createKvCacheCounter(device());
        decoder.setKvCacheCounter(kvCacheCounter);
    }

    uint32_t tokensSoFar_ = 0;

    // Input: image [C, S, S] (or empty), promptTokenIds [T].
    // Output: nextTokenId [1], the first generated token.
    void prefill(const vai::Tensor& image, const vai::Tensor& promptTokenIds, vai::Tensor* nextTokenId)
    {
        VAI_ASSERT(nextTokenId);
        step(image, promptTokenIds, nextTokenId);
    }

    // Input/output: inoutTokenId [1], previous token replaced by the next token.
    void decode(vai::Tensor* inoutTokenId)
    {
        VAI_ASSERT(inoutTokenId);
        step({}, *inoutTokenId, inoutTokenId);
    }

    void resetCache()
    {
        sync();   // The GPU may still be reading kvCacheCounter.
        tokensSoFar_ = 0;
        *(uint32_t*)kvCacheCounter.map() = 0;
    }

private:
    void step(const vai::Tensor& inImage, const vai::Tensor& inTokenIds, vai::Tensor* nextTokenId)
    {
        sync();   // The GPU may still be reading kvCacheCounter.

        *(uint32_t*)kvCacheCounter.map() = tokensSoFar_;
        tokensSoFar_ += inTokenIds.shape()[0] + (inImage.validShape() ? modelCfg.visionCfg.numPatches() : 0);

        topkSampling.setUniform((float)std::rand() / ((float)RAND_MAX + 1.0f));

        vai::TensorRefDict outputs = {{"nextTokenId", nextTokenId}};

        vai::TensorDict inputs = {{"inTokenIds", inTokenIds}};
        if (inImage.validShape())
            inputs.emplace("inImage", inImage);

        (*this)(std::move(inputs), std::move(outputs), false);
    }
};


}   // namespace model::production::paligemma

#endif
