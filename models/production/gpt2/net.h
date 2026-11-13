/*=============================================================================
  Written by   oneway@vaistrata.com
  Model        GPT-2 (124M / 355M / 774M / 1.5B)
=============================================================================*/

#ifndef PRODUCTION_GPT2_NET_H
#define PRODUCTION_GPT2_NET_H

#include "models/config-base.h"
#include "vai/nodes/linear.h"
#include "vai/nodes/topk-sampling.h"
#include "vai/nodes/token-embedding.h"
#include "vai/groups/transformer.h"
#include <algorithm>
#include <cstdlib>


namespace model::production::gpt2 {


using Transformer = vai::group::Transformer<vai::TransformerType::Classic, vai::QkvLayout::Fused>;


/////////////////////////////////////////////////////////////////////////////////////////
// ModelConfig - GPT-2 (124M / 355M / 774M / 1.5B)
/////////////////////////////////////////////////////////////////////////////////////////
struct ModelConfig
{
    const char* variant;   // checkpoint folder under data/
    LanguageModelConfig languageCfg;

    static ModelConfig _124M() { return { .variant = "124M", .languageCfg = lm(12, 768) }; }
    static ModelConfig _355M() { return { .variant = "355M", .languageCfg = lm(24, 1024) }; }
    static ModelConfig _774M() { return { .variant = "774M", .languageCfg = lm(36, 1280) }; }
    static ModelConfig _1_5B() { return { .variant = "1.5B", .languageCfg = lm(48, 1600) }; }

private:
    // maxTrainedContext is the learned positional table's row count, so it is a
    // hard limit. Positions being that table means no layer carries RoPE.
    static LanguageModelConfig lm(uint32_t numLayers, uint32_t embedDim)
    {
        return {
            vai::TransformerConfig{
                .numLayers = numLayers,
                .layerCfg = {
                    .embedDim = embedDim,
                    .headDim = 64,
                    .numQHeads = embedDim / 64,
                    .numKVHeads = embedDim / 64,
                    .ffnHiddenDim = embedDim * 4,
                    .ffnActivation = vai::Epilogue::GeluTanh,
                },
            },
            50257,   // vocabSize
            1024     // maxTrainedContext
        };
    }
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

    vai::node::TokenEmbedding embedding;
    Transformer decoder;
    vai::node::Linear lmHead;
    vai::node::TopKSampling topkSampling;

    eva::Buffer kvCacheCounter;
    eva::Buffer causalMask;

    Net(const vai::NeuralNetConfig& netCfg, const ModelConfig& modelCfg, const RuntimeConfig& runtimeCfg = {})
    : NeuralNet(netCfg)
    , modelCfg(modelCfg)
    , runtimeCfg(runtimeCfg)
    , maxRuntimeContext(std::min(runtimeCfg.maxRuntimeContext, modelCfg.languageCfg.maxTrainedContext))
    , embedding({
          .vocabSize = modelCfg.languageCfg.vocabSize,
          .embedDim = modelCfg.languageCfg.layerCfg.embedDim,
          .posTableSize = modelCfg.languageCfg.maxTrainedContext,
      })
    , decoder(modelCfg.languageCfg.with(
          { .kvCacheSize = maxRuntimeContext, .kvCacheDtype = runtimeCfg.kvCacheDtype, .sliceLastRow = true }))
    , lmHead({ .inDim = modelCfg.languageCfg.layerCfg.embedDim, .outDim = modelCfg.languageCfg.vocabSize })
    {
        this->runtimeCfg.maxRuntimeContext = maxRuntimeContext;

        // clang-format off
        input("inTokenIds") - embedding - decoder - lmHead - topkSampling - output("nextTokenId");
        // clang-format on

        kvCacheCounter = vai::createKvCacheCounter(device());
        embedding.setKvCacheCounter(kvCacheCounter);
        decoder.setKvCacheCounter(kvCacheCounter);

        causalMask = vai::createCausalMask(device(), 1);
        decoder.setCausalMask(causalMask);
    }

    uint32_t tokensSoFar_ = 0;

    // Input: promptTokenIds [T].
    // Output: nextTokenId [1], the first generated token.
    void prefill(const vai::Tensor& promptTokenIds, vai::Tensor* nextTokenId)
    {
        VAI_ASSERT(nextTokenId);
        step(promptTokenIds, nextTokenId);
    }

    // Input/output: inoutTokenId [1], previous token replaced by the next token.
    void decode(vai::Tensor* inoutTokenId)
    {
        VAI_ASSERT(inoutTokenId);
        step(*inoutTokenId, inoutTokenId);
    }

    void resetCache()
    {
        sync();   // The GPU may still be reading kvCacheCounter.
        tokensSoFar_ = 0;
        *(uint32_t*)kvCacheCounter.map() = 0;
    }

private:
    void step(const vai::Tensor& inTokenIds, vai::Tensor* nextTokenId)
    {
        sync();   // The GPU may still be reading kvCacheCounter.

        *(uint32_t*)kvCacheCounter.map() = tokensSoFar_;
        tokensSoFar_ += inTokenIds.shape()[0];

        topkSampling.setUniform((float)std::rand() / ((float)RAND_MAX + 1.0f));
        (*this)({ { "inTokenIds", inTokenIds } }, { { "nextTokenId", nextTokenId } }, false);
    }
};


}   // namespace model::production::gpt2

#endif
