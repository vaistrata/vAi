/*=============================================================================
  Written by   oneway@vaistrata.com
  Model        Llama 3 (3.2-1B / 3.2-3B)
=============================================================================*/

#ifndef PRODUCTION_LLAMA3_NET_H
#define PRODUCTION_LLAMA3_NET_H

#include "models/config-base.h"
#include "vai/nodes/linear.h"
#include "vai/nodes/topk-sampling.h"
#include "vai/nodes/token-embedding.h"
#include "vai/groups/transformer.h"
#include <algorithm>
#include <cstdlib>


namespace model::production::llama3 {


using Transformer = vai::group::Transformer<vai::TransformerType::Default, vai::QkvLayout::Separate>;


/////////////////////////////////////////////////////////////////////////////////////////
// ModelConfig - Llama 3 (3.2-1B / 3.2-3B)
/////////////////////////////////////////////////////////////////////////////////////////
struct ModelConfig
{
    const char* variant;   // checkpoint folder under data/
    LanguageModelConfig languageCfg;

    static ModelConfig _3_2_1B() { return {.variant = "3.2-1B", .languageCfg = lm(16, 2048, 64)}; }
    static ModelConfig _3_2_3B() { return {.variant = "3.2-3B", .languageCfg = lm(28, 3072, 128)}; }

private:
    static vai::RopeConfig rope(float factor)
    {
        return {
            .type = vai::RopeType::Llama3,
            .base = 500000.0f,
            .factor = factor,
            .lowFreqFactor = 1.0f,
            .highFreqFactor = 4.0f,
            .originalMaxPos = 8192,
        };
    }

    static LanguageModelConfig lm(uint32_t numLayers, uint32_t embedDim, uint32_t headDim)
    {
        return {
            vai::TransformerConfig{.numLayers = numLayers,
                                   .layerCfg = {.embedDim = embedDim,
                                                .headDim = headDim,
                                                .numQHeads = embedDim / headDim,
                                                .numKVHeads = 8,
                                                .ropeCfg = rope(32.0f),
                                                .ffnHiddenDim = 8192,
                                                .ffnActivation = vai::Epilogue::Silu}},
            128256,   // vocabSize
            131072    // maxTrainedContext
        };
    }
};


struct RuntimeConfig
{
    vai::Dtype kvCacheDtype = vai::Dtype::Fp16;
    uint32_t maxRuntimeContext = uint32_t(-1);
    uint32_t prefillChunk = 0;
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
        : NeuralNet(netCfg), modelCfg(modelCfg), runtimeCfg(runtimeCfg),
          maxRuntimeContext(std::min(runtimeCfg.maxRuntimeContext, modelCfg.languageCfg.maxTrainedContext)),
          embedding({.vocabSize = modelCfg.languageCfg.vocabSize, .embedDim = modelCfg.languageCfg.layerCfg.embedDim}),
          decoder(modelCfg.languageCfg.with({.kvCacheSize = maxRuntimeContext,
                                             .kvCacheDtype = runtimeCfg.kvCacheDtype,
                                             .sliceLastRow = true,
                                             .gatingLastOut = runtimeCfg.prefillChunk > 0})),
          lmHead({.inDim = modelCfg.languageCfg.layerCfg.embedDim, .outDim = modelCfg.languageCfg.vocabSize})
    {
        this->runtimeCfg.maxRuntimeContext = maxRuntimeContext;

        input("inTokenIds") - embedding - decoder - lmHead - topkSampling - output("nextTokenId");

        kvCacheCounter = vai::createKvCacheCounter(device());
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

        const uint32_t total = promptTokenIds.shape()[0];
        const uint32_t chunk = runtimeCfg.prefillChunk ? runtimeCfg.prefillChunk : total;

        // Reuse the buffer: a single chunk may have no host data after upload.
        vai::Tensor chunkInput = promptTokenIds;

        for (uint32_t off = 0; off < total; off += chunk)
        {
            const uint32_t n = (total - off < chunk) ? (total - off) : chunk;
            const bool producesToken = (off + n == total);

            if (n != total)
            {
                VAI_ASSERT(promptTokenIds.hasHostData());

                if (chunkInput.shape()[0] != n)
                    chunkInput = vai::Tensor({n});
                chunkInput.setData((const uint32_t*)promptTokenIds.hostData() + off);
            }

            step(chunkInput, producesToken ? nextTokenId : nullptr);
        }
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

        decoder.setGateOpen(nextTokenId != nullptr);

        if (nextTokenId)
            topkSampling.setUniform((float)std::rand() / ((float)RAND_MAX + 1.0f));

        vai::TensorRefDict outputs;
        if (nextTokenId)
            outputs.emplace("nextTokenId", nextTokenId);

        vai::TensorDict inputs = {{"inTokenIds", inTokenIds}};

        (*this)(std::move(inputs), std::move(outputs), false);
    }
};


}   // namespace model::production::llama3

#endif
