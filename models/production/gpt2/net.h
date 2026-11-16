/*=============================================================================
  Written by   oneway@vaistrata.com
  Model        GPT-2 (124M / 355M / 774M / 1.5B)
=============================================================================*/

#ifndef PRODUCTION_GPT2_NET_H
#define PRODUCTION_GPT2_NET_H

#include "models/common.h"
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
    vai::TransformerConfig trf;
    uint32_t vocabSize = 50257;
    uint32_t maxTrainedContext = 1024;

    static constexpr ModelConfig _124M() { return { .variant = "124M", .trf = lm(12, 768) }; }
    static constexpr ModelConfig _355M() { return { .variant = "355M", .trf = lm(24, 1024) }; }
    static constexpr ModelConfig _774M() { return { .variant = "774M", .trf = lm(36, 1280) }; }
    static constexpr ModelConfig _1_5B() { return { .variant = "1.5B", .trf = lm(48, 1600) }; }

private:
    // maxTrainedContext is the learned positional table's row count, so it is a
    // hard limit. Positions being that table means no block carries RoPE.
    static constexpr vai::TransformerConfig lm(uint32_t nBlocks, uint32_t embedDim)
    {
        return {
            .nBlocks = nBlocks,
            .embedDim = embedDim,
            .headDim = 64,
            .numQHeads = embedDim / 64,
            .numKVHeads = embedDim / 64,
            .ffnHiddenDim = embedDim * 4,
            .ffnActivation = vai::Epilogue::GeluTanh,
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
    , maxRuntimeContext(std::min(runtimeCfg.maxRuntimeContext, modelCfg.maxTrainedContext))
    , embedding({
          .vocabSize = modelCfg.vocabSize,
          .embedDim = modelCfg.trf.embedDim,
          .posTableSize = modelCfg.maxTrainedContext,
      })
    , decoder({
          .base = modelCfg.trf,
          .kvCacheSize = maxRuntimeContext,
          .kvCacheDtype = runtimeCfg.kvCacheDtype,
          .sliceLastRow = true,
      })
    , lmHead({ .inDim = modelCfg.trf.embedDim, .outDim = modelCfg.vocabSize })
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
