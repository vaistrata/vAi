/*=============================================================================
  Written by   oneway@vaistrata.com
  Model        Llama 3 (3.2-1B / 3.2-3B)
=============================================================================*/

#ifndef PRODUCTION_LLAMA3_NET_H
#define PRODUCTION_LLAMA3_NET_H

#include "models/common.h"
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
    vai::TransformerConfig trf;
    uint32_t vocabSize = 128256;
    uint32_t maxTrainedContext = 131072;

    static constexpr ModelConfig _3_2_1B() { return { .variant = "3.2-1B", .trf = lm(16, 2048, 64) }; }
    static constexpr ModelConfig _3_2_3B() { return { .variant = "3.2-3B", .trf = lm(28, 3072, 128) }; }

private:
    static constexpr vai::RopeConfig rope(float factor)
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

    static constexpr vai::TransformerConfig lm(uint32_t nBlocks, uint32_t embedDim, uint32_t headDim)
    {
        return {
            .nBlocks = nBlocks,
            .embedDim = embedDim,
            .headDim = headDim,
            .numQHeads = embedDim / headDim,
            .numKVHeads = 8,
            .ffnHiddenDim = 8192,
            .ffnActivation = vai::Epilogue::Silu,
            .ropeCfg = rope(32.0f),
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
    : NeuralNet(netCfg)
    , modelCfg(modelCfg)
    , runtimeCfg(runtimeCfg)
    , maxRuntimeContext(std::min(runtimeCfg.maxRuntimeContext, modelCfg.maxTrainedContext))
    , embedding({ .vocabSize = modelCfg.vocabSize, .embedDim = modelCfg.trf.embedDim })
    , decoder({
          .base = modelCfg.trf,
          .kvCacheSize = maxRuntimeContext,
          .kvCacheDtype = runtimeCfg.kvCacheDtype,
          .sliceLastRow = true,
          .gatingLastOut = runtimeCfg.prefillChunk > 0,
      })
    , lmHead({ .inDim = modelCfg.trf.embedDim, .outDim = modelCfg.vocabSize })
    {
        this->runtimeCfg.maxRuntimeContext = maxRuntimeContext;

        // clang-format off
        input("inTokenIds") - embedding - decoder - lmHead - topkSampling - output("nextTokenId");
        // clang-format on

        kvCacheCounter = vai::createKvCacheCounter(device());
        decoder.setKvCacheCounter(kvCacheCounter);

        causalMask = vai::createCausalMask(device(), 1);
        decoder.setCausalMask(causalMask);

        // K / V reach the attention in f16 (the cache's dtype); its precision was verified on NVIDIA / AMD.
        // Intel keeps them in Q's dtype: f16 K / V have not been tested there.
        if (vai::node::Attention::allowsDifferentKvDtype(device()))
        {
            for (auto& block : decoder.blocks)
            {
                auto& mha = block.mhAttn;
                if (mha.ropeK)
                    mha.ropeK->setODtype(runtimeCfg.kvCacheDtype, dtypeKey());
                else
                    mha.perK.setODtype(runtimeCfg.kvCacheDtype, dtypeKey());
                mha.perV.setODtype(runtimeCfg.kvCacheDtype, dtypeKey());
            }
        }
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
                    chunkInput = vai::Tensor({ n });
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

        vai::TensorDict inputs = { { "inTokenIds", inTokenIds } };

        (*this)(std::move(inputs), std::move(outputs), false);
    }
};


}   // namespace model::production::llama3

#endif
