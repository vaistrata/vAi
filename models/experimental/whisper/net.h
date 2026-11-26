/*=============================================================================
  Written by   Shivanshyyy (github.com/Shivanshyyy)
  Model        Whisper (tiny / base / small / medium / large-v3 / large-v3-turbo)
  Nets         encoder::Net (log-mel -> encoder states -> every decoder layer's cross K / V)
               decoder::Net (tokens + cross K / V -> next token, greedy)
               Net          (the two nets, sharing one pipeline cache)
=============================================================================*/

#ifndef EXPERIMENTAL_WHISPER_NET_H
#define EXPERIMENTAL_WHISPER_NET_H

#include "vai/groups/feed-forward.h"
#include "vai/groups/multi-head-attention.h"
#include "vai/groups/transformer.h"
#include "vai/nodes/add.h"
#include "vai/nodes/argmax.h"
#include "vai/nodes/attention.h"
#include "vai/nodes/convolution.h"
#include "vai/nodes/identity.h"
#include "vai/nodes/layer-norm.h"
#include "vai/nodes/linear.h"
#include "vai/nodes/permute.h"
#include "vai/nodes/reshape.h"
#include "vai/nodes/slice.h"
#include "vai/nodes/token-embedding.h"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>


namespace model::experimental::whisper {


/////////////////////////////////////////////////////////////////////////////////////////
// ModelConfig - Whisper (tiny / base / small / medium / large-v3 / large-v3-turbo)
/////////////////////////////////////////////////////////////////////////////////////////
struct ModelConfig
{
    const char* variant;   // checkpoint folder name
    uint32_t numMels;
    uint32_t vocabSize;
    uint32_t audioContext = 1500;   // encoder positions: 30 s of 10 ms frames, halved by conv2
    uint32_t textContext = 448;     // decoder positions (learned table rows)
    vai::TransformerConfig encoderCfg;
    vai::TransformerConfig decoderCfg;

    static ModelConfig tiny() { return make("tiny", 384, 4, 4, 80, 51865); }
    static ModelConfig base() { return make("base", 512, 6, 6, 80, 51865); }
    static ModelConfig small() { return make("small", 768, 12, 12, 80, 51865); }
    static ModelConfig medium() { return make("medium", 1024, 24, 24, 80, 51865); }
    static ModelConfig largeV3() { return make("large-v3", 1280, 32, 32, 128, 51866); }
    static ModelConfig largeV3Turbo() { return make("large-v3-turbo", 1280, 32, 4, 128, 51866); }

    static ModelConfig byName(const std::string& name)
    {
        for (auto cfg : { tiny(), base(), small(), medium(), largeV3(), largeV3Turbo() })
            if (name == cfg.variant)
                return cfg;
        throw std::runtime_error("unknown Whisper variant: " + name);
    }

    uint32_t embedDim() const { return encoderCfg.embedDim; }
    uint32_t numHeads() const { return encoderCfg.numQHeads; }
    uint32_t headDim() const { return encoderCfg.headDim; }

private:
    // Pre-LayerNorm blocks with exact GELU; head dim is 64 at every size
    static vai::TransformerConfig stack(uint32_t nBlocks, uint32_t embedDim)
    {
        return {
            .nBlocks = nBlocks,
            .embedDim = embedDim,
            .headDim = 64,
            .numQHeads = embedDim / 64,
            .numKVHeads = embedDim / 64,
            .ffnHiddenDim = embedDim * 4,
            .ffnActType = vai::ActType::Gelu,
        };
    }

    static ModelConfig make(
        const char* variant, uint32_t embedDim, uint32_t encoderLayers, uint32_t decoderLayers, uint32_t numMels,
        uint32_t vocabSize)
    {
        return {
            .variant = variant,
            .numMels = numMels,
            .vocabSize = vocabSize,
            .encoderCfg = stack(encoderLayers, embedDim),
            .decoderCfg = stack(decoderLayers, embedDim),
        };
    }
};


struct RuntimeConfig
{
    vai::Dtype kvCacheDtype = vai::Dtype::Fp16;   // decoder self-attention cache
};


namespace encoder {

using Transformer = vai::group::Transformer<vai::TransformerType::Classic, vai::QkvLayout::Separate>;


/////////////////////////////////////////////////////////////////////////////////////////
// encoder::Net - log-mel -> encoder states -> every decoder layer's cross K / V
// Input:  inMel       [numMels, 1, 3000], one 30 s window
//         inPositions [audioContext, embedDim], the sinusoid table (no node holds a
//                     constant addend, so it rides in as an input)
// Output: crossK<i> / crossV<i> [H, audioContext, headDim], one pair per decoder layer
//
// Conv1d(k=3, pad=1) is a 3x3 same-padded Convolution over a 1-pixel-high image
// whose kernel rows above and below are zero (see openai-weights.h).
// Linear accumulates in fp32 by default, which Whisper's activation outliers need.
/////////////////////////////////////////////////////////////////////////////////////////
struct Net : public vai::NeuralNet
{
    ModelConfig modelCfg;

    vai::node::Convolution conv1;    // [M, 1, 3000] -> [D, 1, 3000]
    vai::node::Convolution conv2;    //              -> [D, 1, 1500]
    vai::node::Reshape dropHeight;   //              -> [D, 1500]
    vai::node::Permute toTokens;     //              -> [1500, D]
    vai::node::Add addPositions;
    Transformer transformer;

    std::vector<vai::node::Linear> crossK, crossV;    // [1500, D]
    std::vector<vai::node::Reshape> splitK, splitV;   // [1500, H, Dh]
    std::vector<vai::node::Permute> perK, perV;       // [H, 1500, Dh]

    vai::Tensor positions;   // filled by the weight loader
    eva::Buffer bidirectionalMask;

    Net(const vai::NeuralNetConfig& netCfg, const ModelConfig& modelCfg)
    : NeuralNet(netCfg)
    , modelCfg(modelCfg)
    , conv1({
          .inChannels = modelCfg.numMels,
          .outChannels = modelCfg.embedDim(),
          .kernelSize = 3,
          .stride = 1,
          .actType = vai::ActType::Gelu,
      })
    , conv2({
          .inChannels = modelCfg.embedDim(),
          .outChannels = modelCfg.embedDim(),
          .kernelSize = 3,
          .stride = 2,
          .actType = vai::ActType::Gelu,
      })
    , dropHeight(std::vector<int64_t>{ modelCfg.embedDim(), -1 })
    , toTokens(std::vector<uint32_t>{ 1, 0 })
    , transformer({ .base = modelCfg.encoderCfg })
    {
        const uint32_t D = modelCfg.embedDim(), H = modelCfg.numHeads(), Dh = modelCfg.headDim();
        const uint32_t L = modelCfg.decoderCfg.nBlocks;

        // clang-format off
        input("inMel") - conv1 - conv2 - dropHeight - toTokens - "in0"/addPositions - transformer;
        input("inPositions") - "in1"/addPositions;
        // clang-format on

        // Built in place: wired nodes must not move
        for (auto* v : { &crossK, &crossV })
            v->reserve(L);
        for (auto* v : { &splitK, &splitV })
            v->reserve(L);
        for (auto* v : { &perK, &perV })
            v->reserve(L);

        for (uint32_t i = 0; i < L; ++i)
        {
            crossK.emplace_back(vai::node::Linear::Config{ .inDim = D, .outDim = D });
            crossV.emplace_back(vai::node::Linear::Config{ .inDim = D, .outDim = D });
            splitK.emplace_back(std::vector<int64_t>{ -1, H, Dh });
            splitV.emplace_back(std::vector<int64_t>{ -1, H, Dh });
            perK.emplace_back(std::vector<uint32_t>{ 1, 0, 2 });
            perV.emplace_back(std::vector<uint32_t>{ 1, 0, 2 });

            // clang-format off
            transformer - crossK[i] - splitK[i] - perK[i] - output("crossK" + std::to_string(i));
            transformer - crossV[i] - splitV[i] - perV[i] - output("crossV" + std::to_string(i));
            // clang-format on
        }

        bidirectionalMask = vai::createCausalMask(device(), 0);
        transformer.setCausalMask(bidirectionalMask);
    }

    // Outputs land in keys / values, which stay valid until the next encode()
    void encode(const vai::Tensor& mel, std::vector<vai::Tensor>& keys, std::vector<vai::Tensor>& values)
    {
        const uint32_t L = modelCfg.decoderCfg.nBlocks;
        keys.resize(L);
        values.resize(L);

        vai::TensorRefDict outs;
        for (uint32_t i = 0; i < L; ++i)
        {
            outs["crossK" + std::to_string(i)] = &keys[i];
            outs["crossV" + std::to_string(i)] = &values[i];
        }
        (*this)({ { "inMel", mel }, { "inPositions", positions } }, outs);
    }
};

}   // namespace encoder


namespace decoder {


/////////////////////////////////////////////////////////////////////////////////////////
// decoder::Layer - causal self-attention, cross-attention, MLP, each pre-LayerNorm
// Input:  in0 [T, D]; crossK / crossV [H, Tk, Dh] from the encoder
// Output: out0 [T, D]
//
// out0 = ffn(ffnNorm(h2)) + h2
//   h2 = crossAttn(crossNorm(h1), crossK, crossV) + h1
//   h1 = selfAttn(selfNorm(in0)) + in0
//
// Each residual add is folded into its branch's last projection.
/////////////////////////////////////////////////////////////////////////////////////////
struct Layer : public vai::NodeGroup
{
    using MultiHeadAttention = vai::group::MultiHeadAttention<vai::QkvLayout::Separate>;

    vai::node::Identity input;
    vai::node::LayerNorm selfNorm;
    MultiHeadAttention selfAttn;
    vai::node::LayerNorm crossNorm;
    vai::node::Linear crossQ;         // [T, D]
    vai::node::Reshape splitQ;        // [T, H, Dh]
    vai::node::Permute perQ;          // [H, T, Dh]
    vai::node::Attention crossAttn;   // [T, H, Dh]
    vai::node::Reshape mergeOut;      // [T, D]
    vai::node::Linear crossOut;
    vai::node::LayerNorm ffnNorm;
    vai::group::FeedForward ffn;

    Layer(const vai::TransformerConfig& base, uint32_t kvCacheSize, vai::Dtype kvCacheDtype)
    : selfNorm({ .eps = base.normEps })
    , selfAttn({
          .embedDim = base.embedDim,
          .headDim = base.headDim,
          .numQHeads = base.numQHeads,
          .numKVHeads = base.numKVHeads,
          .kvCacheSize = kvCacheSize,
          .kvCacheDtype = kvCacheDtype,
          .acceptResidual = true,
      })
    , crossNorm({ .eps = base.normEps })
    , crossQ({ .inDim = base.embedDim, .outDim = base.embedDim })
    , splitQ(std::vector<int64_t>{ -1, base.numQHeads, base.headDim })
    , perQ(std::vector<uint32_t>{ 1, 0, 2 })
    , crossAttn({ .headDim = base.headDim, .numQHeads = base.numQHeads })
    , mergeOut(std::vector<int64_t>{ -1, base.embedDim })
    , crossOut({ .inDim = base.embedDim, .outDim = base.embedDim, .acceptResidual = true })
    , ffnNorm({ .eps = base.normEps })
    , ffn({
          .inDim = base.embedDim,
          .hiddenDim = base.ffnHiddenDim,
          .outDim = base.embedDim,
          .actType = base.ffnActType,
          .acceptResidual = true,
      })
    {
        // clang-format off
        input - selfNorm - selfAttn - crossNorm - crossQ - splitQ - perQ - "Q"/crossAttn - mergeOut - crossOut - ffnNorm - ffn;
        input - "residual"/selfAttn - "residual"/crossOut - "residual"/ffn;
        // clang-format on

        defineSlot("in0", input.slot("in0"));
        defineSlot("crossK", crossAttn.slot("K"));
        defineSlot("crossV", crossAttn.slot("V"));
        defineSlot("out0", ffn.slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// decoder::Net - tokens + cross K / V -> next token (greedy)
// Input:  inTokenIds [T]; crossK<i> / crossV<i> [H, audioContext, Dh]
// Output: nextTokenId [1]
/////////////////////////////////////////////////////////////////////////////////////////
struct Net : public vai::NeuralNet
{
    ModelConfig modelCfg;

    vai::node::TokenEmbedding embedding;   // token + learned position
    std::vector<Layer> layers;
    vai::node::LayerNorm finalNorm;
    vai::node::Slice lastRow;   // only the last position predicts
    vai::node::Linear lmHead;   // tied to the token embedding
    vai::node::Argmax argmax;

    eva::Buffer kvCacheCounter;
    eva::Buffer causalMask;

    Net(const vai::NeuralNetConfig& netCfg, const ModelConfig& modelCfg, const RuntimeConfig& runtimeCfg = {})
    : NeuralNet(netCfg)
    , modelCfg(modelCfg)
    , embedding({
          .vocabSize = modelCfg.vocabSize,
          .embedDim = modelCfg.embedDim(),
          .posTableSize = modelCfg.textContext,
      })
    , finalNorm({ .eps = modelCfg.decoderCfg.normEps })
    , lastRow({ .start = -1 })
    , lmHead({ .inDim = modelCfg.embedDim(), .outDim = modelCfg.vocabSize })
    {
        const uint32_t L = modelCfg.decoderCfg.nBlocks;

        layers.reserve(L);
        for (uint32_t i = 0; i < L; ++i)
            layers.emplace_back(modelCfg.decoderCfg, modelCfg.textContext, runtimeCfg.kvCacheDtype);

        // clang-format off
        input("inTokenIds") - embedding - layers[0];
        for (size_t i = 1; i < layers.size(); ++i)
            layers[i - 1] - layers[i];
        layers.back() - finalNorm - lastRow - lmHead - argmax - output("nextTokenId");

        for (uint32_t i = 0; i < L; ++i)
        {
            input("crossK" + std::to_string(i)) - "crossK"/layers[i];
            input("crossV" + std::to_string(i)) - "crossV"/layers[i];
        }
        // clang-format on

        kvCacheCounter = vai::createKvCacheCounter(device());
        embedding.setKvCacheCounter(kvCacheCounter);
        for (auto& layer : layers)
            layer.selfAttn.setKvCacheCounter(kvCacheCounter);

        causalMask = vai::createCausalMask(device(), 1);
        for (auto& layer : layers)
            layer.selfAttn.attn.setCausalMask(causalMask);
    }

    uint32_t tokensSoFar = 0;

    // Feeds inTokenIds [T] after the cached ones; nextTokenId [1] gets the prediction
    void step(
        const vai::Tensor& inTokenIds, vai::Tensor* nextTokenId, const std::vector<vai::Tensor>& keys,
        const std::vector<vai::Tensor>& values)
    {
        VAI_ASSERT(nextTokenId);
        VAI_ASSERT(keys.size() == layers.size() && values.size() == layers.size());
        VAI_ASSERT(tokensSoFar + inTokenIds.shape()[0] <= modelCfg.textContext);

        vai::TensorDict ins = { { "inTokenIds", inTokenIds } };
        for (size_t i = 0; i < layers.size(); ++i)
        {
            ins["crossK" + std::to_string(i)] = keys[i];
            ins["crossV" + std::to_string(i)] = values[i];
        }

        sync();   // The GPU may still be reading kvCacheCounter.
        *(uint32_t*)kvCacheCounter.map() = tokensSoFar;
        tokensSoFar += inTokenIds.shape()[0];

        (*this)(ins, { { "nextTokenId", nextTokenId } }, false);
    }

    void resetCache()
    {
        sync();   // The GPU may still be reading kvCacheCounter.
        tokensSoFar = 0;
        *(uint32_t*)kvCacheCounter.map() = 0;
    }
};

}   // namespace decoder


/////////////////////////////////////////////////////////////////////////////////////////
// Net - the two nets and the cross K / V they share
//
// The encoder runs once per 30 s window; the decoder once per token. The cross
// K / V are encoder outputs passed to the decoder as inputs, so they stay on the
// GPU and keep their buffers, and the decoder never re-records between windows.
/////////////////////////////////////////////////////////////////////////////////////////
struct Net
{
    ModelConfig modelCfg;

private:
    // One pipeline cache for both nets, declared first so it exists before them
    std::shared_ptr<vai::PipelinePool> pipelines_;

public:
    encoder::Net encoder;
    decoder::Net decoder;

    std::vector<vai::Tensor> crossK, crossV;

    Net(const vai::NeuralNetConfig& netCfg, const ModelConfig& modelCfg, const RuntimeConfig& runtimeCfg = {})
    : modelCfg(modelCfg)
    , encoder(sharePipelines(netCfg), modelCfg)
    , decoder(sharePipelines(netCfg), modelCfg, runtimeCfg)
    {
    }

    // Input: mel [numMels, 1, 3000]. Starts a new transcript.
    void encode(const vai::Tensor& mel)
    {
        decoder.resetCache();   // also waits until the decoder is done with the old K / V
        encoder.encode(mel, crossK, crossV);
    }

    // Input: promptTokenIds [T]. Output: nextTokenId [1], the first generated token.
    void prefill(const vai::Tensor& promptTokenIds, vai::Tensor* nextTokenId)
    {
        decoder.step(promptTokenIds, nextTokenId, crossK, crossV);
    }

    // Input/output: inoutTokenId [1], previous token replaced by the next token.
    void decode(vai::Tensor* inoutTokenId)
    {
        VAI_ASSERT(inoutTokenId);
        decoder.step(*inoutTokenId, inoutTokenId, crossK, crossV);
    }

    void resetProfileRecords()
    {
        encoder.resetProfileRecords();
        decoder.resetProfileRecords();
    }

private:
    // Buffer pools stay apart: a shared pool could alias live buffers across nets
    vai::NeuralNetConfig sharePipelines(vai::NeuralNetConfig cfg)
    {
        if (!cfg.device)
            cfg.device = vai::getVaiDevice();
        if (!cfg.pipelinePool)
        {
            if (!pipelines_)
                pipelines_ = std::make_shared<vai::PipelinePool>(cfg.device);
            cfg.pipelinePool = pipelines_;
        }
        return cfg;
    }
};


}   // namespace model::experimental::whisper

#endif   // EXPERIMENTAL_WHISPER_NET_H
