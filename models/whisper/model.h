/*=============================================================================
  Written by   chaudharyshivansh9@gmail.com
  Model        Whisper (tiny / base / small / medium / large-v3 / large-v3-turbo)
  Modules      encoder::Model, decoder::Layer, decoder::Model, Model
=============================================================================*/

#ifndef WHISPER_MODEL_H
#define WHISPER_MODEL_H

#include "models/whisper/nodes/whisper-cross-attention.h"
#include "vai/groups/transformer.h"
#include "vai/groups/feed-forward.h"
#include "vai/groups/multi-head-attention.h"
#include "vai/nodes/convolution.h"
#include "vai/nodes/token-embedding.h"
#include "vai/nodes/layer-norm.h"
#include "vai/nodes/reshape.h"
#include "vai/nodes/permute.h"
#include "vai/nodes/argmax.h"
#include "vai/nodes/slice.h"
#include "vai/nodes/add.h"
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>


namespace whisper {


/////////////////////////////////////////////////////////////////////////////////////////
// ModelConfig - Whisper (tiny / base / small / medium / large-v3 / large-v3-turbo)
/////////////////////////////////////////////////////////////////////////////////////////
struct ModelConfig
{
    const char* variant;                    // checkpoint folder under data/whisper/ckpts/
    uint32_t numMels;
    uint32_t vocabSize;
    uint32_t audioContext = 1500;           // encoder positions: 30 s of 10 ms frames, halved by conv2
    uint32_t textContext = 448;             // decoder positions (learned table rows)
    vai::TransformerConfig encoderCfg;
    vai::TransformerConfig decoderCfg;      // its layerCfg also sizes cross-attention

    static ModelConfig tiny()         { return make("tiny",            384,  4,  4,  80, 51865); }
    static ModelConfig base()         { return make("base",            512,  6,  6,  80, 51865); }
    static ModelConfig small()        { return make("small",           768, 12, 12,  80, 51865); }
    static ModelConfig medium()       { return make("medium",         1024, 24, 24,  80, 51865); }
    static ModelConfig largeV3()      { return make("large-v3",       1280, 32, 32, 128, 51866); }
    static ModelConfig largeV3Turbo() { return make("large-v3-turbo", 1280, 32,  4, 128, 51866); }

    static ModelConfig byName(const std::string& name)
    {
        for (auto cfg : {tiny(), base(), small(), medium(), largeV3(), largeV3Turbo()})
            if (name == cfg.variant)
                return cfg;
        throw std::runtime_error("unknown Whisper variant: " + name);
    }

    uint32_t embedDim() const { return encoderCfg.layerCfg.embedDim; }
    uint32_t numHeads() const { return encoderCfg.layerCfg.numQHeads; }
    uint32_t headDim()  const { return encoderCfg.layerCfg.headDim; }

private:
    // Pre-LayerNorm layers with exact GELU; head dim is 64 at every size
    static vai::TransformerConfig stack(uint32_t numLayers, uint32_t embedDim)
    {
        return {
            .numLayers = numLayers,
            .layerCfg = {
                .embedDim = embedDim,
                .headDim = 64,
                .numQHeads = embedDim / 64,
                .numKVHeads = embedDim / 64,
                .qkvBias = true,            // k_proj has none; the loader gives it zeros
                .outBias = true,
                .ffnHiddenDim = embedDim * 4,
                .ffnActivation = vai::Epilogue::Gelu,
                .ffnBias = true}
        };
    }

    static ModelConfig make(const char* variant, uint32_t embedDim,
        uint32_t encoderLayers, uint32_t decoderLayers, uint32_t numMels, uint32_t vocabSize)
    {
        return {
            .variant = variant,
            .numMels = numMels,
            .vocabSize = vocabSize,
            .encoderCfg = stack(encoderLayers, embedDim),
            .decoderCfg = stack(decoderLayers, embedDim)
        };
    }
};


struct RuntimeConfig
{
    vai::Dtype kvCacheDtype = vai::Dtype::Fp16;   // decoder self-attention cache

    // Linear accumulation. vai defaults it to the weight dtype; fp16 sums lose
    // Whisper's activation outliers (13x larger encoder error measured on tiny).
    vai::Dtype linearAccDtype = vai::Dtype::Fp32;
};


// Linear reads accDtype in prepare(), so setting it after construction is safe
inline void setAccDtype(std::initializer_list<vai::node::Linear*> linears, vai::Dtype dtype)
{
    for (auto* lin : linears)
        lin->config_.accDtype = dtype;
}


namespace encoder {

using Transformer = vai::group::Transformer<
    vai::TransformerType::Classic,
    vai::QkvLayout::Separate
>;


/////////////////////////////////////////////////////////////////////////////////////////
// encoder::Model - log-mel -> encoder states -> every decoder layer's cross K / V
// Input:  inMel       [numMels, 1, 3000], one 30 s window
//         inPositions [audioContext, embedDim], the sinusoid table (no node holds a
//                     constant addend, so it rides in as an input)
// Output: crossK<i> / crossV<i> [H, audioContext, headDim], one pair per decoder layer
//
// Conv1d(k=3, pad=1) is a 3x3 same-padded Convolution over a 1-pixel-high image
// whose kernel rows above and below are zero (see openai-weights.h).
/////////////////////////////////////////////////////////////////////////////////////////
struct Model : public vai::NeuralNet
{
    ModelConfig modelCfg;

    vai::node::Convolution conv1;          // [M, 1, 3000] -> [D, 1, 3000]
    vai::node::Convolution conv2;          //              -> [D, 1, 1500]
    vai::node::Reshape dropHeight;         //              -> [D, 1500]
    vai::node::Permute toTokens;           //              -> [1500, D]
    vai::node::Add addPositions;
    Transformer transformer;

    std::vector<vai::node::Linear> crossK, crossV;          // [1500, D]
    std::vector<vai::node::Reshape> splitK, splitV;         // [1500, H, Dh]
    std::vector<vai::node::Permute> perK, perV;             // [H, 1500, Dh]

    vai::Tensor positions;                 // filled by the weight loader
    eva::Buffer bidirectionalMask;

    Model(
        const vai::NeuralNetConfig& netCfg,
        const ModelConfig& modelCfg,
        const RuntimeConfig& runtimeCfg = {})
    : NeuralNet(netCfg)
    , modelCfg(modelCfg)
    , conv1({
        .inChannels = modelCfg.numMels,
        .outChannels = modelCfg.embedDim(),
        .kernelSize = 3,
        .stride = 1,
        .epilogue = vai::Epilogue::Gelu})
    , conv2({
        .inChannels = modelCfg.embedDim(),
        .outChannels = modelCfg.embedDim(),
        .kernelSize = 3,
        .stride = 2,
        .epilogue = vai::Epilogue::Gelu})
    , dropHeight(std::vector<int64_t>{modelCfg.embedDim(), -1})
    , toTokens(std::vector<uint32_t>{1, 0})
    , transformer(modelCfg.encoderCfg.with({.kvCacheDtype = runtimeCfg.kvCacheDtype}))
    {
        const uint32_t D = modelCfg.embedDim(), H = modelCfg.numHeads(), Dh = modelCfg.headDim();
        const uint32_t L = modelCfg.decoderCfg.numLayers;

        input("inMel") - conv1 - conv2 - dropHeight - toTokens - "in0"/addPositions - transformer;
        input("inPositions")                                   - "in1"/addPositions;

        // Built in place: wired nodes must not move
        for (auto* v : {&crossK, &crossV}) v->reserve(L);
        for (auto* v : {&splitK, &splitV}) v->reserve(L);
        for (auto* v : {&perK, &perV})     v->reserve(L);

        for (uint32_t i = 0; i < L; ++i)
        {
            crossK.emplace_back(vai::node::Linear::Config{.inDim = D, .outDim = D});
            crossV.emplace_back(vai::node::Linear::Config{.inDim = D, .outDim = D, .hasBias = true});
            splitK.emplace_back(std::vector<int64_t>{-1, H, Dh});
            splitV.emplace_back(std::vector<int64_t>{-1, H, Dh});
            perK.emplace_back(std::vector<uint32_t>{1, 0, 2});
            perV.emplace_back(std::vector<uint32_t>{1, 0, 2});

            transformer - crossK[i] - splitK[i] - perK[i] - output("crossK" + std::to_string(i));
            transformer - crossV[i] - splitV[i] - perV[i] - output("crossV" + std::to_string(i));
        }

        bidirectionalMask = vai::createCausalMask(device(), 0);
        transformer.setCausalMask(bidirectionalMask);

        const vai::Dtype acc = runtimeCfg.linearAccDtype;
        for (auto& l : transformer.layers)
            setAccDtype({&l.mhAttn.toQ, &l.mhAttn.toK, &l.mhAttn.toV, &l.mhAttn.toOut, &l.ffn.fc1, &l.ffn.fc2}, acc);
        for (uint32_t i = 0; i < L; ++i)
            setAccDtype({&crossK[i], &crossV[i]}, acc);
    }

    // Outputs land in keys / values, which stay valid until the next encode()
    void encode(const vai::Tensor& mel, std::vector<vai::Tensor>& keys, std::vector<vai::Tensor>& values)
    {
        const uint32_t L = modelCfg.decoderCfg.numLayers;
        keys.resize(L);
        values.resize(L);

        vai::TensorRefDict outs;
        for (uint32_t i = 0; i < L; ++i)
        {
            outs["crossK" + std::to_string(i)] = &keys[i];
            outs["crossV" + std::to_string(i)] = &values[i];
        }
        (*this)({{"inMel", mel}, {"inPositions", positions}}, outs);
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
    vai::node::Linear crossQ;              // [T, D]
    vai::node::Reshape splitQ;             // [T, H, Dh]
    vai::node::Permute perQ;               // [H, T, Dh]
    node::CrossAttention crossAttn;        // [T, H, Dh]
    vai::node::Reshape mergeOut;           // [T, D]
    vai::node::Linear crossOut;
    vai::node::LayerNorm ffnNorm;
    vai::group::FeedForward ffn;

    Layer(const vai::TransformerLayerConfig& cfg)
    : selfNorm({.eps = cfg.normEps})
    , selfAttn({
        .embedDim = cfg.embedDim,
        .headDim = cfg.headDim,
        .numQHeads = cfg.numQHeads,
        .numKVHeads = cfg.numKVHeads,
        .kvCacheDtype = cfg.kvCacheDtype,
        .kvCacheSize = cfg.kvCacheSize,
        .qkvBias = cfg.qkvBias,
        .outBias = cfg.outBias,
        .acceptResidual = true})
    , crossNorm({.eps = cfg.normEps})
    , crossQ({.inDim = cfg.embedDim, .outDim = cfg.embedDim, .hasBias = true})
    , splitQ(std::vector<int64_t>{-1, cfg.numQHeads, cfg.headDim})
    , perQ(std::vector<uint32_t>{1, 0, 2})
    , crossAttn({.headDim = cfg.headDim, .numHeads = cfg.numQHeads})
    , mergeOut(std::vector<int64_t>{-1, cfg.embedDim})
    , crossOut({.inDim = cfg.embedDim, .outDim = cfg.embedDim, .hasBias = true, .acceptResidual = true})
    , ffnNorm({.eps = cfg.normEps})
    , ffn({
        .inDim = cfg.embedDim,
        .hiddenDim = cfg.ffnHiddenDim,
        .outDim = cfg.embedDim,
        .activation = cfg.ffnActivation,
        .hasBias = cfg.ffnBias,
        .acceptResidual = true})
    {
        input - selfNorm -            selfAttn - crossNorm - crossQ - splitQ - perQ - "Q"/crossAttn - mergeOut -            crossOut - ffnNorm -            ffn;
        input            - "residual"/selfAttn                                                                 - "residual"/crossOut           - "residual"/ffn;

        defineSlot("in0", input.slot("in0"));
        defineSlot("crossK", crossAttn.slot("K"));
        defineSlot("crossV", crossAttn.slot("V"));
        defineSlot("out0", ffn.slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// decoder::Model - tokens + cross K / V -> next token (greedy)
// Input:  inTokenIds [T]; crossK<i> / crossV<i> [H, audioContext, Dh]
// Output: nextTokenId [1]
/////////////////////////////////////////////////////////////////////////////////////////
struct Model : public vai::NeuralNet
{
    ModelConfig modelCfg;

    vai::node::TokenEmbedding embedding;   // token + learned position
    std::vector<Layer> layers;
    vai::node::LayerNorm finalNorm;
    vai::node::Slice lastRow;              // only the last position predicts
    vai::node::Linear lmHead;              // tied to the token embedding
    vai::node::Argmax argmax;

    eva::Buffer kvCacheCounter;
    eva::Buffer causalMask;

    Model(
        const vai::NeuralNetConfig& netCfg,
        const ModelConfig& modelCfg,
        const RuntimeConfig& runtimeCfg = {})
    : NeuralNet(netCfg)
    , modelCfg(modelCfg)
    , embedding({
        .vocabSize = modelCfg.vocabSize,
        .embedDim = modelCfg.embedDim(),
        .posTableSize = modelCfg.textContext})
    , finalNorm({.eps = modelCfg.decoderCfg.layerCfg.normEps})
    , lastRow({.start = -1})
    , lmHead({.inDim = modelCfg.embedDim(), .outDim = modelCfg.vocabSize})
    {
        const vai::TransformerConfig stackCfg = modelCfg.decoderCfg.with({
            .kvCacheDtype = runtimeCfg.kvCacheDtype,
            .kvCacheSize = modelCfg.textContext});

        layers.reserve(stackCfg.numLayers);
        for (uint32_t i = 0; i < stackCfg.numLayers; ++i)
            layers.emplace_back(stackCfg.layerCfg);

        input("inTokenIds") - embedding - layers[0];
        for (size_t i = 1; i < layers.size(); ++i)
            layers[i - 1] - layers[i];
        layers.back() - finalNorm - lastRow - lmHead - argmax - output("nextTokenId");

        for (uint32_t i = 0; i < stackCfg.numLayers; ++i)
        {
            input("crossK" + std::to_string(i)) - "crossK"/layers[i];
            input("crossV" + std::to_string(i)) - "crossV"/layers[i];
        }

        kvCacheCounter = vai::createKvCacheCounter(device());
        embedding.setKvCacheCounter(kvCacheCounter);
        for (auto& layer : layers)
            layer.selfAttn.setKvCacheCounter(kvCacheCounter);

        causalMask = vai::createCausalMask(device(), 1);
        for (auto& layer : layers)
            layer.selfAttn.attn.setCausalMask(causalMask);

        const vai::Dtype acc = runtimeCfg.linearAccDtype;
        for (auto& l : layers)
            setAccDtype({&l.selfAttn.toQ, &l.selfAttn.toK, &l.selfAttn.toV, &l.selfAttn.toOut,
                         &l.crossQ, &l.crossOut, &l.ffn.fc1, &l.ffn.fc2}, acc);
        setAccDtype({&lmHead}, acc);
    }

    uint32_t tokensSoFar_ = 0;

    // Feeds inTokenIds [T] after the cached ones; nextTokenId [1] gets the prediction
    void step(const vai::Tensor& inTokenIds, vai::Tensor* nextTokenId,
        const std::vector<vai::Tensor>& keys, const std::vector<vai::Tensor>& values)
    {
        VAI_ASSERT(nextTokenId);
        VAI_ASSERT(keys.size() == layers.size() && values.size() == layers.size());
        VAI_ASSERT(tokensSoFar_ + inTokenIds.shape()[0] <= modelCfg.textContext);

        vai::TensorDict ins = {{"inTokenIds", inTokenIds}};
        for (size_t i = 0; i < layers.size(); ++i)
        {
            ins["crossK" + std::to_string(i)] = keys[i];
            ins["crossV" + std::to_string(i)] = values[i];
        }

        sync();  // The GPU may still be reading kvCacheCounter.
        *(uint32_t*)kvCacheCounter.map() = tokensSoFar_;
        tokensSoFar_ += inTokenIds.shape()[0];

        (*this)(ins, {{"nextTokenId", nextTokenId}}, false);
    }

    void resetCache()
    {
        sync();  // The GPU may still be reading kvCacheCounter.
        tokensSoFar_ = 0;
        *(uint32_t*)kvCacheCounter.map() = 0;
    }
};

}   // namespace decoder


/////////////////////////////////////////////////////////////////////////////////////////
// Model - the two nets and the cross K / V they share
//
// The encoder runs once per 30 s window; the decoder once per token. The cross
// K / V are encoder outputs passed to the decoder as inputs, so they stay on the
// GPU and keep their buffers, and the decoder never re-records between windows.
/////////////////////////////////////////////////////////////////////////////////////////
struct Model
{
    ModelConfig modelCfg;

private:
    // One pipeline cache for both nets, declared first so it exists before them
    std::shared_ptr<vai::PipelinePool> pipelines_;

public:
    encoder::Model encoder;
    decoder::Model decoder;

    std::vector<vai::Tensor> crossK, crossV;

    Model(
        const vai::NeuralNetConfig& netCfg,
        const ModelConfig& modelCfg,
        const RuntimeConfig& runtimeCfg = {})
    : modelCfg(modelCfg)
    , encoder(sharePipelines(netCfg), modelCfg, runtimeCfg)
    , decoder(sharePipelines(netCfg), modelCfg, runtimeCfg)
    {}

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


}   // namespace whisper

#endif // WHISPER_MODEL_H
