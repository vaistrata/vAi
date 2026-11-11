/*=============================================================================
  Written by   ippx123@gmail.com
  Model        Z-Image caption encoder (Qwen3-4B, the pipeline's text_encoder)
  Modules      TextEncoderConfig, TextEncoder, EmbeddingTable, encodeCaption
=============================================================================*/

#ifndef ZIMAGE_TEXT_ENCODER_H
#define ZIMAGE_TEXT_ENCODER_H

#include "vai/groups/transformer.h"
#include "vai/utils/tokenizer.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>


namespace zimage {

/*
  ZImagePipeline.encode_prompt: the prompt in Qwen3's chat template
      <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
  (enable_thinking, so no empty <think> block), at most 512 tokens, through
  Qwen3ForCausalLM; the caption features are hidden_states[-2], the output of
  the second-to-last layer before the final norm, padding dropped. The
  pipeline pads to 512 on the right, and causal attention keeps padding from
  reaching the real tokens, so running the real tokens alone gives the same.
  The last layer, the final norm and the LM head never run and are not loaded.
*/

/////////////////////////////////////////////////////////////////////////////////////////
// TextEncoderConfig - Qwen3-4B (Tongyi-MAI/Z-Image-Turbo text_encoder/config.json)
/////////////////////////////////////////////////////////////////////////////////////////
struct TextEncoderConfig
{
    uint32_t vocabSize = 151936;
    uint32_t embedDim = 2560;
    uint32_t headDim = 128;
    uint32_t numQHeads = 32;
    uint32_t numKVHeads = 8;
    uint32_t ffnHiddenDim = 9728;
    uint32_t numLayers = 35;            // of 36: hidden_states[-2]
    float normEps = 1e-6f;
    float ropeTheta = 1000000.0f;
    uint32_t maxTokens = 512;           // the pipeline's max_sequence_length
};

struct TextEncoderRuntimeConfig
{
    // The attention and FFN projections' weights: Q4 (~2.0 GB for 35 layers)
    // or fp16 (~7 GB). The norms stay fp32.
    vai::Dtype projDtype = vai::Dtype::Q4;
};


/////////////////////////////////////////////////////////////////////////////////////////
// EmbeddingTable - token embedding rows, read from their file as needed
//
// A prompt needs a few hundred of its 152K rows (0.8 GB in fp16), so the table
// stays in the safetensors file it came from.
/////////////////////////////////////////////////////////////////////////////////////////
struct EmbeddingTable
{
    std::shared_ptr<const void> owner;  // keeps the file open
    vai::SafetensorRef ref{};

    // [ids.size(), embedDim] fp32, row by row.
    std::vector<float> rows(const std::vector<int>& ids) const
    {
        const uint32_t D = ref.getShape()[1];
        std::vector<float> out;
        out.reserve(ids.size() * D);
        for (int id : ids)
        {
            const vai::Tensor row = vai::slice(ref, uint32_t(id), uint32_t(id) + 1, vai::Dtype::Fp32);
            const float* p = static_cast<const float*>(row.hostData());
            out.insert(out.end(), p, p + D);
        }
        return out;
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// TextEncoder - token embeddings -> caption features
//
// Input    inEmbeds     [T, embedDim]  the tokens' embedding rows (EmbeddingTable)
// Output   outFeatures  [T, embedDim]  the output of layer numLayers - 1
//
// Qwen3's decoder layers, as vai's pre-RMSNorm layer with a per-head RMS norm
// on Q and K: GQA, RoPE (rotate-half, theta 1e6), SwiGLU, causal, no KV cache.
/////////////////////////////////////////////////////////////////////////////////////////
struct TextEncoder : public vai::NeuralNet
{
    using Layer = vai::group::TransformerLayer<vai::TransformerType::Default, vai::QkvLayout::Separate>;

    TextEncoderConfig encCfg;
    TextEncoderRuntimeConfig runtimeCfg;
    std::vector<Layer> layers;
    EmbeddingTable embedding;
    eva::Buffer causalMask;

    static vai::TransformerLayerConfig layerConfig(const TextEncoderConfig& c)
    {
        return {
            .embedDim = c.embedDim,
            .headDim = c.headDim,
            .numQHeads = c.numQHeads,
            .numKVHeads = c.numKVHeads,
            .kvCacheDtype = vai::Dtype::Fp16,   // unused: no cache
            .withQkNorm = true,
            .ropeCfg = vai::RopeConfig{.type = vai::RopeType::Default, .base = c.ropeTheta},
            .ffnHiddenDim = c.ffnHiddenDim,
            .ffnActivation = vai::Epilogue::Silu,
            .normEps = c.normEps,
        };
    }

    TextEncoder(const vai::NeuralNetConfig& netCfg, const TextEncoderConfig& cfg = {},
                const TextEncoderRuntimeConfig& runtimeCfg = {})
    : NeuralNet(netCfg)
    , encCfg(cfg)
    , runtimeCfg(runtimeCfg)
    {
        layers.reserve(cfg.numLayers);
        for (uint32_t i = 0; i < cfg.numLayers; ++i)
        {
            Layer& layer = layers.emplace_back(layerConfig(cfg));
            for (vai::node::Linear* lin : {&layer.mhAttn.toQ, &layer.mhAttn.toK, &layer.mhAttn.toV,
                                           &layer.mhAttn.toOut, &layer.ffn.gateValue, &layer.ffn.out})
                lin->config_.wDtype = runtimeCfg.projDtype;
        }

        input("inEmbeds") - layers[0];
        for (size_t i = 1; i < layers.size(); ++i)
            layers[i - 1] - layers[i];
        layers.back() - output("outFeatures");

        causalMask = vai::createCausalMask(device(), 1);
        for (Layer& layer : layers)
            layer.mhAttn.attn.setCausalMask(causalMask);
    }

    // The app builds an encoder per prompt, so nothing it allocates may outlive it.
    ~TextEncoder()
    {
        sync();
        causalMask.destroy();
    }
};


// The pipeline's prompt template; at most maxTokens tokens.
inline std::vector<int> captionTokens(const TextEncoderConfig& cfg, const Tokenizer& tok,
                                      const std::string& prompt)
{
    std::vector<int> ids = tok.encode("<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n");
    if (ids.size() > cfg.maxTokens)
        ids.resize(cfg.maxTokens);
    return ids;
}

// Caption features [T, embedDim] fp32 of these tokens, on the host.
inline vai::Tensor encodeCaption(TextEncoder& enc, const std::vector<int>& ids)
{
    const uint32_t T = uint32_t(ids.size()), D = enc.encCfg.embedDim;
    vai::Tensor out;
    enc({{"inEmbeds", vai::Tensor({T, D}).setFloats(enc.embedding.rows(ids))}}, {{"outFeatures", &out}});
    const float* p = out.getData<float>();
    return vai::Tensor({T, D}).setFloats(std::vector<float>(p, p + size_t(T) * D));
}

} // namespace zimage

#endif // ZIMAGE_TEXT_ENCODER_H
