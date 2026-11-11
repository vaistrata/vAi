/*=============================================================================
  Written by   chaudharyshivansh9@gmail.com
  Checkpoint   Each size reads data/whisper/ckpts/<ModelConfig::variant>/model.safetensors,
               taken from the matching repo on HuggingFace:
                 tiny             openai/whisper-tiny
                 base             openai/whisper-base
                 small            openai/whisper-small
                 medium           openai/whisper-medium
                 large-v3         openai/whisper-large-v3
                 large-v3-turbo   openai/whisper-large-v3-turbo
=============================================================================*/

#ifndef WHISPER_OPENAI_WEIGHTS_H
#define WHISPER_OPENAI_WEIGHTS_H

#include "model.h"
#include "vai/engine/weights.h"
#include <string>
#include <vector>


namespace whisper::openai {

// HF Whisper uses nn.Linear, already [out, in]: nothing is transposed.
inline void loadWeights(Model& model, const vai::Checkpoint& ckpt)
{
    auto loadLin = [&](vai::node::Linear& lin, const std::string& key)
    {
        vai::Dtype dt = lin.config_.wDtype;
        lin["weight"] = vai::Tensor(ckpt[key + ".weight"], dt);
        if (lin.config_.hasBias)
            lin["bias"] = vai::Tensor(ckpt[key + ".bias"], dt);
    };

    // k_proj ships no bias but sits in a projection set that has one
    auto loadLinZeroBias = [&](vai::node::Linear& lin, const std::string& key)
    {
        vai::Dtype dt = lin.config_.wDtype;
        lin["weight"] = vai::Tensor(ckpt[key + ".weight"], dt);
        lin["bias"]   = vai::Tensor(std::vector<uint32_t>{lin.config_.outDim}, dt).setFloats(std::vector<float>(lin.config_.outDim, 0.0f));
    };

    auto loadLN = [&](vai::node::LayerNorm& ln, const std::string& key)
    {
        vai::Dtype dt = ln.config_.wDtype;
        ln["gamma"] = vai::Tensor(ckpt[key + ".weight"], dt);
        ln["beta"]  = vai::Tensor(ckpt[key + ".bias"],   dt);
    };

    // Conv1d [Co, Ci, 3] -> the middle row of a 3x3 kernel [Co, Ci, 3, 3]
    auto loadConv = [&](vai::node::Convolution& conv, const std::string& key)
    {
        vai::Tensor w1d(ckpt[key + ".weight"], vai::Dtype::Fp32);
        const auto& s = w1d.shape();
        VAI_ASSERT(s.size() == 3 && s[2] == 3);

        const float* src = (const float*)w1d.hostData();
        std::vector<float> w2d(size_t(s[0]) * s[1] * 9, 0.0f);
        for (size_t oc = 0; oc < size_t(s[0]) * s[1]; ++oc)
            for (uint32_t kx = 0; kx < 3; ++kx)
                w2d[oc * 9 + 3 + kx] = src[oc * 3 + kx];

        vai::Dtype dt = conv.config_.wDtype;
        conv["weight"] = vai::Tensor(std::vector<uint32_t>{s[0], s[1], 3, 3}, dt).setFloats(w2d);
        conv["bias"]   = vai::Tensor(ckpt[key + ".bias"], dt);
    };

    // Encoder
    auto& enc = model.encoder;
    loadConv(enc.conv1, "model.encoder.conv1");
    loadConv(enc.conv2, "model.encoder.conv2");
    enc.positions = vai::Tensor(ckpt["model.encoder.embed_positions.weight"], vai::Dtype::Fp32);

    for (uint32_t i = 0; i < model.modelCfg.encoderCfg.numLayers; ++i)
    {
        std::string prefix = "model.encoder.layers." + std::to_string(i) + ".";
        auto& layer = enc.transformer.layers[i];

        loadLN(layer.norm1, prefix + "self_attn_layer_norm");
        loadLin(layer.mhAttn.toQ, prefix + "self_attn.q_proj");
        loadLinZeroBias(layer.mhAttn.toK, prefix + "self_attn.k_proj");
        loadLin(layer.mhAttn.toV, prefix + "self_attn.v_proj");
        loadLin(layer.mhAttn.toOut, prefix + "self_attn.out_proj");

        loadLN(layer.norm2, prefix + "final_layer_norm");
        loadLin(layer.ffn.fc1, prefix + "fc1");
        loadLin(layer.ffn.fc2, prefix + "fc2");
    }
    loadLN(enc.transformer.finalNorm, "model.encoder.layer_norm");

    // Decoder, and the cross K / V projections the encoder runs for it
    auto& dec = model.decoder;
    vai::Dtype edt = dec.embedding.config_.wDtype;
    dec.embedding["weight"]    = vai::Tensor(ckpt["model.decoder.embed_tokens.weight"], edt);
    dec.embedding["posWeight"] = vai::Tensor(ckpt["model.decoder.embed_positions.weight"], edt);

    for (uint32_t i = 0; i < model.modelCfg.decoderCfg.numLayers; ++i)
    {
        std::string prefix = "model.decoder.layers." + std::to_string(i) + ".";
        auto& layer = dec.layers[i];

        loadLN(layer.selfNorm, prefix + "self_attn_layer_norm");
        loadLin(layer.selfAttn.toQ, prefix + "self_attn.q_proj");
        loadLinZeroBias(layer.selfAttn.toK, prefix + "self_attn.k_proj");
        loadLin(layer.selfAttn.toV, prefix + "self_attn.v_proj");
        loadLin(layer.selfAttn.toOut, prefix + "self_attn.out_proj");

        loadLN(layer.crossNorm, prefix + "encoder_attn_layer_norm");
        loadLin(layer.crossQ, prefix + "encoder_attn.q_proj");
        loadLin(enc.crossK[i], prefix + "encoder_attn.k_proj");   // no bias
        loadLin(enc.crossV[i], prefix + "encoder_attn.v_proj");
        loadLin(layer.crossOut, prefix + "encoder_attn.out_proj");

        loadLN(layer.ffnNorm, prefix + "final_layer_norm");
        loadLin(layer.ffn.fc1, prefix + "fc1");
        loadLin(layer.ffn.fc2, prefix + "fc2");
    }
    loadLN(dec.finalNorm, "model.decoder.layer_norm");

    // LM head (weight tying with token embeddings)
    VAI_ASSERT(dec.lmHead.config_.wDtype == edt);
    dec.lmHead["weight"] = dec.embedding["weight"];
}

} // namespace whisper::openai


// loadWeights<"openai">(model) — composes the path in Checkpoint and loads it.
template<>
inline void loadWeights<"openai", whisper::Model>(whisper::Model& model)
{
    std::string dir = std::string(VAI_DATA_DIR "/whisper/ckpts/") + model.modelCfg.variant;
    whisper::openai::loadWeights(model, vai::Checkpoint(dir));
}

#endif // WHISPER_OPENAI_WEIGHTS_H
