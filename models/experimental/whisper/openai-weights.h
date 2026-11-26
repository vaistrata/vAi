/*=============================================================================
  Written by   Shivanshyyy (github.com/Shivanshyyy)
  Checkpoint   Each size reads data/ckpts/whisper/openai/<ModelConfig::variant>/model.safetensors,
               taken from the matching repo on HuggingFace:
                 tiny             openai/whisper-tiny
                 base             openai/whisper-base
                 small            openai/whisper-small
                 medium           openai/whisper-medium
                 large-v3         openai/whisper-large-v3
                 large-v3-turbo   openai/whisper-large-v3-turbo
               The tokenizer is data/tokenizers/whisper/<variant>/tokenizer.json,
               from the same repo.
=============================================================================*/

#ifndef EXPERIMENTAL_WHISPER_OPENAI_WEIGHTS_H
#define EXPERIMENTAL_WHISPER_OPENAI_WEIGHTS_H

#include "net.h"
#include "vai/engine/weights.h"
#include <string>
#include <vector>


namespace model::experimental::whisper::openai {

// HF Whisper uses nn.Linear, already [out, in]: nothing is transposed.
inline void loadWeights(Net& net, const vai::Checkpoint& ckpt)
{
    // Weights load at the net's wDefaultDtype, norm coefficients at Fp32.
    const vai::Dtype wDt = net.encoder.config().wDefaultDtype;
    const vai::Dtype normDt = vai::Dtype::Fp32;

    // A Linear's bias slot is used only when a tensor is assigned to it, so
    // k_proj, which ships no bias, simply leaves the slot empty.
    auto loadLin = [&](vai::node::Linear& lin, const std::string& key) {
        lin["weight"] = vai::Tensor(ckpt[key + ".weight"], wDt);
        if (ckpt.contains(key + ".bias"))
            lin["bias"] = vai::Tensor(ckpt[key + ".bias"], wDt);
    };

    auto loadLN = [&](auto& ln, const std::string& key) {
        ln["scale"] = vai::Tensor(ckpt[key + ".weight"], normDt);
        ln["shift"] = vai::Tensor(ckpt[key + ".bias"], normDt);
    };

    // Conv1d [Co, Ci, 3] -> the middle row of a 3x3 kernel [Co, Ci, 3, 3]
    auto loadConv = [&](vai::node::Convolution& conv, const std::string& key) {
        vai::Tensor w1d(ckpt[key + ".weight"], vai::Dtype::Fp32);
        const auto& s = w1d.shape();
        VAI_ASSERT(s.size() == 3 && s[2] == 3);

        const float* src = (const float*)w1d.hostData();
        std::vector<float> w2d(size_t(s[0]) * s[1] * 9, 0.0f);
        for (size_t oc = 0; oc < size_t(s[0]) * s[1]; ++oc)
            for (uint32_t kx = 0; kx < 3; ++kx)
                w2d[oc * 9 + 3 + kx] = src[oc * 3 + kx];

        conv["weight"] = vai::Tensor(std::vector<uint32_t>{ s[0], s[1], 3, 3 }, wDt).setFloats(w2d);
        conv["bias"] = vai::Tensor(ckpt[key + ".bias"], wDt);
    };

    // Encoder
    auto& enc = net.encoder;
    loadConv(enc.conv1, "model.encoder.conv1");
    loadConv(enc.conv2, "model.encoder.conv2");
    // Added to the conv output, so it takes that tensor's dtype (Add wants both inputs alike).
    enc.positions = vai::Tensor(ckpt["model.encoder.embed_positions.weight"], enc.config().oDefaultDtype);

    for (uint32_t i = 0; i < net.modelCfg.encoderCfg.nBlocks; ++i)
    {
        const std::string prefix = "model.encoder.layers." + std::to_string(i) + ".";
        auto& block = enc.transformer.blocks[i];

        loadLN(block.norm1, prefix + "self_attn_layer_norm");
        loadLin(block.mhAttn.projQ, prefix + "self_attn.q_proj");
        loadLin(block.mhAttn.projK, prefix + "self_attn.k_proj");
        loadLin(block.mhAttn.projV, prefix + "self_attn.v_proj");
        loadLin(block.mhAttn.projOut, prefix + "self_attn.out_proj");

        loadLN(block.norm2, prefix + "final_layer_norm");
        loadLin(block.ffn.fc1, prefix + "fc1");
        loadLin(block.ffn.fc2, prefix + "fc2");
    }
    loadLN(enc.transformer.finalNorm, "model.encoder.layer_norm");

    // Decoder, and the cross K / V projections the encoder runs for it
    auto& dec = net.decoder;
    dec.embedding["weight"] = vai::Tensor(ckpt["model.decoder.embed_tokens.weight"], wDt);
    dec.embedding["posWeight"] = vai::Tensor(ckpt["model.decoder.embed_positions.weight"], wDt);

    for (uint32_t i = 0; i < net.modelCfg.decoderCfg.nBlocks; ++i)
    {
        const std::string prefix = "model.decoder.layers." + std::to_string(i) + ".";
        auto& layer = dec.layers[i];

        loadLN(layer.selfNorm, prefix + "self_attn_layer_norm");
        loadLin(layer.selfAttn.projQ, prefix + "self_attn.q_proj");
        loadLin(layer.selfAttn.projK, prefix + "self_attn.k_proj");
        loadLin(layer.selfAttn.projV, prefix + "self_attn.v_proj");
        loadLin(layer.selfAttn.projOut, prefix + "self_attn.out_proj");

        loadLN(layer.crossNorm, prefix + "encoder_attn_layer_norm");
        loadLin(layer.crossQ, prefix + "encoder_attn.q_proj");
        loadLin(enc.crossK[i], prefix + "encoder_attn.k_proj");
        loadLin(enc.crossV[i], prefix + "encoder_attn.v_proj");
        loadLin(layer.crossOut, prefix + "encoder_attn.out_proj");

        loadLN(layer.ffnNorm, prefix + "final_layer_norm");
        loadLin(layer.ffn.fc1, prefix + "fc1");
        loadLin(layer.ffn.fc2, prefix + "fc2");
    }
    loadLN(dec.finalNorm, "model.decoder.layer_norm");

    // LM head (weight tying with token embeddings)
    dec.lmHead["weight"] = dec.embedding["weight"];
}

}   // namespace model::experimental::whisper::openai


// loadWeights<"openai">(net) — composes the path in Checkpoint and loads it.
template <>
inline void loadWeights<"openai", model::experimental::whisper::Net>(model::experimental::whisper::Net& net)
{
    std::string dir = std::string(VAI_DATA_DIR "/ckpts/whisper/openai/") + net.modelCfg.variant;
    model::experimental::whisper::openai::loadWeights(net, vai::Checkpoint(dir));
}

#endif   // EXPERIMENTAL_WHISPER_OPENAI_WEIGHTS_H
