/*=============================================================================
  Written by   oneway@vaistrata.com
  Checkpoint   Each size reads data/ckpts/gpt2/openai/<ModelConfig::variant>/model.safetensors,
               taken from the matching repo on HuggingFace:
                 124M   openai-community/gpt2
                 355M   openai-community/gpt2-medium
                 774M   openai-community/gpt2-large
                 1.5B   openai-community/gpt2-xl
=============================================================================*/

#ifndef PRODUCTION_GPT2_OPENAI_WEIGHTS_H
#define PRODUCTION_GPT2_OPENAI_WEIGHTS_H

#include "net.h"
#include "vai/engine/weights.h"
#include <string>


namespace model::production::gpt2::openai {

// HuggingFace GPT-2's checkpoint stores every linear transposed (PyTorch
// Conv1D: [in, out]); vAi wants [out, in], so loadLin below transposes each.
// The fused c_attn projection loads as-is — the model keeps Q, K and V fused.
inline void loadWeights(Net& net, const vai::Checkpoint& ckpt)
{
    // Weights load at the net's wDefaultDtype, norm coefficients at Fp32.
    const vai::Dtype wDt = net.config().wDefaultDtype;
    const vai::Dtype normDt = vai::Dtype::Fp32;

    auto loadLin = [&](vai::node::Linear& lin, const std::string& key) {
        lin["weight"] = vai::Tensor(ckpt[key + ".weight"], wDt).permute(1, 0);
        lin["bias"] = vai::Tensor(ckpt[key + ".bias"], wDt);
    };

    auto loadLN = [&](auto& ln, const std::string& key) {
        ln["scale"] = vai::Tensor(ckpt[key + ".weight"], normDt);
        ln["shift"] = vai::Tensor(ckpt[key + ".bias"], normDt);
    };

    // Token and position embeddings are already [count, embedDim].
    net.embedding["weight"] = vai::Tensor(ckpt["wte.weight"], wDt);
    net.embedding["posWeight"] = vai::Tensor(ckpt["wpe.weight"], wDt);

    // Layer layers
    for (uint32_t i = 0; i < net.modelCfg.languageCfg.numLayers; ++i)
    {
        std::string prefix = "h." + std::to_string(i) + ".";
        auto& layer = net.decoder.layers[i];

        loadLN(layer.norm1, prefix + "ln_1");
        loadLN(layer.norm2, prefix + "ln_2");

        loadLin(layer.mhAttn.projQKV, prefix + "attn.c_attn");
        loadLin(layer.mhAttn.projOut, prefix + "attn.c_proj");

        loadLin(layer.ffn.fc1, prefix + "mlp.c_fc");
        loadLin(layer.ffn.fc2, prefix + "mlp.c_proj");
    }

    // Final layer norm
    loadLN(net.decoder.finalNorm, "ln_f");

    // LM Head (weight tying with token embeddings)
    net.lmHead["weight"] = net.embedding["weight"];
}

}   // namespace model::production::gpt2::openai


// loadWeights<"openai">(net) — composes the path in Checkpoint and loads it.
template <>
inline void loadWeights<"openai", model::production::gpt2::Net>(model::production::gpt2::Net& net)
{
    std::string dir = std::string(VAI_DATA_DIR "/ckpts/gpt2/openai/") + net.modelCfg.variant;
    model::production::gpt2::openai::loadWeights(net, vai::Checkpoint(dir));
}

#endif   // PRODUCTION_GPT2_OPENAI_WEIGHTS_H
