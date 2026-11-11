/*=============================================================================
  Written by   oneway@vaistrata.com
  Checkpoint   Each size reads data/ckpts/llama3/meta/<ModelConfig::variant>/model.safetensors,
               taken from the matching Instruct repo on HuggingFace:
                 3.2-1B  meta-llama/Llama-3.2-1B-Instruct
                      or unsloth/Llama-3.2-1B-Instruct
                 3.2-3B  meta-llama/Llama-3.2-3B-Instruct
                      or unsloth/Llama-3.2-3B-Instruct
=============================================================================*/

#ifndef PRODUCTION_LLAMA3_META_WEIGHTS_H
#define PRODUCTION_LLAMA3_META_WEIGHTS_H

#include "net.h"
#include "vai/engine/weights.h"
#include <string>


namespace model::production::llama3::meta {

inline void loadWeights(Net& net, const vai::Checkpoint& ckpt)
{
    // Weights load at the net's wDefaultDtype, norm coefficients at Fp32.
    const vai::Dtype wDt = net.config().wDefaultDtype;
    const vai::Dtype normDt = vai::Dtype::Fp32;

    // Each tensor is loaded at the dtype declared by its owning node.
    auto loadLin = [&](auto& lin, const std::string& wKey) {
        // Llama3 linear layers carry no bias: the "bias" slot stays empty.
        lin["weight"] = vai::Tensor(ckpt[wKey], wDt);
    };
    auto loadRMSNorm = [&](auto& rn, const std::string& wKey) { rn["scale"] = vai::Tensor(ckpt[wKey], normDt); };

    // Token embedding
    net.embedding["weight"] = vai::Tensor(ckpt["model.embed_tokens.weight"], wDt);

    // Layer layers
    for (uint32_t i = 0; i < net.modelCfg.languageCfg.numLayers; ++i)
    {
        std::string prefix = "model.layers." + std::to_string(i) + ".";
        auto& layer = net.decoder.layers[i];

        // RMSNorms (params loaded at Fp32)
        loadRMSNorm(layer.norm1, prefix + "input_layernorm.weight");
        loadRMSNorm(layer.norm2, prefix + "post_attention_layernorm.weight");

        // Self attention projections (no bias in Llama3)
        loadLin(layer.mhAttn.projQ, prefix + "self_attn.q_proj.weight");
        loadLin(layer.mhAttn.projK, prefix + "self_attn.k_proj.weight");
        loadLin(layer.mhAttn.projV, prefix + "self_attn.v_proj.weight");
        loadLin(layer.mhAttn.projOut, prefix + "self_attn.o_proj.weight");

        // FFN projections (SwiGLU). gate and value are one projection here, so
        // their rows are stacked — gate first, the half GatedMul activates.
        {
            layer.ffn.gateValue["weight"] =
                vai::concat({ckpt[prefix + "mlp.gate_proj.weight"], ckpt[prefix + "mlp.up_proj.weight"]}, wDt);
        }
        loadLin(layer.ffn.out, prefix + "mlp.down_proj.weight");
    }

    // Final RMSNorm
    loadRMSNorm(net.decoder.finalNorm, "model.norm.weight");

    // LM Head. Llama 3.2 ties it to the token embedding, so the checkpoint
    // carries no lm_head tensor of its own.
    net.lmHead["weight"] = net.embedding["weight"];
}

}   // namespace model::production::llama3::meta


// loadWeights<"meta">(net) — composes the path in Checkpoint and loads it.
template <>
inline void loadWeights<"meta", model::production::llama3::Net>(model::production::llama3::Net& net)
{
    std::string dir = std::string(VAI_DATA_DIR "/ckpts/llama3/meta/") + net.modelCfg.variant;
    model::production::llama3::meta::loadWeights(net, vai::Checkpoint(dir));
}

#endif   // PRODUCTION_LLAMA3_META_WEIGHTS_H
