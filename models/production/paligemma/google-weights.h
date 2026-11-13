/*=============================================================================
  Written by   oneway@vaistrata.com
  Checkpoint   Each resolution reads every *.safetensors file in
               data/ckpts/paligemma/google/<ModelConfig::variant>/, taken from the matching
               mix repo on HuggingFace:
                 3B-224  google/paligemma-3b-mix-224
                 3B-448  google/paligemma-3b-mix-448
=============================================================================*/

#ifndef PRODUCTION_PALIGEMMA_GOOGLE_WEIGHTS_H
#define PRODUCTION_PALIGEMMA_GOOGLE_WEIGHTS_H

#include "net.h"
#include "vai/engine/weights.h"
#include <string>
#include <vector>


namespace model::production::paligemma::google {

/////////////////////////////////////////////////////////////////////////////////////////
// Weight loading - maps HuggingFace safetensors key names to model members
/////////////////////////////////////////////////////////////////////////////////////////
inline void loadWeights(Net& net, const vai::Checkpoint& ckpt)
{
    // Weights load at the net's wDefaultDtype, norm coefficients at Fp32.
    const vai::Dtype wDt = net.config().wDefaultDtype;
    const vai::Dtype normDt = vai::Dtype::Fp32;

    // Each tensor is loaded at the dtype declared by its owning node.
    auto loadLin = [&](auto& lin, const std::string& prefix) {
        lin["weight"] = vai::Tensor(ckpt[prefix + ".weight"], wDt);
        lin["bias"] = vai::Tensor(ckpt[prefix + ".bias"], wDt);
    };
    // Gemma's projections carry no bias: the "bias" slot stays empty.
    auto loadLinNoBias = [&](auto& lin, const std::string& prefix) {
        lin["weight"] = vai::Tensor(ckpt[prefix + ".weight"], wDt);
    };
    auto loadLayerNorm = [&](auto& ln, const std::string& prefix) {
        ln["scale"] = vai::Tensor(ckpt[prefix + ".weight"], normDt);
        ln["shift"] = vai::Tensor(ckpt[prefix + ".bias"], normDt);
    };
    // Gemma stores the RMS scale centered on zero. The kernel scales by scale as
    // it is, so the +1 is folded in here — once, at load.
    auto loadRMSNorm = [&](auto& rn, const std::string& wKey) {
        vai::Tensor centered(ckpt[wKey], vai::Dtype::Fp32);
        const float* src = static_cast<const float*>(centered.hostData());

        std::vector<float> scale(src, src + centered.numElements());
        for (float& value : scale)
            value += 1.0f;

        rn["scale"] = vai::Tensor(centered.shape(), normDt).setFloats(std::move(scale));
    };

    // Vision tower (SigLIP)
    {
        const auto& modelCfg = net.modelCfg;
        const std::string root = "vision_tower.vision_model.";

        // The patch projection is stored as a [D, C, P, P] kernel, whose trailing axes
        // are the (c, kh, kw) order a patch row carries — so it loads as a matrix.
        auto& patch = net.patchEmbed;
        patch.project["weight"] =
            vai::Tensor(ckpt[root + "embeddings.patch_embedding.weight"], wDt)
                .reshape({
                    modelCfg.visionCfg.layerCfg.embedDim,
                    modelCfg.visionCfg.imageChannels * modelCfg.visionCfg.patchSize * modelCfg.visionCfg.patchSize,
                });
        patch.project["bias"] = vai::sum(
            ckpt[root + "embeddings.position_embedding.weight"], ckpt[root + "embeddings.patch_embedding.bias"], wDt);

        for (uint32_t i = 0; i < modelCfg.visionCfg.numLayers; ++i)
        {
            std::string prefix = root + "encoder.layers." + std::to_string(i) + ".";
            auto& layer = net.encoder.layers[i];

            loadLayerNorm(layer.norm1, prefix + "layer_norm1");
            loadLayerNorm(layer.norm2, prefix + "layer_norm2");

            loadLin(layer.mhAttn.projQ, prefix + "self_attn.q_proj");
            loadLin(layer.mhAttn.projK, prefix + "self_attn.k_proj");
            loadLin(layer.mhAttn.projV, prefix + "self_attn.v_proj");
            loadLin(layer.mhAttn.projOut, prefix + "self_attn.out_proj");

            loadLin(layer.ffn.fc1, prefix + "mlp.fc1");
            loadLin(layer.ffn.fc2, prefix + "mlp.fc2");
        }

        loadLayerNorm(net.encoder.finalNorm, root + "post_layernorm");
    }

    // Multimodal projector
    loadLin(net.projector, "multi_modal_projector.linear");

    // Token embedding. The export pads the table to 257216 rows; the 64 past the
    // vocabulary were never trained, so they never reach the tied head that would
    // otherwise sample them.
    net.embedding["weight"] =
        vai::slice(ckpt["language_model.model.embed_tokens.weight"], 0, net.modelCfg.languageCfg.vocabSize, wDt);

    // Layer layers
    for (uint32_t i = 0; i < net.modelCfg.languageCfg.numLayers; ++i)
    {
        std::string prefix = "language_model.model.layers." + std::to_string(i) + ".";
        auto& layer = net.decoder.layers[i];

        // RMSNorms (params loaded at Fp32)
        loadRMSNorm(layer.norm1, prefix + "input_layernorm.weight");
        loadRMSNorm(layer.norm2, prefix + "post_attention_layernorm.weight");

        // Self attention projections
        loadLinNoBias(layer.mhAttn.projQ, prefix + "self_attn.q_proj");
        loadLinNoBias(layer.mhAttn.projK, prefix + "self_attn.k_proj");
        loadLinNoBias(layer.mhAttn.projV, prefix + "self_attn.v_proj");
        loadLinNoBias(layer.mhAttn.projOut, prefix + "self_attn.o_proj");

        // FFN projections (GeGLU). gate and value are one projection here, so
        // their rows are stacked — gate first, the half GatedMul activates.
        {
            layer.ffn.gateValue["weight"] =
                vai::concat({ ckpt[prefix + "mlp.gate_proj.weight"], ckpt[prefix + "mlp.up_proj.weight"] }, wDt);
        }
        loadLinNoBias(layer.ffn.out, prefix + "mlp.down_proj");
    }

    // Final RMSNorm
    loadRMSNorm(net.decoder.finalNorm, "language_model.model.norm.weight");

    // LM Head. Gemma ties it to the token embedding, so the checkpoint carries
    // no lm_head tensor of its own. The embedding's sqrt(D) scale lives in the
    // node rather than in the weights, which is what keeps the tie sound.
    net.lmHead["weight"] = net.embedding["weight"];
}

}   // namespace model::production::paligemma::google


// loadWeights<"google">(net) — composes the path in Checkpoint and loads it.
template <>
inline void loadWeights<"google", model::production::paligemma::Net>(model::production::paligemma::Net& net)
{
    std::string dir = std::string(VAI_DATA_DIR "/ckpts/paligemma/google/") + net.modelCfg.variant;
    model::production::paligemma::google::loadWeights(net, vai::Checkpoint(dir));
}

#endif   // PRODUCTION_PALIGEMMA_GOOGLE_WEIGHTS_H
