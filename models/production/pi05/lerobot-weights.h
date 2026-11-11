/*=============================================================================
  Written by   oneway@vaistrata.com
  Checkpoint   Each policy reads every *.safetensors file in
               data/ckpts/pi05/lerobot/<ModelConfig::variant>/, taken from the matching
               LeRobot port on HuggingFace:
                 3B  lerobot/pi05_base
=============================================================================*/

#ifndef PRODUCTION_PI05_LEROBOT_WEIGHTS_H
#define PRODUCTION_PI05_LEROBOT_WEIGHTS_H

#include "net.h"
#include "vai/engine/weights.h"
#include "vai/nodes/time-embedding.h"
#include <cmath>
#include <string>
#include <vector>


namespace model::production::pi05::lerobot {

/////////////////////////////////////////////////////////////////////////////////////////
// Expert condition - every denoise step's adaRMS rows, computed once on the host
//
// The condition depends on the step alone: t = 1 - k / denoiseSteps, its sinusoid,
// the time MLP, then one dense per norm. So the time MLP and the 37 dense
// projections never reach the GPU; their outputs do, one table row per step.
// Each dense gives scale, shift and gate; the row stores 1 + scale, so the norm
// multiplies by it as it is.
/////////////////////////////////////////////////////////////////////////////////////////
inline std::vector<float> condTable(const Net& net, const vai::Checkpoint& ckpt)
{
    const auto& modelCfg = net.modelCfg;
    const uint32_t S = net.runtimeCfg.denoiseSteps;
    const uint32_t D = modelCfg.actionCfg.layerCfg.embedDim;
    const uint32_t L = modelCfg.actionCfg.numLayers;
    const uint32_t R = modelCfg.condRows();

    // out[s] = W x[s] + b for every step, W [outDim, inDim] as the checkpoint stores it.
    auto linear = [&](const std::string& prefix, const std::vector<std::vector<float>>& x) {
        vai::Tensor wTen(ckpt[prefix + ".weight"], vai::Dtype::Fp32);
        vai::Tensor bTen(ckpt[prefix + ".bias"], vai::Dtype::Fp32);
        const float* w = static_cast<const float*>(wTen.hostData());
        const float* b = static_cast<const float*>(bTen.hostData());
        const uint32_t outDim = wTen.shape()[0];
        const uint32_t inDim = wTen.shape()[1];

        std::vector<std::vector<float>> out(x.size(), std::vector<float>(outDim));
        for (uint32_t j = 0; j < outDim; ++j)
        {
            const float* row = w + size_t(j) * inDim;
            for (size_t s = 0; s < x.size(); ++s)
            {
                float acc = 0.0f;
                for (uint32_t i = 0; i < inDim; ++i)
                    acc += row[i] * x[s][i];
                out[s][j] = acc + b[j];
            }
        }
        return out;
    };
    auto silu = [](std::vector<std::vector<float>>& x) {
        for (auto& v : x)
            for (float& e : v)
                e = e / (1.0f + std::exp(-e));
    };

    std::vector<std::vector<float>> cond(S);
    for (uint32_t s = 0; s < S; ++s)
        cond[s] = vai::node::timeEmbedding(1.0 - double(s) / S, D, modelCfg.timeMinPeriod, modelCfg.timeMaxPeriod);

    cond = linear("time_mlp_in", cond);
    silu(cond);
    cond = linear("time_mlp_out", cond);
    silu(cond);

    // Rows in the order ActionTransformer cuts them: per layer scale1 shift1 g1 scale2 shift2 g2,
    // then the final norm's scale shift. Its gate has no residual to scale.
    std::vector<float> table(size_t(S) * R * D);
    uint32_t r = 0;
    auto putNorm = [&](const std::string& prefix, uint32_t rows) {
        const auto mod = linear(prefix + ".dense", cond);
        for (uint32_t s = 0; s < S; ++s)
            for (uint32_t k = 0; k < rows; ++k)
                for (uint32_t j = 0; j < D; ++j)
                    table[(size_t(s) * R + r + k) * D + j] = mod[s][size_t(k) * D + j] + (k == 0 ? 1.0f : 0.0f);
        r += rows;
    };

    const std::string root = "paligemma_with_expert.gemma_expert.model.";
    for (uint32_t i = 0; i < L; ++i)
    {
        const std::string prefix = root + "layers." + std::to_string(i) + ".";
        putNorm(prefix + "input_layernorm", 3);
        putNorm(prefix + "post_attention_layernorm", 3);
    }
    putNorm(root + "norm", 2);
    VAI_ASSERT(r == R);

    return table;
}


/////////////////////////////////////////////////////////////////////////////////////////
// Weight loading - maps HuggingFace safetensors key names to model members
/////////////////////////////////////////////////////////////////////////////////////////
inline void loadWeights(Net& net, const vai::Checkpoint& ckpt)
{
    // Weights load at the net's wDefaultDtype, norm coefficients and the condition table at Fp32.
    const vai::Dtype wDt = net.config().wDefaultDtype;
    const vai::Dtype normDt = vai::Dtype::Fp32;

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

    // The PaliGemma half of the policy. Its lm_head sits one level above this
    // root, outside the wrapped model.
    const std::string vlm = "paligemma_with_expert.paligemma.model.";

    // Vision tower (SigLIP)
    {
        const auto& modelCfg = net.modelCfg;
        const std::string root = vlm + "vision_tower.vision_model.";

        // The patch projection is stored as a [D, C, P, P] kernel, whose trailing axes
        // are the (c, kh, kw) order a patch row carries — so it loads as a matrix.
        auto& patch = net.patchEmbed;
        patch.project["weight"] = vai::Tensor(ckpt[root + "embeddings.patch_embedding.weight"], wDt)
                                      .reshape({modelCfg.visionCfg.layerCfg.embedDim,
                                                modelCfg.visionCfg.imageChannels * modelCfg.visionCfg.patchSize *
                                                    modelCfg.visionCfg.patchSize});
        patch.project["bias"] = vai::sum(ckpt[root + "embeddings.position_embedding.weight"],
                                         ckpt[root + "embeddings.patch_embedding.bias"], wDt);

        for (uint32_t i = 0; i < modelCfg.visionCfg.numLayers; ++i)
        {
            std::string prefix = root + "encoder.layers." + std::to_string(i) + ".";
            auto& layer = net.visionEncoder.layers[i];

            loadLayerNorm(layer.norm1, prefix + "layer_norm1");
            loadLayerNorm(layer.norm2, prefix + "layer_norm2");

            loadLin(layer.mhAttn.projQ, prefix + "self_attn.q_proj");
            loadLin(layer.mhAttn.projK, prefix + "self_attn.k_proj");
            loadLin(layer.mhAttn.projV, prefix + "self_attn.v_proj");
            loadLin(layer.mhAttn.projOut, prefix + "self_attn.out_proj");

            loadLin(layer.ffn.fc1, prefix + "mlp.fc1");
            loadLin(layer.ffn.fc2, prefix + "mlp.fc2");
        }

        loadLayerNorm(net.visionEncoder.finalNorm, root + "post_layernorm");
    }

    // Multimodal projector
    loadLin(net.projector, vlm + "multi_modal_projector.linear");

    // Token embedding. Gemma ties the table to the LM head, and the export
    // wrote only the head — so that tensor is the table, read from the tied end.
    net.promptEmbed["weight"] = vai::Tensor(ckpt["paligemma_with_expert.paligemma.lm_head.weight"], wDt);

    // Both Gemma stacks share the attention and FFN key names. Only the
    // language stack has norm weights; the expert's norms are the condition.
    auto loadGemmaLayer = [&](auto& layer, const std::string& prefix) {
        // Self attention projections
        loadLinNoBias(layer.mhAttn.projQ, prefix + "self_attn.q_proj");
        loadLinNoBias(layer.mhAttn.projK, prefix + "self_attn.k_proj");
        loadLinNoBias(layer.mhAttn.projV, prefix + "self_attn.v_proj");
        loadLinNoBias(layer.mhAttn.projOut, prefix + "self_attn.o_proj");

        // FFN projections (GeGLU). gate and value are one projection here, so
        // their rows are stacked — gate first, the half GatedMul activates.
        layer.ffn.gateValue["weight"] =
            vai::concat({ckpt[prefix + "mlp.gate_proj.weight"], ckpt[prefix + "mlp.up_proj.weight"]}, wDt);
        loadLinNoBias(layer.ffn.out, prefix + "mlp.down_proj");
    };

    for (uint32_t i = 0; i < net.modelCfg.languageCfg.numLayers; ++i)
    {
        std::string prefix = vlm + "language_model.layers." + std::to_string(i) + ".";
        auto& layer = net.languageEncoder.layers[i];

        loadRMSNorm(layer.norm1, prefix + "input_layernorm.weight");
        loadRMSNorm(layer.norm2, prefix + "post_attention_layernorm.weight");
        loadGemmaLayer(layer, prefix);
    }
    loadRMSNorm(net.languageEncoder.finalNorm, vlm + "language_model.norm.weight");

    for (uint32_t i = 0; i < net.modelCfg.actionCfg.numLayers; ++i)
    {
        std::string prefix = "paligemma_with_expert.gemma_expert.model.layers." + std::to_string(i) + ".";
        loadGemmaLayer(net.actionDecoder.layers[i], prefix);
    }

    // Flow-matching head — the projections that open and close the suffix, and
    // the condition table that stands in for the time MLP and the expert's norms.
    loadLin(net.actionProj, "action_in_proj");
    loadLin(net.velocityHead, "action_out_proj");

    const auto& modelCfg = net.modelCfg;
    net.condTable["table"] =
        vai::Tensor({net.runtimeCfg.denoiseSteps, modelCfg.condRows(), modelCfg.actionCfg.layerCfg.embedDim},
                    vai::Dtype::Fp32)
            .setFloats(condTable(net, ckpt));
}

}   // namespace model::production::pi05::lerobot


// loadWeights<"lerobot">(net) — composes the path in Checkpoint and loads it.
template <>
inline void loadWeights<"lerobot", model::production::pi05::Net>(model::production::pi05::Net& net)
{
    std::string dir = std::string(VAI_DATA_DIR "/ckpts/pi05/lerobot/") + net.modelCfg.variant;
    model::production::pi05::lerobot::loadWeights(net, vai::Checkpoint(dir));
}

#endif   // PRODUCTION_PI05_LEROBOT_WEIGHTS_H
