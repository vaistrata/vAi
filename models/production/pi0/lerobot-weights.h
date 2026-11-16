/*=============================================================================
  Written by   oneway@vaistrata.com
  Checkpoint   Each policy reads every *.safetensors file in
               data/ckpts/pi0/lerobot/<ModelConfig::variant>/, taken from the matching
               LeRobot port on HuggingFace:
                 3B  lerobot/pi0_base
=============================================================================*/

#ifndef PRODUCTION_PI0_LEROBOT_WEIGHTS_H
#define PRODUCTION_PI0_LEROBOT_WEIGHTS_H

#include "net.h"
#include "vai/engine/weights.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>


namespace model::production::pi0::lerobot {

/////////////////////////////////////////////////////////////////////////////////////////
// Timestep embedding - the flow-matching time t as sinusoids, sin half then cos half
//
// emb[i] = sin(2*pi * t / p_i),  emb[D/2 + i] = cos(2*pi * t / p_i),
// p_i = minPeriod * (maxPeriod / minPeriod)^(i / (D/2 - 1))
//
// The periods are the ones the checkpoint was trained with (openpi: 4e-3 and 4.0).
// The time MLP only fits those, so they are constants here, not settings.
/////////////////////////////////////////////////////////////////////////////////////////
inline std::vector<float> timeEmbedding(double t, uint32_t embedDim)
{
    constexpr double minPeriod = 4e-3;
    constexpr double maxPeriod = 4.0;

    const uint32_t half = embedDim / 2;
    std::vector<float> out(embedDim);
    for (uint32_t i = 0; i < half; ++i)
    {
        const double period = minPeriod * std::pow(maxPeriod / minPeriod, double(i) / (half - 1));
        const double angle = 2.0 * 3.14159265358979323846 * t / period;
        out[i] = float(std::sin(angle));
        out[half + i] = float(std::cos(angle));
    }
    return out;
}


/////////////////////////////////////////////////////////////////////////////////////////
// Weight loading - maps HuggingFace safetensors key names to model members
/////////////////////////////////////////////////////////////////////////////////////////
inline void loadWeights(Net& net, const vai::Checkpoint& ckpt)
{
    // Weights load at the net's wDefaultDtype, norm coefficients and the step table at Fp32.
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

    // The PaliGemma half of the policy: the VLM pi0 was initialized from. Its
    // lm_head sits one level above this root, outside the wrapped model.
    const std::string vlm = "paligemma_with_expert.paligemma.model.";

    // Vision tower (SigLIP)
    {
        const auto& modelCfg = net.modelCfg;
        const std::string root = vlm + "vision_tower.vision_model.";

        // The patch projection is stored as a [D, C, P, P] kernel, whose trailing axes
        // are the (c, kh, kw) order a patch row carries — so it loads as a matrix.
        auto& patch = net.patchEmbed;
        patch.project["weight"] =
            vai::Tensor(ckpt[root + "embeddings.patch_embedding.weight"], wDt)
                .reshape({
                    modelCfg.visionTrf.embedDim,
                    modelCfg.visionTrf.imageChannels * modelCfg.visionTrf.patchSize * modelCfg.visionTrf.patchSize,
                });
        patch.project["bias"] = vai::sum(
            ckpt[root + "embeddings.position_embedding.weight"], ckpt[root + "embeddings.patch_embedding.bias"], wDt);

        for (uint32_t i = 0; i < modelCfg.visionTrf.nBlocks; ++i)
        {
            std::string prefix = root + "encoder.layers." + std::to_string(i) + ".";
            auto& block = net.visionEncoder.blocks[i];

            loadLayerNorm(block.norm1, prefix + "layer_norm1");
            loadLayerNorm(block.norm2, prefix + "layer_norm2");

            loadLin(block.mhAttn.projQ, prefix + "self_attn.q_proj");
            loadLin(block.mhAttn.projK, prefix + "self_attn.k_proj");
            loadLin(block.mhAttn.projV, prefix + "self_attn.v_proj");
            loadLin(block.mhAttn.projOut, prefix + "self_attn.out_proj");

            loadLin(block.ffn.fc1, prefix + "mlp.fc1");
            loadLin(block.ffn.fc2, prefix + "mlp.fc2");
        }

        loadLayerNorm(net.visionEncoder.finalNorm, root + "post_layernorm");
    }

    // Multimodal projector
    loadLin(net.projector, vlm + "multi_modal_projector.linear");

    // Token embedding. Gemma ties the table to the LM head, and the export
    // wrote only the head — so that tensor is the table, read from the tied end.
    // pi0 never decodes text, so no head of our own reads it back.
    net.promptEmbed["weight"] = vai::Tensor(ckpt["paligemma_with_expert.paligemma.lm_head.weight"], wDt);

    // Both Gemma stacks, block by block. They read one stack of key names each and
    // are otherwise identical, so one loop loads the pair.
    auto loadGemma = [&](auto& gemma, const std::string& root) {
        for (uint32_t i = 0; i < net.modelCfg.languageTrf.nBlocks; ++i)
        {
            std::string prefix = root + "layers." + std::to_string(i) + ".";
            auto& block = gemma.blocks[i];

            // RMSNorms (params loaded at Fp32)
            loadRMSNorm(block.norm1, prefix + "input_layernorm.weight");
            loadRMSNorm(block.norm2, prefix + "post_attention_layernorm.weight");

            // Self attention projections
            loadLinNoBias(block.mhAttn.projQ, prefix + "self_attn.q_proj");
            loadLinNoBias(block.mhAttn.projK, prefix + "self_attn.k_proj");
            loadLinNoBias(block.mhAttn.projV, prefix + "self_attn.v_proj");
            loadLinNoBias(block.mhAttn.projOut, prefix + "self_attn.o_proj");

            // FFN projections (GeGLU). gate and value are one projection here, so
            // their rows are stacked — gate first, the half GatedMul activates.
            {
                block.ffn.gateValue["weight"] =
                    vai::concat({ ckpt[prefix + "mlp.gate_proj.weight"], ckpt[prefix + "mlp.up_proj.weight"] }, wDt);
            }
            loadLinNoBias(block.ffn.out, prefix + "mlp.down_proj");
        }

        loadRMSNorm(gemma.finalNorm, root + "norm.weight");
    };

    loadGemma(net.languageEncoder, vlm + "language_model.");
    loadGemma(net.actionDecoder, "paligemma_with_expert.gemma_expert.model.");

    // Flow-matching head — the projections that open and close the suffix.
    loadLin(net.stateProj, "state_proj");
    loadLin(net.actionProj, "action_in_proj");
    loadLin(net.actionTimeOut, "action_time_mlp_out");
    loadLin(net.velocityHead, "action_out_proj");

    // action_time_mlp_in [D, 2D] splits by its input: the action half stays a
    // projection, and the time half with the bias becomes one row per step,
    // W_t · emb(t_k) + b at t_k = 1 - k / denoiseSteps.
    {
        const uint32_t S = net.runtimeCfg.denoiseSteps;
        const uint32_t D = net.modelCfg.actionTrf.embedDim;

        vai::Tensor wTen(ckpt["action_time_mlp_in.weight"], vai::Dtype::Fp32);
        vai::Tensor bTen(ckpt["action_time_mlp_in.bias"], vai::Dtype::Fp32);
        VAI_ASSERT(wTen.isShapeOf(D, 2 * D));
        const float* w = static_cast<const float*>(wTen.hostData());
        const float* b = static_cast<const float*>(bTen.hostData());

        std::vector<float> wAction(size_t(D) * D);
        for (uint32_t j = 0; j < D; ++j)
            std::copy_n(w + size_t(j) * 2 * D, D, wAction.begin() + size_t(j) * D);
        net.actionTimeIn["weight"] = vai::Tensor({ D, D }, wDt).setFloats(std::move(wAction));

        std::vector<float> table(size_t(S) * D);
        for (uint32_t k = 0; k < S; ++k)
        {
            const std::vector<float> emb = timeEmbedding(1.0 - double(k) / S, D);
            for (uint32_t j = 0; j < D; ++j)
            {
                const float* wTime = w + size_t(j) * 2 * D + D;
                float acc = b[j];
                for (uint32_t i = 0; i < D; ++i)
                    acc += wTime[i] * emb[i];
                table[size_t(k) * D + j] = acc;
            }
        }
        net.stepTimeBias["table"] = vai::Tensor({ S, 1, D }, vai::Dtype::Fp32).setFloats(std::move(table));
    }
}

}   // namespace model::production::pi0::lerobot


// loadWeights<"lerobot">(net) — composes the path in Checkpoint and loads it.
template <>
inline void loadWeights<"lerobot", model::production::pi0::Net>(model::production::pi0::Net& net)
{
    std::string dir = std::string(VAI_DATA_DIR "/ckpts/pi0/lerobot/") + net.modelCfg.variant;
    model::production::pi0::lerobot::loadWeights(net, vai::Checkpoint(dir));
}

#endif   // PRODUCTION_PI0_LEROBOT_WEIGHTS_H
