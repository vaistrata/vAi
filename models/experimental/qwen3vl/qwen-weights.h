/*=============================================================================
  Written by   oneway@vaistrata.com
  Checkpoint   Each size reads every *.safetensors in data/ckpts/qwen3vl/qwen/<ModelConfig::variant>/,
               taken from the matching Instruct repo on HuggingFace:
                 2B  Qwen/Qwen3-VL-2B-Instruct
                 4B  Qwen/Qwen3-VL-4B-Instruct
                 8B  Qwen/Qwen3-VL-8B-Instruct
=============================================================================*/

#ifndef EXPERIMENTAL_QWEN3VL_QWEN_WEIGHTS_H
#define EXPERIMENTAL_QWEN3VL_QWEN_WEIGHTS_H

#include "net.h"
#include "vai/engine/weights.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>


namespace model::experimental::qwen3vl::qwen {

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
    // The decoder's projections carry no bias: the "bias" slot stays empty.
    auto loadLinNoBias = [&](auto& lin, const std::string& prefix) {
        lin["weight"] = vai::Tensor(ckpt[prefix + ".weight"], wDt);
    };
    auto loadLayerNorm = [&](auto& ln, const std::string& prefix) {
        ln["scale"] = vai::Tensor(ckpt[prefix + ".weight"], normDt);
        ln["shift"] = vai::Tensor(ckpt[prefix + ".bias"], normDt);
    };
    auto loadRMSNorm = [&](auto& rn, const std::string& wKey) { rn["scale"] = vai::Tensor(ckpt[wKey], normDt); };

    // Vision tower
    {
        const VisionConfig& cfg = net.modelCfg.vision;
        auto& vision = net.visionEncoder;
        const std::string root = "model.visual.";

        const uint32_t D = cfg.trf.embedDim;
        const uint32_t C = cfg.trf.imageChannels;
        const uint32_t PP = cfg.trf.patchSize * cfg.trf.patchSize;
        const uint32_t G = vision.patchesPerSide;

        // The patch projection is a Conv3d over two frames, [D, C, 2, P, P]. An image
        // is its own frame twice, so the two temporal taps sum to one [D, C * P * P]
        // kernel, whose rows carry the (c, kh, kw) order a patch row does.
        {
            const vai::Tensor proj(ckpt[root + "patch_embed.proj.weight"], vai::Dtype::Fp32);
            const float* w = static_cast<const float*>(proj.hostData());

            std::vector<float> kernel(size_t(D) * C * PP);
            for (uint32_t o = 0; o < D; ++o)
                for (uint32_t c = 0; c < C; ++c)
                    for (uint32_t k = 0; k < PP; ++k)
                    {
                        const size_t frame0 = ((size_t(o) * C + c) * 2) * PP + k;
                        kernel[(size_t(o) * C + c) * PP + k] = w[frame0] + w[frame0 + PP];
                    }
            vision.patchEmbed.project["weight"] = vai::Tensor({ D, C * PP }, wDt).setFloats(std::move(kernel));
        }

        // The learned position table is a side x side grid, resampled bilinearly
        // (corners aligned) to G x G and added to the projection's bias, so each
        // raster patch carries its own [D] bias.
        {
            const vai::Tensor table(ckpt[root + "pos_embed.weight"], vai::Dtype::Fp32);
            const vai::Tensor bias(ckpt[root + "patch_embed.proj.bias"], vai::Dtype::Fp32);
            const float* t = static_cast<const float*>(table.hostData());
            const float* b = static_cast<const float*>(bias.hostData());
            const uint32_t side = uint32_t(std::lround(std::sqrt(double(table.shape()[0]))));

            struct Taps
            {
                uint32_t lo, hi;
                float frac;
            };
            auto taps = [&](uint32_t i) {
                const float src = G > 1 ? float(i) * float(side - 1) / float(G - 1) : 0.0f;
                const uint32_t lo = uint32_t(src);
                return Taps{ lo, std::min(lo + 1, side - 1), src - float(lo) };
            };

            std::vector<float> patchBias(size_t(G) * G * D);
            for (uint32_t r = 0; r < G; ++r)
                for (uint32_t c = 0; c < G; ++c)
                {
                    const Taps tr = taps(r), tc = taps(c);
                    const float w00 = (1 - tr.frac) * (1 - tc.frac), w01 = (1 - tr.frac) * tc.frac;
                    const float w10 = tr.frac * (1 - tc.frac), w11 = tr.frac * tc.frac;
                    const float* t00 = t + (size_t(tr.lo) * side + tc.lo) * D;
                    const float* t01 = t + (size_t(tr.lo) * side + tc.hi) * D;
                    const float* t10 = t + (size_t(tr.hi) * side + tc.lo) * D;
                    const float* t11 = t + (size_t(tr.hi) * side + tc.hi) * D;

                    float* out = patchBias.data() + (size_t(r) * G + c) * D;
                    for (uint32_t d = 0; d < D; ++d)
                        out[d] = b[d] + w00 * t00[d] + w01 * t01[d] + w10 * t10[d] + w11 * t11[d];
                }
            vision.patchEmbed.project["bias"] = vai::Tensor({ G * G, D }, wDt).setFloats(std::move(patchBias));
        }

        for (uint32_t i = 0; i < cfg.trf.nBlocks; ++i)
        {
            const std::string prefix = root + "blocks." + std::to_string(i) + ".";
            auto& block = vision.encoder.blocks[i];

            loadLayerNorm(block.norm1, prefix + "norm1");
            loadLayerNorm(block.norm2, prefix + "norm2");

            loadLin(block.mhAttn.projQKV, prefix + "attn.qkv");
            loadLin(block.mhAttn.projOut, prefix + "attn.proj");

            loadLin(block.ffn.fc1, prefix + "mlp.linear_fc1");
            loadLin(block.ffn.fc2, prefix + "mlp.linear_fc2");
        }

        // The checkpoint keeps the ViT's closing norm in its merger (our connector): each
        // patch normalized before the regroup. A DeepStack connector normalizes the regrouped row.
        loadLayerNorm(vision.encoder.finalNorm, root + "merger.norm");
        loadLin(vision.connector.projector.fc1, root + "merger.linear_fc1");
        loadLin(vision.connector.projector.fc2, root + "merger.linear_fc2");

        for (size_t j = 0; j < vision.deepConnectors.size(); ++j)
        {
            const std::string prefix = root + "deepstack_merger_list." + std::to_string(j) + ".";
            auto& connector = vision.deepConnectors[j];

            loadLayerNorm(*connector.tokenNorm, prefix + "norm");
            loadLin(connector.projector.fc1, prefix + "linear_fc1");
            loadLin(connector.projector.fc2, prefix + "linear_fc2");
        }
    }

    // Token embedding
    const std::string root = "model.language_model.";
    net.embedding["weight"] = vai::Tensor(ckpt[root + "embed_tokens.weight"], wDt);

    // Transformer blocks
    for (uint32_t i = 0; i < net.modelCfg.trf.nBlocks; ++i)
    {
        const std::string prefix = root + "layers." + std::to_string(i) + ".";
        auto& block = net.decoder.blocks[i];

        // RMSNorms (params loaded at Fp32)
        loadRMSNorm(block.norm1, prefix + "input_layernorm.weight");
        loadRMSNorm(block.norm2, prefix + "post_attention_layernorm.weight");

        // Self attention projections
        loadLinNoBias(block.mhAttn.projQ, prefix + "self_attn.q_proj");
        loadLinNoBias(block.mhAttn.projK, prefix + "self_attn.k_proj");
        loadLinNoBias(block.mhAttn.projV, prefix + "self_attn.v_proj");
        loadLinNoBias(block.mhAttn.projOut, prefix + "self_attn.o_proj");

        // Per-head Q/K norms
        loadRMSNorm(*block.mhAttn.normQ, prefix + "self_attn.q_norm.weight");
        loadRMSNorm(*block.mhAttn.normK, prefix + "self_attn.k_norm.weight");

        // FFN projections (SwiGLU). gate and value are one projection here, so
        // their rows are stacked — gate first, the half GatedMul activates.
        {
            block.ffn.gateValue["weight"] =
                vai::concat({ ckpt[prefix + "mlp.gate_proj.weight"], ckpt[prefix + "mlp.up_proj.weight"] }, wDt);
        }
        loadLinNoBias(block.ffn.out, prefix + "mlp.down_proj");
    }

    // Final RMSNorm
    loadRMSNorm(net.decoder.finalNorm, root + "norm.weight");

    // LM Head. 2B and 4B tie it to the token embedding; 8B ships its own.
    if (net.modelCfg.tiedHead)
        net.lmHead["weight"] = net.embedding["weight"];
    else
        net.lmHead["weight"] = vai::Tensor(ckpt["lm_head.weight"], wDt);
}

}   // namespace model::experimental::qwen3vl::qwen


// loadWeights<"qwen">(net) — composes the path in Checkpoint and loads it.
template <>
inline void loadWeights<"qwen", model::experimental::qwen3vl::Net>(model::experimental::qwen3vl::Net& net)
{
    std::string dir = std::string(VAI_DATA_DIR "/ckpts/qwen3vl/qwen/") + net.modelCfg.variant;
    model::experimental::qwen3vl::qwen::loadWeights(net, vai::Checkpoint(dir));
}

#endif   // EXPERIMENTAL_QWEN3VL_QWEN_WEIGHTS_H
