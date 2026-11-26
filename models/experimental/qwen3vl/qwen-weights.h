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

        // The learned position table stays a side x side grid; posEmbed resamples it
        // to each image's patch grid.
        vision.patchEmbed.project["bias"] = vai::Tensor(ckpt[root + "patch_embed.proj.bias"], wDt);
        vision.posEmbed["table"] = vai::Tensor(ckpt[root + "pos_embed.weight"], wDt);

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
