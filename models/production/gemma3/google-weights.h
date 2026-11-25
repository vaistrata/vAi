/*=============================================================================
  Written by   qkrtmddn0507@vaistrata.com

  Checkpoint   Each size reads every *.safetensors in data/ckpts/gemma3/google/<ModelConfig::variant>/,
               taken from the matching Instruct repo on HuggingFace:

               Text-only models:
                 270M  google/gemma-3-270m-it
                    or unsloth/gemma-3-270m-it
                 1B    google/gemma-3-1b-it
                    or unsloth/gemma-3-1b-it

               Vision-language models:
                 4B    google/gemma-3-4b-it
                 12B   google/gemma-3-12b-it
=============================================================================*/

#ifndef PRODUCTION_GEMMA3_GOOGLE_WEIGHTS_H
#define PRODUCTION_GEMMA3_GOOGLE_WEIGHTS_H

#include "net.h"
#include "vai/engine/weights.h"
#include <string>
#include <vector>


namespace model::production::gemma3::google {

/////////////////////////////////////////////////////////////////////////////////////////
// Weight loading - maps HuggingFace safetensors key names to model members
/////////////////////////////////////////////////////////////////////////////////////////
inline void loadWeights(Net& net, const vai::Checkpoint& ckpt)
{
    // Weights load at the net's wDefaultDtype, norm coefficients at Fp32.
    const vai::Dtype wDt = net.config().wDefaultDtype;
    const vai::Dtype normDt = vai::Dtype::Fp32;

    const bool hasVision = net.modelCfg.hasVision();

    // Text-only checkpoints use:
    //   model.*
    //
    // Gemma 3 VLM checkpoints use:
    //   language_model.model.*
    const std::string textRoot = hasVision ? "language_model.model." : "model.";

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

    // Vision encoder and multimodal projector, only in the VLM checkpoints
    if (hasVision)
    {
        const VisionTransformerConfig& cfg = VisionConfig::trf;
        auto& vision = net.visionEncoder.value();

        const std::string visionRoot = "vision_tower.vision_model.";

        // Patch embedding
        {
            auto& patch = vision.patchEmbed;

            patch.project["weight"] =
                vai::Tensor(ckpt[visionRoot + "embeddings.patch_embedding.weight"], wDt)
                    .reshape({ cfg.embedDim, cfg.imageChannels * cfg.patchSize * cfg.patchSize });

            patch.project["bias"] = vai::sum(ckpt[visionRoot + "embeddings.position_embedding.weight"],
                                             ckpt[visionRoot + "embeddings.patch_embedding.bias"],
                                             wDt);
        }

        // Vision encoder blocks
        for (uint32_t i = 0; i < cfg.nBlocks; ++i)
        {
            const std::string prefix = visionRoot + "encoder.layers." + std::to_string(i) + ".";

            auto& block = vision.encoder.blocks[i];

            loadLayerNorm(block.norm1, prefix + "layer_norm1");
            loadLayerNorm(block.norm2, prefix + "layer_norm2");
            loadLin(block.mhAttn.projQ, prefix + "self_attn.q_proj");
            loadLin(block.mhAttn.projK, prefix + "self_attn.k_proj");
            loadLin(block.mhAttn.projV, prefix + "self_attn.v_proj");
            loadLin(block.mhAttn.projOut, prefix + "self_attn.out_proj");
            loadLin(block.ffn.fc1, prefix + "mlp.fc1");
            loadLin(block.ffn.fc2, prefix + "mlp.fc2");
        }

        // Final vision LayerNorm
        loadLayerNorm(vision.encoder.finalNorm, visionRoot + "post_layernorm");

        // Multimodal projector input norm
        loadRMSNorm(vision.connector.tokenNorm, "multi_modal_projector.mm_soft_emb_norm.weight");
        // Multimodal projector
        {
            vision.connector.projector["weight"] =
                vai::Tensor(ckpt["multi_modal_projector.mm_input_projection_weight"], wDt).permute(1, 0);
        }
    }

    // Token embedding
    {
        net.embedding["weight"] = vai::Tensor(ckpt[textRoot + "embed_tokens.weight"], wDt);
    }

    // Decoder blocks
    for (uint32_t i = 0; i < net.modelCfg.trf.nBlocks; ++i)
    {
        const std::string prefix = textRoot + "layers." + std::to_string(i) + ".";
        auto& block = net.decoder.blocks[i];

        // RMSNorms
        loadRMSNorm(block.norm1, prefix + "input_layernorm.weight");
        loadRMSNorm(block.norm2, prefix + "post_attention_layernorm.weight");
        loadRMSNorm(block.norm3, prefix + "pre_feedforward_layernorm.weight");
        loadRMSNorm(block.norm4, prefix + "post_feedforward_layernorm.weight");

        // Self attention projections
        loadLinNoBias(block.mhAttn.projQ, prefix + "self_attn.q_proj");
        loadLinNoBias(block.mhAttn.projK, prefix + "self_attn.k_proj");
        loadLinNoBias(block.mhAttn.projV, prefix + "self_attn.v_proj");
        loadLinNoBias(block.mhAttn.projOut, prefix + "self_attn.o_proj");

        // Per-head Q/K norms
        loadRMSNorm(*block.mhAttn.normQ, prefix + "self_attn.q_norm.weight");
        loadRMSNorm(*block.mhAttn.normK, prefix + "self_attn.k_norm.weight");

        // FFN projections (GeGLU).
        //
        // gate and value are represented as one projection internally, so the
        // checkpoint tensors are stacked row-wise: gate first, then up/value.
        {
            block.ffn.gateValue["weight"] =
                vai::concat({ ckpt[prefix + "mlp.gate_proj.weight"], ckpt[prefix + "mlp.up_proj.weight"] }, wDt);
        }
        loadLinNoBias(block.ffn.out, prefix + "mlp.down_proj");
    }

    // Final RMSNorm
    loadRMSNorm(net.decoder.finalNorm, textRoot + "norm.weight");

    // LM Head.
    //
    // Gemma 3 ties it to the token embedding, so the checkpoint carries no
    // separate lm_head tensor.
    net.lmHead["weight"] = net.embedding["weight"];
}

}   // namespace model::production::gemma3::google


// loadWeights<"google">(net)
//
// Text-only models:
//   data/ckpts/gemma3/google/<variant>/model.safetensors
//
// VLM models:
//   all *.safetensors under data/ckpts/gemma3/google/<variant>/
template <>
inline void loadWeights<"google", model::production::gemma3::Net>(model::production::gemma3::Net& net)
{
    const std::string dir = std::string(VAI_DATA_DIR "/ckpts/gemma3/google/") + net.modelCfg.variant;

    model::production::gemma3::google::loadWeights(net, vai::Checkpoint(dir));
}

#endif   // PRODUCTION_GEMMA3_GOOGLE_WEIGHTS_H
