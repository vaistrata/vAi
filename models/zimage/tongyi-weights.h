/*=============================================================================
  Written by   ippx123@gmail.com
  Checkpoint   Reads every shard of data/zimage/<ModelConfig::name>/transformer/
               (diffusion_pytorch_model-*.safetensors), the diffusers layout of
               the matching HuggingFace repo:
                 Z-Image-Turbo  Tongyi-MAI/Z-Image-Turbo   (transformer/ only)
               The DiT alone; the caption encoder and VAE are not read.
=============================================================================*/

#ifndef ZIMAGE_TONGYI_WEIGHTS_H
#define ZIMAGE_TONGYI_WEIGHTS_H

#include "model.h"
#include "vai/engine/weights.h"
#include <cstring>
#include <string>
#include <vector>


namespace zimage::tongyi {

// A checkpoint tensor at `dtype`, every element multiplied by `scale`.
inline vai::Tensor scaledTensor(const vai::SafetensorRef& ref, vai::Dtype dtype, float scale)
{
    vai::Tensor f32(ref, vai::Dtype::Fp32);
    std::vector<float> v(f32.numElements());
    std::memcpy(v.data(), f32.hostData(), v.size() * sizeof(float));
    for (float& x : v) x *= scale;
    return vai::Tensor(f32.shape(), dtype).setFloats(v);
}

inline void loadWeights(Model& model, const vai::Checkpoint& ckpt)
{
    // Each tensor is loaded at the dtype declared by its owning node.
    auto loadLin = [&](vai::node::Linear& lin, const std::string& key)
    {
        lin["weight"] = vai::Tensor(ckpt[key + ".weight"], lin.config_.wDtype);
        if (lin.config_.hasBias)
            lin["bias"] = vai::Tensor(ckpt[key + ".bias"], lin.config_.wDtype);
    };
    auto loadNorm = [&](auto& norm, const std::string& key)
    {
        norm["gamma"] = vai::Tensor(ckpt[key], norm.config_.wDtype);
    };

    // The FFN runs on hidden * ffnHiddenScale (see FeedForward): w3 carries
    // the scale, w2 its inverse. w1 and w3 stack into one projection, w1
    // first (in fp32, since Q4 tensors cannot be concatenated), or stay two
    // when the model splits the up projection.
    const float s = model.modelCfg.ffnHiddenScale;
    auto loadBlock = [&](TransformerBlock& b, const std::string& prefix)
    {
        loadNorm(b.norm1,    prefix + "attention_norm1.weight");
        loadNorm(b.norm2,    prefix + "attention_norm2.weight");
        loadNorm(b.ffnNorm1, prefix + "ffn_norm1.weight");
        loadNorm(b.ffnNorm2, prefix + "ffn_norm2.weight");

        loadLin(b.attn.toQ,   prefix + "attention.to_q");
        loadLin(b.attn.toK,   prefix + "attention.to_k");
        loadLin(b.attn.toV,   prefix + "attention.to_v");
        loadLin(b.attn.toOut, prefix + "attention.to_out.0");
        loadNorm(b.attn.normRopeQ, prefix + "attention.norm_q.weight");
        loadNorm(b.attn.normRopeK, prefix + "attention.norm_k.weight");

        const vai::Dtype up = b.ffn.gateValue.config_.wDtype;
        if (model.runtimeCfg.splitFfnUp)
        {
            b.ffn.gateValue["weight"] = vai::Tensor(ckpt[prefix + "feed_forward.w1.weight"], up);
            b.ffn.value["weight"] = scaledTensor(ckpt[prefix + "feed_forward.w3.weight"], up, s);
        }
        else
        {
            const vai::Tensor w13 = vai::Tensor(ckpt[prefix + "feed_forward.w1.weight"], vai::Dtype::Fp32)
                .concat(scaledTensor(ckpt[prefix + "feed_forward.w3.weight"], vai::Dtype::Fp32, s));
            b.ffn.gateValue["weight"] = vai::Tensor(w13.shape(), up)
                .setFloats(static_cast<const float*>(w13.hostData()));
        }
        b.ffn.out["weight"] = scaledTensor(ckpt[prefix + "feed_forward.w2.weight"],
                                           b.ffn.out.config_.wDtype, 1.0f / s);

        if (b.modulated)
            loadLin(b.adaLN, prefix + "adaLN_modulation.0");
    };

    loadLin(model.tMlp1, "t_embedder.mlp.0");
    loadLin(model.tMlp2, "t_embedder.mlp.2");
    loadLin(model.xEmbed, "all_x_embedder.2-1");
    loadNorm(model.capNorm, "cap_embedder.0.weight");
    loadLin(model.capEmbed, "cap_embedder.1");
    model.xPad["padToken"]   = vai::Tensor(ckpt["x_pad_token"], model.xPad.config_.wDtype);
    model.capPad["padToken"] = vai::Tensor(ckpt["cap_pad_token"], model.capPad.config_.wDtype);

    for (size_t i = 0; i < model.noiseRefiner.size(); ++i)
        loadBlock(model.noiseRefiner[i], "noise_refiner." + std::to_string(i) + ".");
    for (size_t i = 0; i < model.contextRefiner.size(); ++i)
        loadBlock(model.contextRefiner[i], "context_refiner." + std::to_string(i) + ".");
    for (size_t i = 0; i < model.layers.size(); ++i)
        loadBlock(model.layers[i], "layers." + std::to_string(i) + ".");

    loadLin(model.finalAdaLN, "all_final_layer.2-1.adaLN_modulation.1");
    loadLin(model.finalLinear, "all_final_layer.2-1.linear");
}

} // namespace zimage::tongyi


// loadWeights<"tongyi">(model) — composes the path in Checkpoint and loads it.
template<>
inline void loadWeights<"tongyi", zimage::Model>(zimage::Model& model)
{
    std::string dir = std::string(VAI_DATA_DIR "/zimage/") + model.modelCfg.name + "/transformer";
    zimage::tongyi::loadWeights(model, vai::Checkpoint(dir));
}

#endif // ZIMAGE_TONGYI_WEIGHTS_H
