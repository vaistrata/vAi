/*=============================================================================
  Written by   ippx123@gmail.com
  Checkpoint   Reads every shard of data/zimage/<ModelConfig::name>/transformer/
               (diffusion_pytorch_model-*.safetensors), the diffusers layout of
               the matching HuggingFace repo:
                 Z-Image-Turbo  Tongyi-MAI/Z-Image-Turbo   (transformer/ only)
               The DiT alone; the VAE decoder is tongyi-vae-weights.h, and
               the caption encoder is not read.
               loadWeightsPrepared() also writes the weights as the model holds
               them to data/zimage/<name>[-q4][-split].safetensors on its first
               load and reads that afterwards (see preparedPath()).
=============================================================================*/

#ifndef ZIMAGE_TONGYI_WEIGHTS_H
#define ZIMAGE_TONGYI_WEIGHTS_H

#include "model.h"
#include "vai/engine/weights.h"
#include <cstring>
#include <filesystem>
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

// Every weight slot of the model, once: fn(key, slot, dtype, make). key names
// the tensor as the model holds it, which is its checkpoint key except where
// the loader changes it (w13 for the stacked up projection, the FFN scale
// folded into w3 and w2); dtype is the slot's dtype; make(ckpt) builds it from
// the Tongyi checkpoint. loadWeights() calls make, and the prepared file
// (loadPrepared / savePrepared) keeps the result under key.
template<class Fn>
inline void forEachWeight(Model& model, Fn&& fn)
{
    using vai::Checkpoint;
    using vai::Tensor;

    // Each tensor is loaded at the dtype declared by its owning node.
    auto plain = [&](Tensor& slot, const std::string& key, vai::Dtype dtype)
    {
        fn(key, slot, dtype, [key, dtype](const Checkpoint& ckpt) { return Tensor(ckpt[key], dtype); });
    };
    auto loadLin = [&](vai::node::Linear& lin, const std::string& key)
    {
        plain(lin["weight"], key + ".weight", lin.config_.wDtype);
        if (lin.config_.hasBias)
            plain(lin["bias"], key + ".bias", lin.config_.wDtype);
    };
    auto loadNorm = [&](auto& norm, const std::string& key)
    {
        plain(norm["gamma"], key, norm.config_.wDtype);
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

        const std::string ff = prefix + "feed_forward.";
        const vai::Dtype up = b.ffn.gateValue.config_.wDtype;
        if (model.runtimeCfg.splitFfnUp)
        {
            plain(b.ffn.gateValue["weight"], ff + "w1.weight", up);
            fn(ff + "w3.weight", b.ffn.value["weight"], up, [=](const Checkpoint& ckpt) {
                return scaledTensor(ckpt[ff + "w3.weight"], up, s);
            });
        }
        else
            fn(ff + "w13.weight", b.ffn.gateValue["weight"], up, [=](const Checkpoint& ckpt) {
                const Tensor w13 = Tensor(ckpt[ff + "w1.weight"], vai::Dtype::Fp32)
                    .concat(scaledTensor(ckpt[ff + "w3.weight"], vai::Dtype::Fp32, s));
                return Tensor(w13.shape(), up).setFloats(static_cast<const float*>(w13.hostData()));
            });
        const vai::Dtype down = b.ffn.out.config_.wDtype;
        fn(ff + "w2.weight", b.ffn.out["weight"], down, [=](const Checkpoint& ckpt) {
            return scaledTensor(ckpt[ff + "w2.weight"], down, 1.0f / s);
        });

        if (b.modulated)
            loadLin(b.adaLN, prefix + "adaLN_modulation.0");
    };

    loadLin(model.tMlp1, "t_embedder.mlp.0");
    loadLin(model.tMlp2, "t_embedder.mlp.2");
    loadLin(model.xEmbed, "all_x_embedder.2-1");
    loadNorm(model.capNorm, "cap_embedder.0.weight");
    loadLin(model.capEmbed, "cap_embedder.1");
    plain(model.xPad["padToken"],   "x_pad_token",   model.xPad.config_.wDtype);
    plain(model.capPad["padToken"], "cap_pad_token", model.capPad.config_.wDtype);

    for (size_t i = 0; i < model.noiseRefiner.size(); ++i)
        loadBlock(model.noiseRefiner[i], "noise_refiner." + std::to_string(i) + ".");
    for (size_t i = 0; i < model.contextRefiner.size(); ++i)
        loadBlock(model.contextRefiner[i], "context_refiner." + std::to_string(i) + ".");
    for (size_t i = 0; i < model.layers.size(); ++i)
        loadBlock(model.layers[i], "layers." + std::to_string(i) + ".");

    loadLin(model.finalAdaLN, "all_final_layer.2-1.adaLN_modulation.1");
    loadLin(model.finalLinear, "all_final_layer.2-1.linear");
}

inline void loadWeights(Model& model, const vai::Checkpoint& ckpt)
{
    forEachWeight(model, [&](const std::string&, vai::Tensor& slot, vai::Dtype, auto make) {
        slot = make(ckpt);
    });
}


/////////////////////////////////////////////////////////////////////////////////////////
// Prepared weights - the model's weights as it holds them, in one file
//
// The checkpoint is fp32 (~23 GB for Turbo), and every load reads all of it
// and quantizes the block projections again. The prepared file is what that
// produces: Q4 stored packed, the FFN already stacked and scaled. Loading it
// is one read of ~3.7 GB (Q4) and no conversion. It depends on the runtime
// config that shapes the weights, which preparedPath() spells into its name;
// it is derived data, so deleting it only costs the next load the long way.
/////////////////////////////////////////////////////////////////////////////////////////
inline std::string preparedPath(const Model& model)
{
    const vai::Dtype bw = model.runtimeCfg.blockWeightDtype;
    return std::string(VAI_DATA_DIR "/zimage/") + model.modelCfg.name
         + (bw == vai::Dtype::Q4 ? "-q4" : bw == vai::Dtype::None ? "" : "-custom")
         + (model.runtimeCfg.splitFfnUp ? "-split" : "") + ".safetensors";
}

// Writes the weights loaded into `model` (before its first forward, which
// moves them to the GPU) to `path`.
inline void savePrepared(Model& model, const std::string& path)
{
    std::vector<std::pair<std::string, const vai::Tensor*>> tensors;
    forEachWeight(model, [&](const std::string& key, vai::Tensor& slot, vai::Dtype, auto) {
        tensors.emplace_back(key, &slot);
    });
    vai::saveSafetensors(path, tensors);
}

inline void loadPrepared(Model& model, const std::string& path)
{
    vai::SafetensorsParser file(path.c_str());
    forEachWeight(model, [&](const std::string& key, vai::Tensor& slot, vai::Dtype dtype, auto) {
        slot = vai::Tensor(file[key], dtype);
    });
}

} // namespace zimage::tongyi


// loadWeights<"tongyi">(model) — composes the path in Checkpoint and loads it.
template<>
inline void loadWeights<"tongyi", zimage::Model>(zimage::Model& model)
{
    std::string dir = std::string(VAI_DATA_DIR "/zimage/") + model.modelCfg.name + "/transformer";
    zimage::tongyi::loadWeights(model, vai::Checkpoint(dir));
}


namespace zimage::tongyi {

// Reads the prepared file when there is one and returns true. Otherwise loads
// the checkpoint (loadWeights<"tongyi">), writes the prepared file for the
// next load, and returns false. Once the prepared file exists, the checkpoint
// is not needed.
inline bool loadWeightsPrepared(Model& model)
{
    const std::string prepared = preparedPath(model);
    if (std::filesystem::exists(prepared))
    {
        loadPrepared(model, prepared);
        return true;
    }
    ::loadWeights<"tongyi">(model);
    savePrepared(model, prepared);
    return false;
}

} // namespace zimage::tongyi


#endif // ZIMAGE_TONGYI_WEIGHTS_H
