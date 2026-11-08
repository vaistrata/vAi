/*=============================================================================
  Written by   ippx123@gmail.com
  Checkpoint   data/zimage/Z-Image-Turbo/text_encoder/ (Tongyi-MAI/Z-Image-Turbo,
               Qwen3-4B in bf16, three shards, 8 GB) and tokenizer/tokenizer.json.
               loadTextEncoderPrepared() also writes the weights as the encoder
               holds them to data/zimage/Z-Image-Turbo-text-encoder[-q4].safetensors
               on its first load and reads that afterwards.
=============================================================================*/

#ifndef ZIMAGE_TONGYI_TEXT_ENCODER_WEIGHTS_H
#define ZIMAGE_TONGYI_TEXT_ENCODER_WEIGHTS_H

#include "text-encoder.h"
#include "vai/engine/weights.h"
#include <filesystem>
#include <memory>
#include <string>
#include <vector>


namespace zimage::tongyi {

inline std::string tokenizerPath()
{
    return VAI_DATA_DIR "/zimage/Z-Image-Turbo/tokenizer/tokenizer.json";
}

inline std::string textEncoderDir()
{
    return VAI_DATA_DIR "/zimage/Z-Image-Turbo/text_encoder";
}

// Every weight slot of the encoder, once: fn(key, slot, dtype, make), as
// forEachWeight() in tongyi-weights.h. gate_proj and up_proj stack into one
// projection, gate first.
template<class Fn>
inline void forEachWeight(TextEncoder& enc, Fn&& fn)
{
    using vai::Checkpoint;
    using vai::Tensor;

    auto plain = [&](Tensor& slot, const std::string& key, vai::Dtype dtype)
    {
        fn(key, slot, dtype, [key, dtype](const Checkpoint& ckpt) { return Tensor(ckpt[key], dtype); });
    };

    for (uint32_t i = 0; i < enc.layers.size(); ++i)
    {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        auto& layer = enc.layers[i];
        auto& attn = layer.mhAttn;
        plain(layer.norm1["scale"], p + "input_layernorm.weight", layer.norm1.config_.wDtype);
        plain(attn.toQ["weight"], p + "self_attn.q_proj.weight", attn.toQ.config_.wDtype);
        plain(attn.toK["weight"], p + "self_attn.k_proj.weight", attn.toK.config_.wDtype);
        plain(attn.toV["weight"], p + "self_attn.v_proj.weight", attn.toV.config_.wDtype);
        plain((*attn.normQ)["scale"], p + "self_attn.q_norm.weight", attn.normQ->config_.wDtype);
        plain((*attn.normK)["scale"], p + "self_attn.k_norm.weight", attn.normK->config_.wDtype);
        plain(attn.toOut["weight"], p + "self_attn.o_proj.weight", attn.toOut.config_.wDtype);
        plain(layer.norm2["scale"], p + "post_attention_layernorm.weight", layer.norm2.config_.wDtype);

        const vai::Dtype gv = layer.ffn.gateValue.config_.wDtype;
        fn(p + "mlp.gate_up_proj.weight", layer.ffn.gateValue["weight"], gv, [p, gv](const Checkpoint& ckpt) {
            return vai::concat({ckpt[p + "mlp.gate_proj.weight"], ckpt[p + "mlp.up_proj.weight"]}, gv);
        });
        plain(layer.ffn.out["weight"], p + "mlp.down_proj.weight", layer.ffn.out.config_.wDtype);
    }
}

constexpr const char* kEmbeddingKey = "model.embed_tokens.weight";

// From the bf16 checkpoint; the embedding rows are read from it as needed.
inline void loadTextEncoder(TextEncoder& enc, const std::string& dir = textEncoderDir())
{
    auto ckpt = std::make_shared<vai::Checkpoint>(dir);
    forEachWeight(enc, [&](const std::string&, vai::Tensor& slot, vai::Dtype, auto make) {
        slot = make(*ckpt);
    });
    enc.embedding = {ckpt, (*ckpt)[kEmbeddingKey]};
}


// The prepared file: the projections as the encoder holds them (Q4 packed,
// gate and up stacked) and the embedding table in fp16, ~2.8 GB for Q4.
inline std::string preparedTextEncoderPath(const TextEncoder& enc)
{
    const vai::Dtype d = enc.runtimeCfg.projDtype;
    return std::string(VAI_DATA_DIR "/zimage/Z-Image-Turbo-text-encoder")
         + (d == vai::Dtype::Q4 ? "-q4" : d == vai::Dtype::Fp16 ? "" : "-custom") + ".safetensors";
}

// Writes the weights loaded into `enc` (before its first forward) to `path`.
inline void savePreparedTextEncoder(TextEncoder& enc, const std::string& path)
{
    std::vector<std::pair<std::string, const vai::Tensor*>> tensors;
    forEachWeight(enc, [&](const std::string& key, vai::Tensor& slot, vai::Dtype, auto) {
        tensors.emplace_back(key, &slot);
    });
    const vai::Tensor table(enc.embedding.ref, vai::Dtype::Fp16);
    tensors.emplace_back(kEmbeddingKey, &table);
    vai::saveSafetensors(path, tensors);
}

inline void loadPreparedTextEncoder(TextEncoder& enc, const std::string& path)
{
    auto file = std::make_shared<vai::SafetensorsParser>(path.c_str());
    forEachWeight(enc, [&](const std::string& key, vai::Tensor& slot, vai::Dtype dtype, auto) {
        const vai::SafetensorRef ref = (*file)[key];
        if (ref.getDataType() == "BF16" || vai::dtypeFromString(ref.getDataType()) != dtype)
        {
            slot = vai::Tensor(ref, dtype);
            return;
        }
        slot = enc.paramTensor(ref.getShape(), dtype, [&](void* dst) { vai::readStored(ref, dst); });
    });
    enc.embedding = {file, (*file)[kEmbeddingKey]};
}

// Reads the prepared file when there is one and returns true; otherwise loads
// the checkpoint, writes the prepared file and returns false.
inline bool loadTextEncoderPrepared(TextEncoder& enc)
{
    const std::string prepared = preparedTextEncoderPath(enc);
    if (std::filesystem::exists(prepared))
    {
        loadPreparedTextEncoder(enc, prepared);
        return true;
    }
    loadTextEncoder(enc);
    savePreparedTextEncoder(enc, prepared);
    return false;
}

} // namespace zimage::tongyi

#endif // ZIMAGE_TONGYI_TEXT_ENCODER_WEIGHTS_H
