/*=============================================================================
  Written by   ippx123@gmail.com
  Checkpoint   Reads data/zimage/Z-Image-Turbo/vae/ (the diffusers layout of
               Tongyi-MAI/Z-Image-Turbo, vae/ only): the VAE decoder's weights,
               decoder.* (bf16, ~160 MB with the unused encoder).
=============================================================================*/

#ifndef ZIMAGE_TONGYI_VAE_WEIGHTS_H
#define ZIMAGE_TONGYI_VAE_WEIGHTS_H

#include "vae.h"
#include "vai/engine/weights.h"
#include <string>


namespace zimage::tongyi {

// q, k and v of the mid-block attention stack into its one qkv conv, q first.
inline void loadWeights(VaeDecoder& vae, const vai::Checkpoint& ckpt)
{
    using vai::Tensor;
    auto loadConv = [&](VaeConvNode& conv, const std::string& key)
    {
        conv["weight"] = VaeConvNode::packWeight(Tensor(ckpt[key + ".weight"], vai::Dtype::Fp32));
        conv["bias"] = VaeConvNode::packBias(Tensor(ckpt[key + ".bias"], vai::Dtype::Fp32));
    };
    auto loadNorm = [&](vai::node::GroupNorm& norm, const std::string& key)
    {
        norm["gamma"] = Tensor(ckpt[key + ".weight"], norm.config_.wDtype);
        norm["beta"] = Tensor(ckpt[key + ".bias"], norm.config_.wDtype);
    };
    auto loadResnet = [&](VaeResnet& r, const std::string& key)
    {
        loadNorm(r.norm1, key + ".norm1");
        loadConv(r.conv1, key + ".conv1");
        loadNorm(r.norm2, key + ".norm2");
        loadConv(r.conv2, key + ".conv2");
        if (r.hasShortcut)
            loadConv(r.shortcut, key + ".conv_shortcut");
    };

    const std::string d = "decoder.";
    loadConv(vae.convIn, d + "conv_in");
    loadResnet(vae.mid0, d + "mid_block.resnets.0");
    loadResnet(vae.mid1, d + "mid_block.resnets.1");

    const std::string a = d + "mid_block.attentions.0.";
    loadNorm(vae.midAttn.norm, a + "group_norm");
    // Linear [C, C] weights as 1x1 conv weights [C, C, 1, 1].
    const vai::Dtype f32 = vai::Dtype::Fp32;
    const uint32_t C = vae.midAttn.qkv.config_.inChannels;
    vae.midAttn.qkv["weight"] = VaeConvNode::packWeight(vai::concat({ckpt[a + "to_q.weight"],
        ckpt[a + "to_k.weight"], ckpt[a + "to_v.weight"]}, f32).reshape({3 * C, C, 1, 1}));
    vae.midAttn.qkv["bias"] = VaeConvNode::packBias(vai::concat({ckpt[a + "to_q.bias"],
        ckpt[a + "to_k.bias"], ckpt[a + "to_v.bias"]}, f32));
    vae.midAttn.toOut["weight"] = VaeConvNode::packWeight(
        Tensor(ckpt[a + "to_out.0.weight"], f32).reshape({C, C, 1, 1}));
    vae.midAttn.toOut["bias"] = VaeConvNode::packBias(Tensor(ckpt[a + "to_out.0.bias"], f32));

    const uint32_t perBlock = vae.vaeCfg.layersPerBlock + 1;
    for (size_t i = 0; i < vae.resnets.size(); ++i)
        loadResnet(vae.resnets[i], d + "up_blocks." + std::to_string(i / perBlock)
                                   + ".resnets." + std::to_string(i % perBlock));
    for (size_t b = 0; b < vae.upConv.size(); ++b)
        loadConv(vae.upConv[b], d + "up_blocks." + std::to_string(b) + ".upsamplers.0.conv");

    loadNorm(vae.normOut, d + "conv_norm_out");
    loadConv(vae.convOut, d + "conv_out");
}

} // namespace zimage::tongyi


// loadWeights<"tongyi">(vae) — the VAE decoder of Z-Image-Turbo.
template<>
inline void loadWeights<"tongyi", zimage::VaeDecoder>(zimage::VaeDecoder& vae)
{
    zimage::tongyi::loadWeights(vae, vai::Checkpoint(VAI_DATA_DIR "/zimage/Z-Image-Turbo/vae"));
}


#endif // ZIMAGE_TONGYI_VAE_WEIGHTS_H
