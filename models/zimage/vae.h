/*=============================================================================
  Written by   ippx123@gmail.com
  Model        Z-Image VAE decoder (diffusers AutoencoderKL, the Flux VAE:
               16 latent channels, blocks of 128 / 256 / 512 / 512)
  Modules      VaeResnet, VaeMidAttention, VaeDecoder
=============================================================================*/

#ifndef ZIMAGE_VAE_H
#define ZIMAGE_VAE_H

#include "vai/nodes/identity.h"
#include "vai/nodes/group-norm.h"
#include "models/zimage/nodes/zimage-vae-attention.h"
#include "models/zimage/nodes/zimage-vae-conv.h"
#include <array>
#include <cmath>
#include <vector>


namespace zimage {

/*
  Latent [16, h, w] -> RGB [3, 8h, 8w] in [-1, 1]. Every tensor is
  channel-major ([C, H, W]), the layout the convs read:

    conv_in -> mid (resnet, attention, resnet)
            -> up 0..3 (3 resnets each; 2x nearest upsample + conv after
               the first three) -> GroupNorm + SiLU -> conv_out

  The convs are VaeConvNode: each residual add is the conv before it, and
  each upsample the conv after it.

  The DiT's latent goes in as latent / scalingFactor + shiftFactor
  (decoderInput()).
*/

/////////////////////////////////////////////////////////////////////////////////////////
// VaeConfig - the decoder's shape (Tongyi-MAI/Z-Image-Turbo vae/config.json)
/////////////////////////////////////////////////////////////////////////////////////////
struct VaeConfig
{
    uint32_t latentChannels = 16;
    std::array<uint32_t, 4> blockOutChannels = {128, 256, 512, 512};
    uint32_t layersPerBlock = 2;            // the decoder runs one more resnet per block
    uint32_t numGroups = 32;
    float eps = 1e-6f;
    float scalingFactor = 0.3611f;
    float shiftFactor = 0.1159f;
};


/////////////////////////////////////////////////////////////////////////////////////////
// VaeResnet - diffusers ResnetBlock2D without a time embedding
//   out0 = shortcut(x) + conv2(silu(norm2(conv1(silu(norm1(x))))))
// shortcut is a 1x1 conv where the channel count changes, else identity.
/////////////////////////////////////////////////////////////////////////////////////////
struct VaeResnet : public vai::NodeGroup {
    bool hasShortcut;

    vai::node::Identity input;
    vai::node::GroupNorm norm1;
    VaeConvNode conv1;
    vai::node::GroupNorm norm2;
    VaeConvNode conv2;                     // + the shortcut, as its residual
    VaeConvNode shortcut;                  // only when hasShortcut

    VaeResnet(uint32_t inCh, uint32_t outCh, const VaeConfig& cfg)
    : hasShortcut(inCh != outCh)
    , norm1({.numGroups = cfg.numGroups, .eps = cfg.eps, .epilogue = vai::Epilogue::Silu})
    , conv1({.inChannels = inCh, .outChannels = outCh})
    , norm2({.numGroups = cfg.numGroups, .eps = cfg.eps, .epilogue = vai::Epilogue::Silu})
    , conv2({.inChannels = outCh, .outChannels = outCh, .hasResidual = true})
    , shortcut({.inChannels = inCh, .outChannels = outCh, .kernelSize = 1})
    {
        input - norm1 - conv1 - norm2 - conv2;
        if (hasShortcut)
            input - shortcut - "residual"/conv2;
        else
            input - "residual"/conv2;

        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", conv2.slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// VaeMidAttention - diffusers Attention, one head over all channels, residual
//   out0 = x + toOut(attention(qkv(groupNorm(x))))
// q, k and v run as one 1x1 conv (the loader stacks them), so the attention
// reads them channel-major (VaeAttentionNode).
/////////////////////////////////////////////////////////////////////////////////////////
struct VaeMidAttention : public vai::NodeGroup {
    vai::node::Identity input;
    vai::node::GroupNorm norm;
    VaeConvNode qkv;                       // (C, H, W) -> (3C, H, W)
    VaeAttentionNode attn;
    VaeConvNode toOut;                     // (C, H, W) -> (C, H, W), + x

    VaeMidAttention(uint32_t ch, const VaeConfig& cfg)
    : norm({.numGroups = cfg.numGroups, .eps = cfg.eps})
    , qkv({.inChannels = ch, .outChannels = 3 * ch, .kernelSize = 1})
    , toOut({.inChannels = ch, .outChannels = ch, .kernelSize = 1, .hasResidual = true})
    {
        input - norm - qkv - attn - toOut;
        input - "residual"/toOut;

        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", toOut.slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// VaeDecoder - latent -> image, one decode per forward
//
// Input    inLatent  [latentChannels, h, w]   decoderInput() of the DiT's latent
// Output   outImage  [3, 8h, 8w], RGB in about [-1, 1] (toRgb8() clamps)
/////////////////////////////////////////////////////////////////////////////////////////
struct VaeDecoder : public vai::NeuralNet
{
    VaeConfig vaeCfg;

    VaeConvNode convIn;
    VaeResnet mid0;
    VaeMidAttention midAttn;
    VaeResnet mid1;

    // up block b holds resnets[3b .. 3b+2]; blocks 0..2 then upsample (upConv
    // reads its input 2x nearest-upsampled).
    std::vector<VaeResnet> resnets;
    std::vector<VaeConvNode> upConv;

    vai::node::GroupNorm normOut;          // conv_norm_out + conv_act (SiLU)
    VaeConvNode convOut;

    static uint32_t topChannels(const VaeConfig& c) { return c.blockOutChannels.back(); }

    VaeDecoder(const vai::NeuralNetConfig& netCfg, const VaeConfig& cfg = {})
    : NeuralNet(netCfg)
    , vaeCfg(cfg)
    , convIn({.inChannels = cfg.latentChannels, .outChannels = topChannels(cfg)})
    , mid0(topChannels(cfg), topChannels(cfg), cfg)
    , midAttn(topChannels(cfg), cfg)
    , mid1(topChannels(cfg), topChannels(cfg), cfg)
    , normOut({.numGroups = cfg.numGroups, .eps = cfg.eps, .epilogue = vai::Epilogue::Silu})
    , convOut({.inChannels = cfg.blockOutChannels[0], .outChannels = 3})
    {
        const uint32_t numBlocks = uint32_t(cfg.blockOutChannels.size());
        const uint32_t perBlock = cfg.layersPerBlock + 1;
        resnets.reserve(numBlocks * perBlock);
        upConv.reserve(numBlocks - 1);

        // Up blocks run the channel list reversed: 512, 512, 256, 128.
        uint32_t ch = topChannels(cfg);
        for (uint32_t b = 0; b < numBlocks; ++b)
        {
            const uint32_t out = cfg.blockOutChannels[numBlocks - 1 - b];
            for (uint32_t r = 0; r < perBlock; ++r, ch = out)
                resnets.emplace_back(ch, out, cfg);
            if (b + 1 < numBlocks)
                upConv.emplace_back(VaeConvNode::Config{.inChannels = out, .outChannels = out,
                                                        .upsample = true});
        }

        input("inLatent") - convIn - mid0 - midAttn - mid1 - resnets[0];
        for (uint32_t b = 0; b < numBlocks; ++b)
        {
            for (uint32_t r = 1; r < perBlock; ++r)
                resnets[b * perBlock + r - 1] - resnets[b * perBlock + r];
            auto& last = resnets[b * perBlock + perBlock - 1];
            if (b + 1 < numBlocks)
                last - upConv[b] - resnets[(b + 1) * perBlock];
            else
                last - normOut - convOut - output("outImage");
        }
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// Host-side input and output
/////////////////////////////////////////////////////////////////////////////////////////

// The DiT's latent [C, h, w] as the decoder reads it: z / scaling + shift.
inline vai::Tensor decoderInput(const VaeConfig& cfg, const float* latent, uint32_t h, uint32_t w)
{
    std::vector<float> z(size_t(cfg.latentChannels) * h * w);
    for (size_t i = 0; i < z.size(); ++i)
        z[i] = latent[i] / cfg.scalingFactor + cfg.shiftFactor;
    return vai::Tensor({cfg.latentChannels, h, w}).setFloats(z);
}

// The decoder's [3, H, W] output as interleaved 8-bit RGB, rounded as
// diffusers' image processor does: (clamp(x, -1, 1) + 1) * 127.5.
inline std::vector<uint8_t> toRgb8(const float* image, uint32_t H, uint32_t W)
{
    std::vector<uint8_t> rgb(size_t(3) * H * W);
    const size_t plane = size_t(H) * W;
    for (size_t i = 0; i < plane; ++i)
        for (size_t c = 0; c < 3; ++c)
        {
            const float x = std::fmin(std::fmax(image[c * plane + i], -1.0f), 1.0f);
            rgb[i * 3 + c] = uint8_t(std::lround((x + 1.0f) * 127.5f));
        }
    return rgb;
}

}   // namespace zimage

#endif // ZIMAGE_VAE_H
