/*=============================================================================
  Written by   ippx123@gmail.com
  Model        Z-Image DiT (S3-DiT 6B; Z-Image-Turbo / Z-Image)
  Modules      Attention, FeedForward, TransformerBlock
=============================================================================*/

#ifndef ZIMAGE_MODEL_H
#define ZIMAGE_MODEL_H

#include "vai/nodes/identity.h"
#include "vai/nodes/linear.h"
#include "vai/nodes/concat.h"
#include "vai/nodes/gated-mul.h"
#include "vai/nodes/activation.h"
#include "models/zimage/nodes/zimage-ada-norm.h"
#include "models/zimage/nodes/zimage-attention.h"
#include "models/zimage/nodes/zimage-qk-norm-rope.h"
#include "models/zimage/nodes/zimage-pad-rows.h"
#include <array>
#include <cmath>
#include <vector>


namespace zimage {

// Every projection accumulates in fp32: Linear would otherwise follow the
// fp16 wDefaultDtype, and fp16 sums over K = 3840 / 10240 cost the DiT ~10x
// the error bf16 makes.
inline constexpr vai::Dtype acc = vai::Dtype::Fp32;

/*
  The DiT only — the Qwen3 caption encoder and the VAE stay outside. One
  forward is one denoising step over a single-stream sequence:

    image patches -> x_embedder -> pad -> noise_refiner x2 (modulated)
    caption feats -> cap_embedder -> pad -> context_refiner x2 (plain)
    [image ; caption] -> layers x30 (modulated) -> final layer

  Nx: image tokens, Nc: caption tokens, both padded to multiples of 32 (pad
  rows hold the learned pad tokens and are attended — batch 1 masks nothing).
*/

/////////////////////////////////////////////////////////////////////////////////////////
// Attention - bidirectional MHA with per-head QK RMS norm and 3-axis RoPE
//  in0 [N, D], cs [N, Dh/2, 2] -> out0 [N, D]
/////////////////////////////////////////////////////////////////////////////////////////
struct Attention : public vai::NodeGroup {
    uint32_t D;       // embedding dimension
    uint32_t H;       // number of heads

    vai::node::Identity input;
    vai::node::Identity cs;                       // (cos, sin) per token and pair
    vai::node::Linear toQ, toK, toV;              // (N, D) -> (N, D)
    QkNormRopeNode normRopeQ, normRopeK;        // (N, D) -> (H, N, D/H) normed + rotated
    FullAttentionNode attn;                     // Q, K head-major + V token-major -> (N, D)
    vai::node::Linear toOut;                      // (N, D) -> (N, D)

    Attention(uint32_t embedDim, uint32_t numHeads, float eps, bool allowCoopmat, vai::Dtype wDtype)
    : D(embedDim), H(numHeads)
    , toQ({.inDim = D, .outDim = D, .wDtype = wDtype, .accDtype = acc})
    , toK({.inDim = D, .outDim = D, .wDtype = wDtype, .accDtype = acc})
    , toV({.inDim = D, .outDim = D, .wDtype = wDtype, .accDtype = acc})
    , normRopeQ({.numHeads = H, .headDim = D/H, .eps = eps,
                 .wDtype = vai::Dtype::Fp32, .oDtype = vai::Dtype::Fp16})
    , normRopeK({.numHeads = H, .headDim = D/H, .eps = eps,
                 .wDtype = vai::Dtype::Fp32, .oDtype = vai::Dtype::Fp16})
    , attn({.numHeads = H, .headDim = D/H, .allowCoopmat = allowCoopmat})
    , toOut({.inDim = D, .outDim = D, .wDtype = wDtype, .accDtype = acc})
    {
        input - toQ - normRopeQ - "Q"/attn - toOut;
        input - toK - normRopeK - "K"/attn;
        input - toV - "V"/attn;
        cs - "cs"/normRopeQ;
        cs - "cs"/normRopeK;

        defineSlot("in0", input.slot("in0"));
        defineSlot("cs", cs.slot("in0"));
        defineSlot("out0", toOut.slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// FeedForward - SwiGLU: out0 = w2(silu(w1(in0)) * w3(in0))
//
// w1 and w3 read the same input, so they run as one projection over the
// concatenated rows — w1 first, the half GatedMul activates. The loader
// stacks them in that order.
//
// The hidden activation peaks past the fp16 range (~2e5 in Turbo's first main
// layer) and the down projection reads its input as fp16, so the loader runs
// the FFN on hidden * ffnHiddenScale: w3 scaled by s, w2 by 1/s. Both are
// powers of two, so the product is exact wherever fp16 stays normal.
//
// splitUp runs w1 and w3 as two Linears joined by a Concat instead: the same
// math, but the longest node is half as long. Mobile drivers reset the GPU
// when one submission runs past a couple of seconds (Adreno 830: ~1.7 s), and
// the fused projection is the DiT's longest node.
/////////////////////////////////////////////////////////////////////////////////////////
struct FeedForward : public vai::NodeGroup {
    vai::node::Identity input;
    vai::node::Linear gateValue;           // (N, D) -> (N, 2F); w1 alone (N, F) when split
    vai::node::Linear value;               // (N, D) -> (N, F), w3; only when split
    vai::node::Concat join;                // [w1 | w3] -> (N, 2F); only when split
    vai::node::GatedMul gatedMul;          // (N, 2F) -> (N, F)
    vai::node::Linear out;                 // (N, F) -> (N, D)

    FeedForward(uint32_t dim, uint32_t hiddenDim, vai::Dtype wDtype, bool splitUp)
    : gateValue({.inDim = dim, .outDim = (splitUp ? 1 : 2) * hiddenDim, .wDtype = wDtype, .accDtype = acc})
    , value({.inDim = dim, .outDim = hiddenDim, .wDtype = wDtype, .accDtype = acc})
    , join({.axis = -1})
    , gatedMul({vai::Epilogue::Silu})
    , out({.inDim = hiddenDim, .outDim = dim, .wDtype = wDtype, .accDtype = acc})
    {
        if (splitUp)
        {
            input - gateValue - "in0"/join - gatedMul;
            input - value     - "in1"/join;
        }
        else
            input - gateValue - gatedMul;
        gatedMul - out;

        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", out.slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// TransformerBlock - sandwich-norm block, adaLN-modulated or plain
//
//   modulated:  [scaleA, gateA, scaleF, gateF] = adaLN(temb)          (4 x D)
//     x = x + tanh(gateA) * norm2(attn(norm1(x) * (1 + scaleA)))
//     x = x + tanh(gateF) * ffnNorm2(ffn(ffnNorm1(x) * (1 + scaleF)))
//   plain (context refiner): the same with every scale / gate dropped.
//
// Each norm site, its modulation and its residual are one AdaNormNode.
// wDtype is the attention and FFN projections' weight dtype; the adaLN
// projection keeps the default.
/////////////////////////////////////////////////////////////////////////////////////////
struct TransformerBlock : public vai::NodeGroup {
    bool modulated;

    vai::node::Identity input;
    vai::node::Linear adaLN;               // (1, adaLNDim) -> (1, 4D); only when modulated
    AdaNormNode norm1;                   // attention_norm1 (* (1 + scaleA))
    Attention attn;
    AdaNormNode norm2;                   // x + tanh(gateA) * attention_norm2(attn)
    AdaNormNode ffnNorm1;                // ffn_norm1 (* (1 + scaleF))
    FeedForward ffn;
    AdaNormNode ffnNorm2;                // x + tanh(gateF) * ffn_norm2(ffn)

    TransformerBlock(uint32_t D, uint32_t numHeads, uint32_t ffnHiddenDim, uint32_t adaLNDim,
                     float eps, bool modulated, bool allowCoopmat, vai::Dtype wDtype, bool splitFfnUp)
    : modulated(modulated)
    , adaLN({.inDim = adaLNDim, .outDim = 4 * D, .hasBias = true, .accDtype = acc})
    , norm1({.eps = eps, .hasScale = modulated, .scaleOffset = 0, .wDtype = vai::Dtype::Fp32})
    , attn(D, numHeads, eps, allowCoopmat, wDtype)
    , norm2({.eps = eps, .hasGate = modulated, .hasResidual = true, .gateOffset = D,
             .wDtype = vai::Dtype::Fp32})
    , ffnNorm1({.eps = eps, .hasScale = modulated, .scaleOffset = 2 * D, .wDtype = vai::Dtype::Fp32})
    , ffn(D, ffnHiddenDim, wDtype, splitFfnUp)
    , ffnNorm2({.eps = eps, .hasGate = modulated, .hasResidual = true, .gateOffset = 3 * D,
                .wDtype = vai::Dtype::Fp32})
    {
        input - norm1 - "in0"/attn - norm2 - ffnNorm1 - ffn - ffnNorm2;
        input                      - "residual"/norm2;
                                     norm2      - "residual"/ffnNorm2;

        if (modulated)
        {
            adaLN - "mod"/norm1;
            adaLN - "mod"/norm2;
            adaLN - "mod"/ffnNorm1;
            adaLN - "mod"/ffnNorm2;
            defineSlot("temb", adaLN.slot("in0"));
        }

        defineSlot("in0", input.slot("in0"));
        defineSlot("cs", attn.slot("cs"));
        defineSlot("out0", ffnNorm2.slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// ModelConfig - Configuration for Z-Image checkpoints
/////////////////////////////////////////////////////////////////////////////////////////
struct ModelConfig
{
    const char* name;                               // checkpoint folder under data/zimage/
    uint32_t inChannels = 16;                       // latent channels
    uint32_t patchSize = 2;                         // spatial patch (frame patch is 1)
    uint32_t dim = 3840;                            // hidden size
    uint32_t numHeads = 30;
    uint32_t numLayers = 30;                        // main single-stream layers
    uint32_t numRefinerLayers = 2;                  // noise / context refiner layers each
    uint32_t ffnHiddenDim = 10240;                  // int(dim / 3 * 8)
    uint32_t capFeatDim = 2560;                     // caption encoder (Qwen3-4B) hidden size
    uint32_t adaLNDim = 256;                        // min(dim, 256)
    uint32_t tMidDim = 1024;                        // timestep MLP hidden size
    uint32_t tFreqDim = 256;                        // sinusoidal timestep features
    float tScale = 1000.0f;
    float normEps = 1e-5f;
    float ropeTheta = 256.0f;
    std::array<uint32_t, 3> axesDims = {32, 48, 48};   // RoPE dims per (frame, h, w) axis
    uint32_t seqMultiple = 32;                      // both sequences pad to this
    float ffnHiddenScale = 1.0f / 32;               // FFN hidden carried scaled (see FeedForward)

    uint32_t patchDim() const { return patchSize * patchSize * inChannels; }
    uint32_t headDim() const { return dim / numHeads; }

    static constexpr ModelConfig Turbo()
    {
        return { .name = "Z-Image-Turbo" };
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// RuntimeConfig - caller-chosen settings, independent of the checkpoint
/////////////////////////////////////////////////////////////////////////////////////////
struct RuntimeConfig {
    bool coopmatAttention = true;   // false: scalar attention kernel even where coopmat runs

    // Weight dtype of every block's attention and FFN projections, 97% of
    // the weights. None: Linear's default (fp16). Q4 brings the DiT from
    // ~12 GB to ~3.7 GB, so it fits a phone's memory.
    vai::Dtype blockWeightDtype = vai::Dtype::None;

    // Run each FFN's up projection as two Linears (see FeedForward), for
    // GPUs that reset after a short submission. Pair with nodesPerSubmit.
    bool splitFfnUp = false;
};


/////////////////////////////////////////////////////////////////////////////////////////
// Model - Z-Image DiT, one denoising step per forward
//
// Inputs   inPatches   [Nx0, patchDim]   patchify() of the latent
//          inCapFeats  [Nc0, capFeatDim] caption encoder hidden states
//          inTimestep  [1, tFreqDim]     timestepFeatures(t)
//          inImageCs   [Nx, Dh/2, 2]     ropeTable() of imagePositions()
//          inCapCs     [Nc, Dh/2, 2]     ropeTable() of captionPositions()
// Output   outPatches  [Nx + Nc, patchDim]; unpatchify() reads the first Nx0
//
// makeInputs() assembles all five from a latent, caption features and t.
/////////////////////////////////////////////////////////////////////////////////////////
struct Model : public vai::NeuralNet
{
    ModelConfig modelCfg;
    RuntimeConfig runtimeCfg;

    // Timestep embedder: sinusoid -> Linear + SiLU -> Linear = temb (adaLN input).
    vai::node::Linear tMlp1, tMlp2;
    vai::node::Identity temb;

    // Image stream
    vai::node::Linear xEmbed;              // (Nx0, patchDim) -> (Nx0, D)
    PadRowsNode xPad;                    // -> (Nx, D), x_pad_token rows
    vai::node::Identity imageCs;
    std::vector<TransformerBlock> noiseRefiner;

    // Caption stream
    AdaNormNode capNorm;                 // cap_embedder.0 RMS norm
    vai::node::Linear capEmbed;            // (Nc0, capFeatDim) -> (Nc0, D)
    PadRowsNode capPad;                  // -> (Nc, D), cap_pad_token rows
    vai::node::Identity capCs;
    std::vector<TransformerBlock> contextRefiner;

    // Single stream
    vai::node::Concat unify;               // [image ; caption] tokens
    vai::node::Concat unifyCs;             // their RoPE tables, same order
    std::vector<TransformerBlock> layers;

    // Final layer: LayerNorm (no affine) * (1 + Linear(SiLU(temb))) -> Linear
    vai::node::Activation finalSilu;
    vai::node::Linear finalAdaLN;          // (1, adaLNDim) -> (1, D)
    AdaNormNode finalNorm;
    vai::node::Linear finalLinear;         // (N, D) -> (N, patchDim)

    Model(const vai::NeuralNetConfig& netCfg, const ModelConfig& modelCfg,
          const RuntimeConfig& runtimeCfg = {})
    : NeuralNet(netCfg)
    , modelCfg(modelCfg)
    , runtimeCfg(runtimeCfg)
    , tMlp1({.inDim = modelCfg.tFreqDim, .outDim = modelCfg.tMidDim, .hasBias = true,
             .epilogue = vai::Epilogue::Silu, .accDtype = acc})
    , tMlp2({.inDim = modelCfg.tMidDim, .outDim = modelCfg.adaLNDim, .hasBias = true, .accDtype = acc})
    , xEmbed({.inDim = modelCfg.patchDim(), .outDim = modelCfg.dim, .hasBias = true, .accDtype = acc})
    , xPad({.multiple = modelCfg.seqMultiple, .wDtype = vai::Dtype::Fp32})
    , capNorm({.eps = modelCfg.normEps, .wDtype = vai::Dtype::Fp32})
    , capEmbed({.inDim = modelCfg.capFeatDim, .outDim = modelCfg.dim, .hasBias = true, .accDtype = acc})
    , capPad({.multiple = modelCfg.seqMultiple, .wDtype = vai::Dtype::Fp32})
    , unify({.axis = 0})
    , unifyCs({.axis = 0})
    , finalSilu({.epilogue = vai::Epilogue::Silu})
    , finalAdaLN({.inDim = modelCfg.adaLNDim, .outDim = modelCfg.dim, .hasBias = true, .accDtype = acc})
    , finalNorm({.eps = 1e-6f, .center = true, .hasGamma = false, .hasScale = true})
    , finalLinear({.inDim = modelCfg.dim, .outDim = modelCfg.patchDim(), .hasBias = true, .accDtype = acc})
    {
        const uint32_t D = modelCfg.dim, H = modelCfg.numHeads, F = modelCfg.ffnHiddenDim;
        const uint32_t A = modelCfg.adaLNDim;
        const float eps = modelCfg.normEps;
        const bool cm = runtimeCfg.coopmatAttention;
        const vai::Dtype bw = runtimeCfg.blockWeightDtype;
        const bool su = runtimeCfg.splitFfnUp;

        noiseRefiner.reserve(modelCfg.numRefinerLayers);
        contextRefiner.reserve(modelCfg.numRefinerLayers);
        for (uint32_t i = 0; i < modelCfg.numRefinerLayers; ++i)
        {
            noiseRefiner.emplace_back(D, H, F, A, eps, true, cm, bw, su);
            contextRefiner.emplace_back(D, H, F, A, eps, false, cm, bw, su);
        }
        layers.reserve(modelCfg.numLayers);
        for (uint32_t i = 0; i < modelCfg.numLayers; ++i)
            layers.emplace_back(D, H, F, A, eps, true, cm, bw, su);

        // Timestep -> temb, fanned out to every modulated block and the final layer.
        input("inTimestep") - tMlp1 - tMlp2 - temb;
        for (auto& b : noiseRefiner) temb - "temb"/b;
        for (auto& b : layers)       temb - "temb"/b;
        temb - finalSilu - finalAdaLN - "mod"/finalNorm;

        // Image stream
        input("inImageCs") - imageCs;
        for (auto& b : noiseRefiner) imageCs - "cs"/b;
        input("inPatches") - xEmbed - xPad - noiseRefiner[0];
        for (size_t i = 1; i < noiseRefiner.size(); ++i)
            noiseRefiner[i - 1] - noiseRefiner[i];

        // Caption stream
        input("inCapCs") - capCs;
        for (auto& b : contextRefiner) capCs - "cs"/b;
        input("inCapFeats") - capNorm - capEmbed - capPad - contextRefiner[0];
        for (size_t i = 1; i < contextRefiner.size(); ++i)
            contextRefiner[i - 1] - contextRefiner[i];

        // Single stream: image tokens first, then caption tokens.
        noiseRefiner.back()   - "in0"/unify;
        contextRefiner.back() - "in1"/unify;
        imageCs - "in0"/unifyCs;
        capCs   - "in1"/unifyCs;
        for (auto& b : layers) unifyCs - "cs"/b;

        unify - layers[0];
        for (size_t i = 1; i < layers.size(); ++i)
            layers[i - 1] - layers[i];
        layers.back() - finalNorm - finalLinear - output("outPatches");
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// Host-side input preparation
/////////////////////////////////////////////////////////////////////////////////////////

inline uint32_t padTo(uint32_t n, uint32_t multiple)
{
    return (n + multiple - 1) / multiple * multiple;
}

// Sinusoidal timestep features of t * tScale: [cos(args), sin(args)], args =
// t * tScale * exp(-ln(10000) * i / half). t is the pipeline's flow time in
// [0, 1] (1 - sigma).
inline vai::Tensor timestepFeatures(const ModelConfig& cfg, float t)
{
    const uint32_t half = cfg.tFreqDim / 2;
    std::vector<float> f(cfg.tFreqDim);
    for (uint32_t i = 0; i < half; ++i)
    {
        float freq = std::exp(-std::log(10000.0f) * float(i) / float(half));
        float arg  = t * cfg.tScale * freq;
        f[i]        = std::cos(arg);
        f[half + i] = std::sin(arg);
    }
    return vai::Tensor({1, cfg.tFreqDim}).setFloats(f);
}

// Latent [C, H, W] -> patches [(H/p)*(W/p), p*p*C], token-major over
// (h, w), features ordered (ph, pw, c). H and W must be multiples of p
// (image sides multiples of 16 px), as diffusers requires.
inline vai::Tensor patchify(const ModelConfig& cfg, const float* latent, uint32_t H, uint32_t W)
{
    const uint32_t C = cfg.inChannels, p = cfg.patchSize;
    VAI_ASSERT(H % p == 0 && W % p == 0);
    const uint32_t Ht = H / p, Wt = W / p, P = cfg.patchDim();
    std::vector<float> out(size_t(Ht) * Wt * P);
    for (uint32_t ht = 0; ht < Ht; ++ht)
    for (uint32_t wt = 0; wt < Wt; ++wt)
    for (uint32_t ph = 0; ph < p; ++ph)
    for (uint32_t pw = 0; pw < p; ++pw)
    for (uint32_t c = 0; c < C; ++c)
        out[(size_t(ht) * Wt + wt) * P + (ph * p + pw) * C + c] =
            latent[(size_t(c) * H + ht * p + ph) * W + wt * p + pw];
    return vai::Tensor({Ht * Wt, P}).setFloats(out);
}

// Inverse of patchify(): the first (H/p)*(W/p) rows of outPatches -> [C, H, W].
inline std::vector<float> unpatchify(const ModelConfig& cfg, const float* patches, uint32_t H, uint32_t W)
{
    const uint32_t C = cfg.inChannels, p = cfg.patchSize;
    const uint32_t Ht = H / p, Wt = W / p, P = cfg.patchDim();
    std::vector<float> out(size_t(C) * H * W);
    for (uint32_t ht = 0; ht < Ht; ++ht)
    for (uint32_t wt = 0; wt < Wt; ++wt)
    for (uint32_t ph = 0; ph < p; ++ph)
    for (uint32_t pw = 0; pw < p; ++pw)
    for (uint32_t c = 0; c < C; ++c)
        out[(size_t(c) * H + ht * p + ph) * W + wt * p + pw] =
            patches[(size_t(ht) * Wt + wt) * P + (ph * p + pw) * C + c];
    return out;
}

using PosId = std::array<uint32_t, 3>;   // (frame, h, w)

// Caption tokens sit on the frame axis at 1 .. Nc, pad rows included.
inline std::vector<PosId> captionPositions(uint32_t numPadded)
{
    std::vector<PosId> ids(numPadded);
    for (uint32_t i = 0; i < numPadded; ++i) ids[i] = {1 + i, 0, 0};
    return ids;
}

// Image tokens take frame Nc + 1 and their (h, w) grid cell; pad rows sit at
// the origin.
inline std::vector<PosId> imagePositions(uint32_t Ht, uint32_t Wt, uint32_t numPadded, uint32_t capPadded)
{
    std::vector<PosId> ids(numPadded, PosId{0, 0, 0});
    for (uint32_t h = 0; h < Ht; ++h)
        for (uint32_t w = 0; w < Wt; ++w)
            ids[h * Wt + w] = {capPadded + 1, h, w};
    return ids;
}

// (cos, sin) per token and rotation pair: [N, Dh/2, 2]. Pair k of axis a
// turns at theta^(-2k / axesDims[a]) per position step; the axes' pairs are
// laid end to end (frame, h, w), matching the checkpoint's head-dim order.
inline vai::Tensor ropeTable(const ModelConfig& cfg, const std::vector<PosId>& ids)
{
    const uint32_t pairs = cfg.headDim() / 2;
    std::vector<float> cs(ids.size() * pairs * 2);
    for (size_t n = 0; n < ids.size(); ++n)
    {
        uint32_t k = 0;
        for (uint32_t a = 0; a < 3; ++a)
        {
            const uint32_t d = cfg.axesDims[a];
            for (uint32_t i = 0; i < d / 2; ++i, ++k)
            {
                double freq  = 1.0 / std::pow(double(cfg.ropeTheta), double(2 * i) / double(d));
                float  angle = float(double(ids[n][a]) * freq);   // fp32 angle, as upstream
                cs[(n * pairs + k) * 2]     = std::cos(angle);
                cs[(n * pairs + k) * 2 + 1] = std::sin(angle);
            }
        }
    }
    return vai::Tensor({uint32_t(ids.size()), pairs, 2}).setFloats(cs);
}

// All five forward inputs from a latent [C, H, W], caption features
// [Nc0, capFeatDim] and the flow time t.
inline vai::TensorDict makeInputs(const ModelConfig& cfg, const float* latent, uint32_t H, uint32_t W,
                                  const vai::Tensor& capFeats, float t)
{
    const uint32_t Ht = H / cfg.patchSize, Wt = W / cfg.patchSize;
    const uint32_t Nx = padTo(Ht * Wt, cfg.seqMultiple);
    const uint32_t Nc = padTo(capFeats.shape()[0], cfg.seqMultiple);

    return {
        {"inPatches",  patchify(cfg, latent, H, W)},
        {"inCapFeats", capFeats},
        {"inTimestep", timestepFeatures(cfg, t)},
        {"inImageCs",  ropeTable(cfg, imagePositions(Ht, Wt, Nx, Nc))},
        {"inCapCs",    ropeTable(cfg, captionPositions(Nc))},
    };
}

}   // namespace zimage

#endif // ZIMAGE_MODEL_H
