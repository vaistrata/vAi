#ifndef ZIMAGE_VAE_CONV_H
#define ZIMAGE_VAE_CONV_H

#include "vai/nodes/node-common.h"

namespace zimage {

/*=============================================================================
  VaeConvNode - stride-1 "same" 1x1 / 3x3 convolution with bias, for the VAE
  decoder, optionally reading its input 2x nearest-upsampled and adding a
  residual
  Input : in0 [Ci, H, W]; residual [Co, Ho, Wo] when hasResidual
  Params: weight [Ci, k*k, Cop] fp16 (packWeight of [Co, Ci, k, k]),
          bias [Cop] fp16 (packBias); Cop = Co rounded up to 8
  Output: out0 [Co, Ho, Wo], (Ho, Wo) = (H, W), or (2H, 2W) when upsample;
          Wo % 4 == 0

  A direct convolution with a compile-time tile (zimage-vae-conv.glsl),
  written for the decoder on Adreno, where vai::node::Convolution's fallback
  and Winograd plans return wrong values; on the Adreno 830 a 512 -> 512 3x3
  at 64x64 takes 18.8 ms (~1 TFLOPS). Folding the upsample into the conv
  that follows it skips a 4x larger intermediate; the residual takes the place
  of the Add after a ResNet block or the attention.
=============================================================================*/
class VaeConvNode : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    static constexpr uint32_t kTileM = 8;    // the kernel's tM (channels) and tN (positions)
    static constexpr uint32_t kTileN = 4;
    static constexpr uint32_t kWgSize = 64;

    // [Co, Ci, k, k] fp32 on the host -> the kernel's [Ci, k*k, Cop] fp16.
    static vai::Tensor packWeight(const vai::Tensor& w);
    // [Co] fp32 on the host -> [Cop] fp16.
    static vai::Tensor packBias(const vai::Tensor& b);
    static uint32_t paddedChannels(uint32_t co) { return (co + kTileM - 1) / kTileM * kTileM; }

    struct Config {
        uint32_t inChannels;
        uint32_t outChannels;
        uint32_t kernelSize = 3;                // 1 or 3
        bool upsample = false;
        bool hasResidual = false;
        vai::Dtype oDtype = vai::Dtype::None;   // None: oDefaultDtype
    } config_;

    VaeConvNode(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    vai::Dtype iDtype_ = vai::Dtype::None;
};


inline VaeConvNode::VaeConvNode(Config config)
: config_(config)
{
    addSlot("out0", vai::NodeSlot::output);
    addSlot("in0", vai::NodeSlot::input);
    if (config_.hasResidual)
        addSlot("residual", vai::NodeSlot::input);
    addSlot("weight", vai::NodeSlot::persistent);
    addSlot("bias", vai::NodeSlot::persistent);

    VAI_ASSERT(config_.kernelSize == 1 || config_.kernelSize == 3);
    config_.oDtype = (config_.oDtype == vai::Dtype::None) ? oDefaultDtype() : config_.oDtype;
}

inline void VaeConvNode::prepare()
{
    const auto& c = config_;
    const vai::Tensor& in = (*this)["in0"];
    VAI_ASSERT(in.isShapeOf(c.inChannels, -1, -1));
    const uint32_t cop = paddedChannels(c.outChannels);
    VAI_ASSERT((*this)["weight"].dtype() == vai::Dtype::Fp16);
    VAI_ASSERT((*this)["weight"].isShapeOf(c.inChannels, c.kernelSize * c.kernelSize, cop));
    VAI_ASSERT((*this)["bias"].dtype() == vai::Dtype::Fp16 && (*this)["bias"].isShapeOf(cop));

    const uint32_t s = c.upsample ? 2 : 1;
    const std::vector<uint32_t> outShape = {c.outChannels, s * in.shape()[1], s * in.shape()[2]};
    VAI_ASSERT(outShape[2] % kTileN == 0);
    if (c.hasResidual)
    {
        const vai::Tensor& r = (*this)["residual"];
        VAI_ASSERT(r.shape() == outShape && r.dtype() == c.oDtype);
    }

    if (!pipe || in.dtype() != iDtype_)
    {
        iDtype_ = in.dtype();
        pipe = pipelinePool().request(ZIMAGE_SHADER_OUTPUT_DIR,
            c.kernelSize == 3 ? "zimage-vae-conv3x3" : "zimage-vae-conv1x1", {
            eva::constant_id<0>(c.upsample ? 1u : 0u),
            eva::constant_id<1>(c.hasResidual ? 1u : 0u),
            eva::constant_id<SPEC_ID_TYPE_I>(iDtype_),
            eva::constant_id<SPEC_ID_TYPE_O>(c.oDtype),
        });
        if (!descSet)
            descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
    }

    (*this)["out0"] = vai::Tensor(outShape, c.oDtype);
}

inline vai::Tensor VaeConvNode::packWeight(const vai::Tensor& w)
{
    VAI_ASSERT(w.shape().size() == 4 && w.shape()[2] == w.shape()[3]);
    VAI_ASSERT(w.dtype() == vai::Dtype::Fp32 && w.hostData());
    const uint32_t co = w.shape()[0], ci = w.shape()[1], taps = w.shape()[2] * w.shape()[3];
    const uint32_t cop = paddedChannels(co);
    const float* src = static_cast<const float*>(w.hostData());
    std::vector<float> dst(size_t(ci) * taps * cop, 0.0f);
    for (uint32_t o = 0; o < co; ++o)
        for (uint32_t i = 0; i < ci; ++i)
            for (uint32_t t = 0; t < taps; ++t)
                dst[(size_t(i) * taps + t) * cop + o] = src[(size_t(o) * ci + i) * taps + t];
    return vai::Tensor({ci, taps, cop}, vai::Dtype::Fp16).setFloats(std::move(dst));
}

inline vai::Tensor VaeConvNode::packBias(const vai::Tensor& b)
{
    VAI_ASSERT(b.shape().size() == 1 && b.dtype() == vai::Dtype::Fp32 && b.hostData());
    const float* src = static_cast<const float*>(b.hostData());
    std::vector<float> v(src, src + b.shape()[0]);
    v.resize(paddedChannels(uint32_t(v.size())), 0.0f);
    return vai::Tensor({uint32_t(v.size())}, vai::Dtype::Fp16).setFloats(std::move(v));
}

inline void VaeConvNode::run(eva::CommandBuffer cmdBuff)
{
    const auto& c = config_;
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf = (*this)["in0"].buffer();
    // Without a residual the binding is never read; any buffer fills it.
    auto resBuf = c.hasResidual ? (*this)["residual"].buffer() : inBuf;
    descSet.write({ outBuf, inBuf, (*this)["weight"].buffer(), (*this)["bias"].buffer(), resBuf });

    const auto& in = (*this)["in0"].shape();
    const auto& out = (*this)["out0"].shape();
    struct {
        uint32_t Ci, Co, Hi, Wi, Wo, Ho;
    } constants = { c.inChannels, c.outChannels, in[1], in[2], out[2], out[1] };

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch(CEIL_DIV(out[1] * out[2], kTileN * kWgSize), paddedChannels(c.outChannels) / kTileM)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

} // namespace zimage

#endif // ZIMAGE_VAE_CONV_H
