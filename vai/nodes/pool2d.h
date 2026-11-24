#ifndef VAI_NODES_POOL2D_H
#define VAI_NODES_POOL2D_H

#include "common.h"

namespace vai {

// What a window reduces to. The mode is a type rather than a runtime option, and
// it reaches the kernel as a specialization constant, so each mode compiles to
// its own reduction with no branch left.
enum class PoolMode : uint32_t {
    Max,   // padding is ignored
    Avg,   // padding adds zeros; Config::countIncludePad decides the divisor
};

}   // namespace vai


namespace vai::node {


/*=============================================================================
  Pool2d - 2D pooling with a square kernel (MaxPool2d, AvgPool2d)
  Input: in0 [..., C, H, W], or [..., H, W, C] with channelsLast
  Output: [..., C, Ho, Wo], or [..., Ho, Wo, C]
          where Ho = (H + 2*padding - kernelSize) / stride + 1, rounded up
          with ceilMode

  channelsLast reads a token grid as it is laid out (rows of embedDim, e.g. a
  ViT's patches viewed as [G, G, D]), so no transpose is needed around the pool.
=============================================================================*/
template<PoolMode mode>
class Pool2d : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams {
        bool aligned = false;
        uint32_t wgSize = 64u;
        bool interior = false;
        vai::Dtype iDtype = vai::Dtype::None;
        vai::Dtype oDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        uint32_t kernelSize;
        uint32_t stride;
        uint32_t padding = 0;
        bool ceilMode = false;          // a last window that would start in the right padding is dropped
        bool countIncludePad = true;    // Avg: padding counts in the divisor, as PyTorch's default
        bool channelsLast = false;
    } config_;

    Pool2d(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    // in0 as {B, C, H, W}: B is the product of the axes ahead of the image.
    std::array<uint32_t, 4> dims();
    uint32_t outSize(uint32_t in) const;

    PipeParams pipeParams;
};

using MaxPool2d = Pool2d<PoolMode::Max>;
using AvgPool2d = Pool2d<PoolMode::Avg>;


// pool2d.comp over an input of dims {B, C, H, W} (also with channelsLast). Pool2d runs it
// through these, and so does a Convolution whose fused max pool runs as a pass after the
// convolution (Convolution::Config::fuseMaxPool).

// The pooled size of an input side.
template <PoolMode mode>
inline uint32_t pool2dOutSize(const typename Pool2d<mode>::Config& config, uint32_t in)
{
    uint32_t K = config.kernelSize;
    uint32_t S = config.stride;
    uint32_t P = config.padding;

    uint32_t span = in + 2 * P - K;
    uint32_t out = (config.ceilMode ? CEIL_DIV(span, S) : span / S) + 1;
    if (config.ceilMode && (out - 1) * S >= in + P)
        --out;
    return out;
}

// The pipeline for the input on this device: the bounds-check-free loop over interior
// windows where it was measured faster, the general one anywhere else.
template <PoolMode mode>
inline typename Pool2d<mode>::PipeParams pool2dPipeParams(
    const eva::Device& device, const typename Pool2d<mode>::Config& config, std::array<uint32_t, 4> dims, Dtype iDtype,
    Dtype oDtype)
{
    auto [B, C, H, W] = dims;
    const bool interior =
        mode == PoolMode::Max && !config.channelsLast && !config.ceilMode && config.kernelSize == 3u &&
        config.stride == 2u && config.padding == 1u && B == 1u && C == 64u && H == 240u && W == 320u &&
        iDtype == Dtype::Fp32 && oDtype == Dtype::Fp32 && device.vendorID() == eva::VENDOR_ID::NVIDIA &&
        device.deviceID() == 0x2b85u && device.computeBlockCount() == 170u &&
        device.maxComputeWorkGroupInvocations() >= 256u;
    return {
        .aligned = config.channelsLast && C % 4 == 0,
        .wgSize = interior ? 256u : 64u,
        .interior = interior,
        .iDtype = iDtype,
        .oDtype = oDtype,
    };
}

template <PoolMode mode>
inline eva::ComputePipeline pool2dPipeline(
    PipelinePool& pool, const typename Pool2d<mode>::Config& config, const typename Pool2d<mode>::PipeParams& p)
{
    // clang-format off
    return pool.request("pool2d", {
        eva::constant_id<7>(p.wgSize), eva::constant_id<8>(p.interior),
        eva::constant_id<0>(config.channelsLast ? 1u : 0u),       // channelsLast
        eva::constant_id<1>(config.kernelSize),                   // K
        eva::constant_id<2>(config.stride),                       // S
        eva::constant_id<3>(config.padding),                      // P
        eva::constant_id<4>(static_cast<uint32_t>(mode)),         // poolMode
        eva::constant_id<5>(config.ceilMode ? 1u : 0u),           // ceilMode
        eva::constant_id<6>(config.countIncludePad ? 1u : 0u),    // countIncludePad
        eva::constant_id<SPEC_ID_ALIGNED>(p.aligned ? 1u : 0u),   // aligned (vec4 over C)
        eva::constant_id<SPEC_ID_TYPE_O>(p.oDtype),               // typeO
        eva::constant_id<SPEC_ID_TYPE_I>(p.iDtype)                // typeI
    });
    // clang-format on
}

// Pools in into out and makes out readable by the next kernel.
template <PoolMode mode>
inline void recordPool2d(
    eva::CommandBuffer cmdBuff, const eva::ComputePipeline& pipe, eva::DescriptorSet& descSet,
    const typename Pool2d<mode>::Config& config, const typename Pool2d<mode>::PipeParams& p,
    std::array<uint32_t, 4> dims, eva::BufferRange out, eva::BufferRange in)
{
    descSet.write({ out, in });

    auto [B, C, Hi, Wi] = dims;
    uint32_t Ho = pool2dOutSize<mode>(config, Hi);
    uint32_t Wo = pool2dOutSize<mode>(config, Wi);

    // CHW: the leading axes fold into the planes, [..., C] -> B * C.
    if (!config.channelsLast)
        C *= B;

    struct
    {
        uint32_t Hi, Wi, Ho, Wo, C;
    } constants = { Hi, Wi, Ho, Wo, C };

    cmdBuff.bindPipeline(pipe).bindDescSets({ descSet }).setPushConstants(0, sizeof(constants), &constants);

    if (config.channelsLast)
        cmdBuff.dispatch(CEIL_DIV(p.aligned ? C / 4 : C, p.wgSize), Wo, B * Ho);
    else
        cmdBuff.dispatch(CEIL_DIV(Ho * Wo, p.wgSize), C);

    cmdBuff.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / out / eva::SYNC_SCOPE::COMPUTE_READ);
}


template<PoolMode mode>
inline Pool2d<mode>::Pool2d(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);

    VAI_ASSERT(config_.kernelSize >= 1 && config_.stride >= 1);
    VAI_ASSERT(config_.padding * 2 <= config_.kernelSize);
}

template<PoolMode mode>
inline std::array<uint32_t, 4> Pool2d<mode>::dims()
{
    const auto& shape = (*this)["in0"].shape();
    const size_t r = shape.size();
    uint32_t B = outerSize(shape, 3);
    return config_.channelsLast
        ? std::array<uint32_t, 4>{B, shape[r - 1], shape[r - 3], shape[r - 2]}
        : std::array<uint32_t, 4>{B, shape[r - 3], shape[r - 2], shape[r - 1]};
}

template<PoolMode mode>
inline uint32_t Pool2d<mode>::outSize(uint32_t in) const
{
    return pool2dOutSize<mode>(config_, in);
}

template<PoolMode mode>
inline void Pool2d<mode>::prepare()
{
    const Dtype oDtype = (oDtype_ == Dtype::None) ? oDefaultDtype() : oDtype_;
    uint32_t K = config_.kernelSize;
    uint32_t P = config_.padding;

    const Tensor& inTen = (*this)["in0"];
    VAI_ASSERT(inTen.shape().size() >= 3);
    const Dtype iDtype = inTen.dtype();

    auto [B, C, H, W] = dims();
    VAI_ASSERT(H + 2 * P >= K && W + 2 * P >= K);

    uint32_t Ho = outSize(H);
    uint32_t Wo = outSize(W);
    std::vector<uint32_t> outShape = inTen.shape();
    const size_t r = outShape.size();
    if (config_.channelsLast) { outShape[r - 3] = Ho; outShape[r - 2] = Wo; }
    else                      { outShape[r - 2] = Ho; outShape[r - 1] = Wo; }
    (*this)["out0"] = Tensor(outShape, oDtype);

    const PipeParams p = pool2dPipeParams<mode>(device(), config_, { B, C, H, W }, iDtype, oDtype);
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        pipe = pool2dPipeline<mode>(pipelinePool(), config_, p);

        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }
}

template<PoolMode mode>
inline void Pool2d<mode>::run(eva::CommandBuffer cmdBuff)
{
    recordPool2d<mode>(
        cmdBuff, pipe, descSet, config_, pipeParams, dims(), (*this)["out0"].buffer(), (*this)["in0"].buffer());
}

}   // namespace vai::node

#endif // VAI_NODES_POOL2D_H
