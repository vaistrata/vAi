#ifndef EXPERIMENTAL_DIFFUSION_POLICY_NODES_CONV_TRANSPOSE_1D_H
#define EXPERIMENTAL_DIFFUSION_POLICY_NODES_CONV_TRANSPOSE_1D_H

#include "vai/nodes/common.h"

namespace model::experimental::diffusion_policy::node {


/*=============================================================================
  ConvTranspose1d - transposed 1D convolution (torch.nn.ConvTranspose1d)
  Input : in0  [..., Ci, Ti]
  Output: out0 [..., Co, To], To = (Ti − 1) · stride − 2 · padding + K

  O[co, t] = epilogue(bias[co] + Σ_ci Σ_j W[ci, co, j] · I[ci, s]),  t + padding − j = s · stride

  The diffusion UNet's upsample, ConvTranspose1d(C, C, 4, 2, 1): T doubles.
  weight [Ci, Co, K] (torch's transposed layout) and bias [Co] are parameters,
  any dtype; bias applies when the loader filled it. Axes ahead of Ci are
  independent sequences, one dispatch z each.

  Each output gathers the ceil(K / stride) inputs that scatter into it, so no
  atomics; one thread per (co, t), accumulating over Ci in fp32. The UNet's
  upsamples are small (Ci·Co·To·2 ≈ 17M MACs at most for pushT), so this direct
  kernel stays the one path.
=============================================================================*/
class ConvTranspose1d : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;
        vai::Dtype wDtype = vai::Dtype::None;
        vai::Dtype pDtype = vai::Dtype::None;   // bias
        bool hasBias = false;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        uint32_t inChannels;
        uint32_t outChannels;
        uint32_t kernelSize;
        uint32_t stride = 1;
        uint32_t padding = 0;
        vai::Epilogue epilogue = vai::Epilogue::None;
    } config_;

    static constexpr uint32_t maxTaps = 8;   // the shader's MAX_TAPS: ceil(K / stride)

    ConvTranspose1d(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

    // The weight loader filled "bias"; fixed once the weights are up.
    bool hasBias() const { return (*this)["bias"].validShape(); }

private:
    PipeParams pipeParams;
    uint32_t seqs_ = 0, Ti_ = 0, To_ = 0;
};


inline ConvTranspose1d::ConvTranspose1d(Config config)
: config_(config)
{
    VAI_ASSERT(config_.inChannels != 0u && config_.outChannels != 0u);
    VAI_ASSERT(config_.kernelSize != 0u && config_.stride != 0u);
    VAI_ASSERT(CEIL_DIV(config_.kernelSize, config_.stride) <= maxTaps);
    VAI_ASSERT(2u * config_.padding < config_.kernelSize);   // To > 0 down to Ti = 1
    VAI_ASSERT(config_.epilogue < vai::Epilogue::Max);
    addSlot("out0", vai::NodeSlot::output);
    addSlot("in0", vai::NodeSlot::input);
    addSlot("weight", vai::NodeSlot::persistent);
    addSlot("bias", vai::NodeSlot::persistent);
}

inline void ConvTranspose1d::prepare()
{
    const vai::Dtype oDtype = (oDtype_ == vai::Dtype::None) ? oDefaultDtype() : oDtype_;
    const vai::Tensor& inTen = (*this)["in0"];
    const vai::Tensor& wTen = (*this)["weight"];
    VAI_ASSERT(inTen.validShape());

    const auto& shape = inTen.shape();
    const size_t rank = shape.size();
    VAI_ASSERT(rank >= 2u && shape[rank - 2u] == config_.inChannels);
    Ti_ = shape[rank - 1u];
    To_ = (Ti_ - 1u) * config_.stride + config_.kernelSize - 2u * config_.padding;
    seqs_ = uint32_t(inTen.numElements() / (size_t(config_.inChannels) * Ti_));
    VAI_ASSERT(Ti_ != 0u && seqs_ != 0u);
    VAI_ASSERT(wTen.isShapeOf(config_.inChannels, config_.outChannels, config_.kernelSize));
    if (hasBias())
        VAI_ASSERT((*this)["bias"].isShapeOf(config_.outChannels));

    const PipeParams p = {
        .iDtype = inTen.dtype(),
        .wDtype = wTen.dtype(),
        .pDtype = hasBias() ? (*this)["bias"].dtype() : vai::Dtype::Fp32,
        .hasBias = hasBias(),
    };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        // clang-format off
        pipe = pipelinePool().request(EXPERIMENTAL_DIFFUSION_POLICY_SHADER_OUTPUT_DIR, "conv-transpose-1d", {
            eva::constant_id<0>(config_.kernelSize),                          // K
            eva::constant_id<1>(config_.stride),                              // stride
            eva::constant_id<2>(config_.padding),                             // pad
            eva::constant_id<SPEC_ID_TYPE_O>(oDtype),                         // typeO
            eva::constant_id<SPEC_ID_TYPE_I>(p.iDtype),                       // typeI
            eva::constant_id<SPEC_ID_TYPE_W>(p.wDtype),                       // typeW
            eva::constant_id<SPEC_ID_TYPE_P>(p.pDtype),                       // typeP (bias)
            eva::constant_id<SPEC_ID_EPILOGUE>((uint32_t) config_.epilogue),  // epilogue
            eva::constant_id<SPEC_ID_HAS_BIAS>(p.hasBias ? 1u : 0u)           // hasBias
        });
        // clang-format on
        if (!descSet)
            descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
    }

    std::vector<uint32_t> outShape = shape;
    outShape[rank - 2u] = config_.outChannels;
    outShape[rank - 1u] = To_;
    (*this)["out0"] = vai::Tensor(outShape, oDtype);
}

inline void ConvTranspose1d::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf = (*this)["in0"].buffer();
    auto wBuf = (*this)["weight"].buffer();
    // An unused bias binds the weight; specialization removes every read.
    auto biasBuf = pipeParams.hasBias ? (*this)["bias"].buffer() : wBuf;

    descSet.write({ outBuf, inBuf, wBuf, biasBuf });

    struct {
        uint32_t Ci, Co, Ti, To;
    } constants = { config_.inChannels, config_.outChannels, Ti_, To_ };

    cmdBuff
        .bindPipeline(pipe)
        .setPushConstants(0, sizeof(constants), &constants)
        .bindDescSets({descSet})
        .dispatch(CEIL_DIV(config_.outChannels * To_, 64u), 1u, seqs_)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace model::experimental::diffusion_policy::node

#endif // EXPERIMENTAL_DIFFUSION_POLICY_NODES_CONV_TRANSPOSE_1D_H
