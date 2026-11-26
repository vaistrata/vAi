#ifndef EXPERIMENTAL_DIFFUSION_POLICY_NODES_CONVOLUTION_1D_H
#define EXPERIMENTAL_DIFFUSION_POLICY_NODES_CONVOLUTION_1D_H

#include "vai/nodes/common.h"

namespace model::experimental::diffusion_policy::node {


/*=============================================================================
  Convolution1d - 1D convolution over a [C, T] sequence (torch.nn.Conv1d)
  Input : in0  [..., Ci, Ti], optional residual [..., Co, To]
  Output: out0 [..., Co, To], To = (Ti - 1) / stride + 1

  O[co, t] = act(Σ_{ci, j} W[co, ci, j] · I[ci, t·stride + j − K/2] + bias[co]) + R[co, t]

  Same padding K/2 with an odd K: torch's Conv1d(k, stride, padding=k//2), which
  covers the diffusion UNet's k5 blocks, its k3 stride-2 downsample and its 1x1
  convs. Axes ahead of Ci are independent sequences, one dispatch z each.
  weight [Co, Ci, K] and bias [Co] are parameters; bias applies when the loader
  filled it. acceptResidual opens the "residual" input (in0's dtype), added
  after the activation or, with ResidualOrder::BeforeActivation, before it.

  Self-contained: it shares no code with vai::node::Convolution. One path so far,
  the cooperative-matrix one (conv1d-cm): fp16 weights on a device with a 16x16x16
  fp16 coopmat shape and 32-lane subgroups. It is an implicit GEMM of M = Co,
  N = To over Ci·K, the input gathered on the fly; the weight is packed once, at
  the first run(), into 16x16 fragments (conv1d-cm-transform). Accumulation is
  fp32 where the device has the f16 x f16 -> f32 shape, fp16 otherwise.

  The tile follows To: the UNet's sequences are short (pushT: 16 -> 8 -> 4), so
  To <= 16 takes a 64 x 16 tile (four subgroups down Co), longer ones 32 x 32.
  The pipeline is requested in prepare() once the shape is known and
  re-requested from the pool when the parameters change.
=============================================================================*/
class Convolution1d : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;
    eva::ComputePipeline transformPipe;
    eva::DescriptorSet transformDescSet;
    eva::Buffer packedWeight;   // [Mp/16][Kp/16][16][16] fp16, written once

public:
    struct PipeParams {
        uint32_t wgSize = 0;    // threads per workgroup, 32-lane subgroups
        uint32_t wgM = 0;       // Co rows per workgroup, a multiple of 16
        uint32_t wgN = 0;       // To columns per workgroup, a multiple of 16
        uint32_t panelK = 0;    // reduction rows staged per shared-memory panel
        bool accF32 = false;    // fp32 coopmat accumulators, else fp16
        vai::Dtype iDtype = vai::Dtype::None;
        vai::Dtype pDtype = vai::Dtype::None;   // bias
        bool hasBias = false;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        uint32_t inChannels;
        uint32_t outChannels;
        uint32_t kernelSize;
        uint32_t stride = 1;

        bool acceptResidual = false;   // input "residual"
        vai::ResidualOrder residualOrder = vai::ResidualOrder::AfterActivation;
        vai::ActType actType = vai::ActType::None;
    } config_;

    Convolution1d(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

    // The weight loader filled "bias"; fixed once the weights are up.
    bool hasBias() const { return (*this)["bias"].validShape(); }

private:
    void packWeight();

    PipeParams pipeParams;
    bool weightPacked_ = false;
    uint32_t seqs_ = 0, Ti_ = 0, To_ = 0;   // sequences, input / output lengths
};


// The device's 16x16x16 fp16 coopmat shapes: whether it has one accumulating
// into accType at subgroup scope.
inline bool conv1dHasCMShape(const eva::Device& device, eva::COMPONENT_TYPE accType)
{
    using CT = eva::COMPONENT_TYPE;
    for (const auto& shape : device.cooperativeMatrixProperties())
    {
        if (shape.aType == CT::FLOAT16 && shape.bType == CT::FLOAT16 && shape.cType == accType &&
            shape.resultType == accType && shape.scope == eva::SCOPE::SUBGROUP && !shape.saturatingAccumulation &&
            shape.M == 16u && shape.N == 16u && shape.K == 16u)
            return true;
    }
    return false;
}

// Tile geometry: four 32-lane subgroups, each one 16x16 output tile.
inline Convolution1d::PipeParams conv1dTuning(uint32_t To, eva::Device device)
{
    Convolution1d::PipeParams t;
    t.wgSize = 128u;
    t.wgN = To <= 16u ? 16u : 32u;
    t.wgM = (t.wgSize / 32u) * 256u / t.wgN;
    t.panelK = 32u;
    t.accF32 = conv1dHasCMShape(device, eva::COMPONENT_TYPE::FLOAT32);
    return t;
}


inline Convolution1d::Convolution1d(Config config)
: config_(config)
{
    VAI_ASSERT(config_.inChannels != 0u && config_.outChannels != 0u);
    VAI_ASSERT(config_.kernelSize != 0u && config_.kernelSize % 2u != 0u);
    VAI_ASSERT(config_.stride != 0u);
    VAI_ASSERT(config_.actType < vai::ActType::Max);
    addSlot("out0", vai::NodeSlot::output);
    addSlot("in0", vai::NodeSlot::input);
    addSlot("weight", vai::NodeSlot::persistent);
    addSlot("bias", vai::NodeSlot::persistent);
    if (config_.acceptResidual)
        addSlot("residual", vai::NodeSlot::input);
}

inline void Convolution1d::prepare()
{
    const vai::Dtype oDtype = (oDtype_ == vai::Dtype::None) ? oDefaultDtype() : oDtype_;
    const vai::Tensor& inTen = (*this)["in0"];
    const vai::Tensor& wTen = (*this)["weight"];
    VAI_ASSERT(inTen.validShape());

    const auto& shape = inTen.shape();
    const size_t rank = shape.size();
    VAI_ASSERT(rank >= 2u && shape[rank - 2u] == config_.inChannels);
    Ti_ = shape[rank - 1u];
    To_ = (Ti_ - 1u) / config_.stride + 1u;   // (Ti + 2·(K/2) − K) / stride + 1 with K odd
    seqs_ = uint32_t(inTen.numElements() / (size_t(config_.inChannels) * Ti_));
    VAI_ASSERT(Ti_ != 0u && seqs_ != 0u);
    VAI_ASSERT(seqs_ <= device().maxComputeWorkGroupCount()[2]);

    VAI_ASSERT(wTen.dtype() == vai::Dtype::Fp16 && "Convolution1d: the coopmat path takes fp16 weights");
    VAI_ASSERT(wTen.isShapeOf(config_.outChannels, config_.inChannels, config_.kernelSize));
    if (hasBias())
        VAI_ASSERT((*this)["bias"].isShapeOf(config_.outChannels));
    if (config_.acceptResidual)
    {
        const vai::Tensor& rTen = (*this)["residual"];
        VAI_ASSERT(rTen.dtype() == inTen.dtype());
        VAI_ASSERT(rTen.numElements() == size_t(seqs_) * config_.outChannels * To_);
    }

    const eva::Device dev = device();
    VAI_ASSERT(dev.supportsCooperativeMatrix() && dev.minSubgroupSize() <= 32u && 32u <= dev.maxSubgroupSize() &&
               (conv1dHasCMShape(dev, eva::COMPONENT_TYPE::FLOAT32) ||
                conv1dHasCMShape(dev, eva::COMPONENT_TYPE::FLOAT16)) &&
               "Convolution1d: no 16x16x16 fp16 cooperative matrix on this device (no other path yet)");

    PipeParams p = conv1dTuning(To_, dev);
    p.iDtype = inTen.dtype();
    p.pDtype = hasBias() ? (*this)["bias"].dtype() : vai::Dtype::Fp32;
    p.hasBias = hasBias();
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        // clang-format off
        pipe = pipelinePool().request(EXPERIMENTAL_DIFFUSION_POLICY_SHADER_OUTPUT_DIR, "conv1d-cm", {
            eva::constant_id<0>(p.wgSize),                                   // wgSize
            eva::constant_id<1>(p.wgM),                                      // wgM
            eva::constant_id<2>(p.wgN),                                      // wgN
            eva::constant_id<3>(p.panelK),                                   // panelK
            eva::constant_id<4>(config_.kernelSize),                         // K
            eva::constant_id<5>(config_.stride),                             // stride
            eva::constant_id<6>(p.accF32 ? 1u : 0u),                         // accF32
            eva::constant_id<7>(config_.acceptResidual ? 1u : 0u),           // hasResidual
            eva::constant_id<8>((uint32_t) config_.residualOrder),           // residualOrder
            eva::constant_id<SPEC_ID_TYPE_O>(oDtype),                        // typeO
            eva::constant_id<SPEC_ID_TYPE_I>(p.iDtype),                      // typeI
            eva::constant_id<SPEC_ID_TYPE_P>(p.pDtype),                      // typeP (bias)
            eva::constant_id<SPEC_ID_ACT_TYPE>((uint32_t) config_.actType), // actType
            eva::constant_id<SPEC_ID_HAS_BIAS>(p.hasBias ? 1u : 0u)          // hasBias
        }, { .requiredSubgroupSize = 32u });
        // clang-format on
        if (!descSet)
            descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
    }
    if (!transformPipe)
    {
        transformPipe = pipelinePool().request(EXPERIMENTAL_DIFFUSION_POLICY_SHADER_OUTPUT_DIR, "conv1d-cm-transform", eva::SpecializationConstant{});
        transformDescSet = vai::gVaiDestSetPool(transformPipe.descSetLayout(0));
    }

    std::vector<uint32_t> outShape = shape;
    outShape[rank - 2u] = config_.outChannels;
    outShape[rank - 1u] = To_;
    (*this)["out0"] = vai::Tensor(outShape, oDtype);
}

// Packs W [Co, Ci, K] once into the GEMM's 16x16 fragments, on its own submission
// ahead of the first recording that reads them.
inline void Convolution1d::packWeight()
{
    if (weightPacked_)
        return;

    const uint32_t Co = config_.outChannels;
    const uint32_t CiK = config_.inChannels * config_.kernelSize;
    const uint32_t elements = ROUNDUP(Co, 16u) * ROUNDUP(CiK, 16u);
    packedWeight = device().createBuffer({
        .size = size_t(elements) * sizeof(uint16_t),
        .usage = eva::BUFFER_USAGE::STORAGE_BUFFER,
        .reqMemProps = eva::MEMORY_PROPERTY::DEVICE_LOCAL,
    });
    transformDescSet.write({ packedWeight, (*this)["weight"].buffer() });

    struct {
        uint32_t Co, CiK;
    } constants = { Co, CiK };

    auto cmd = device().newCommandBuffer(eva::queue_compute);
    cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .bindPipeline(transformPipe)
        .bindDescSets({ transformDescSet })
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch2(elements)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / packedWeight / eva::SYNC_SCOPE::COMPUTE_READ);
    cmd.end();
    device().queue() << cmd << eva::waiting;
    weightPacked_ = true;
}

inline void Convolution1d::run(eva::CommandBuffer cmdBuff)
{
    packWeight();

    auto outBuf = (*this)["out0"].buffer();
    auto inBuf = (*this)["in0"].buffer();
    // Unused slots bind an existing buffer; specialization removes every read.
    auto biasBuf = pipeParams.hasBias ? (*this)["bias"].buffer() : inBuf;
    auto rBuf = config_.acceptResidual ? (*this)["residual"].buffer() : inBuf;

    descSet.write({ outBuf, inBuf, packedWeight, biasBuf, rBuf });

    struct {
        uint32_t Ci, Co, Ti, To;
    } constants = { config_.inChannels, config_.outChannels, Ti_, To_ };

    cmdBuff
        .bindPipeline(pipe)
        .setPushConstants(0, sizeof(constants), &constants)
        .bindDescSets({descSet})
        .dispatch(CEIL_DIV(To_, pipeParams.wgN), CEIL_DIV(config_.outChannels, pipeParams.wgM), seqs_)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace model::experimental::diffusion_policy::node

#endif // EXPERIMENTAL_DIFFUSION_POLICY_NODES_CONVOLUTION_1D_H
