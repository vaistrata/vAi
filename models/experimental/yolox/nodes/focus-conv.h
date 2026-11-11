#ifndef EXPERIMENTAL_YOLOX_NODES_FOCUS_CONV_H
#define EXPERIMENTAL_YOLOX_NODES_FOCUS_CONV_H

#include "vai/nodes/convolution.h"

namespace model::experimental::yolox::node {

inline eva::SpvBlob focusConvSpv = eva::SpvBlob::readFrom(EXPERIMENTAL_YOLOX_SHADER_OUTPUT_DIR "/focus-conv.comp.spv");
inline eva::SpvBlob focusConvCMSpv =
    eva::SpvBlob::readFrom(EXPERIMENTAL_YOLOX_SHADER_OUTPUT_DIR "/focus-conv-cm.comp.spv");
inline eva::SpvBlob focusConvTransformSpv =
    eva::SpvBlob::readFrom(EXPERIMENTAL_YOLOX_SHADER_OUTPUT_DIR "/focus-conv-transform.comp.spv");

/////////////////////////////////////////////////////////////////////////////////////////
// FocusConv - YOLOX Focus folded into its stem Conv-BN-SiLU.
// Input : in0  [Ci, 2H, 2W]
// Params: weight [Co, 4Ci, 3, 3], official Megvii BatchNorm tensors [Co]
// Output: out0 [Co, H, W]
//
// It preserves the standard convolution parameter-slot/weight-loader contract,
// while its prepare/run implementation and shaders remain model-local.
/////////////////////////////////////////////////////////////////////////////////////////
class FocusConv : public vai::Node
{
public:
    using Config = vai::node::Convolution::Config;
    Config config_;

    FocusConv(uint32_t inChannels, uint32_t outChannels, uint32_t kernelSize = 3u);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    struct Plan
    {
        bool cooperativeMatrix = false;
        uint32_t wgSize = 128u;
        uint32_t wgM = 64u;
        uint32_t wgN = 32u;
        uint32_t wgK = 8u;
        uint32_t tM = 4u;
        uint32_t panelK = 32u;
        uint32_t sgM = 16u;
        uint32_t sgN = 16u;
        uint32_t accType = 1u;
        bool alignM = false;
        bool alignN = false;
        bool alignK = false;

        auto operator<=>(const Plan&) const = default;
    };

    uint32_t inputChannels_;
    vai::node::PipelineBinding gemm_;
    vai::node::PipelineBinding transform_;
    eva::Buffer transformedWeight_;
    vai::node::ShapeSpec preparedShape_;
    vai::Dtype preparedIDtype_ = vai::Dtype::None;   // the input dtype the pipeline was built for
    Plan plan_;
    bool pipelinePrepared_ = false;
    bool weightTransformed_ = false;

    vai::node::Workload makeFocusWorkload() const;
    Plan selectPlan(const vai::node::Workload& workload) const;
    void preparePipeline(const vai::node::Workload& workload, const Plan& plan);
    void transformWeight(const vai::node::Workload& workload);
};

inline FocusConv::FocusConv(uint32_t inChannels, uint32_t outChannels, uint32_t kernelSize)
    : config_({
          .inChannels = 4u * inChannels,
          .outChannels = outChannels,
          .kernelSize = kernelSize,
          .stride = 1u,
          .epilogue = vai::Epilogue::Silu,
      }),
      inputChannels_(inChannels)
{
    VAI_ASSERT(inChannels != 0u && outChannels != 0u);
    VAI_ASSERT(kernelSize == 3u);
    addSlot("out0", vai::NodeSlot::output);
    addSlot("in0", vai::NodeSlot::input);
    addSlot("weight", vai::NodeSlot::persistent);
    addSlot("scale", vai::NodeSlot::persistent);
    addSlot("bias", vai::NodeSlot::persistent);
}

inline vai::node::Workload FocusConv::makeFocusWorkload() const
{
    const vai::Dtype oDtype = (oDtype_ == vai::Dtype::None) ? oDefaultDtype() : oDtype_;
    const vai::Tensor& input = (*this)["in0"];
    const std::vector<uint32_t>& shape = input.shape();
    VAI_ASSERT(shape.size() == 3u);
    VAI_ASSERT(shape[0] == inputChannels_);
    VAI_ASSERT(shape[1] != 0u && shape[2] != 0u);
    VAI_ASSERT((shape[1] % 2u) == 0u && (shape[2] % 2u) == 0u);

    const uint32_t height = shape[1] / 2u;
    const uint32_t width = shape[2] / 2u;
    return {
        .ci = config_.inChannels,
        .co = config_.outChannels,
        .kernel = 3u,
        .stride = 1u,
        .hi = height,
        .wi = width,
        .ho = height,
        .wo = width,
        .iDtype = input.dtype(),
        .oDtype = oDtype,
        .wDtype = (*this)["weight"].dtype(),
        .fusion = vai::node::ConvolutionFusion::Scale,
    };
}

inline FocusConv::Plan FocusConv::selectPlan(const vai::node::Workload& workload) const
{
    const eva::Device currentDevice = device();
    const vai::node::ConvolutionCaps caps{
        .hasCM = vai::node::hasCooperativeMatrixUnit(currentDevice.vendorID(), currentDevice.architectureID()),
        .accFP16 = vai::node::supportsCMShape(currentDevice, vai::Dtype::Fp16, vai::Dtype::Fp16),
    };

    Plan plan;
    plan.cooperativeMatrix = workload.wDtype == vai::Dtype::Fp16 && vai::node::enableCM(currentDevice, caps);

    if (plan.cooperativeMatrix)
    {
        const bool navi48Large =
            currentDevice.vendorID() == eva::VENDOR_ID::AMD &&
            vai::node::classifyAMDArchitecture(currentDevice.vendorID(), currentDevice.deviceID()) ==
                vai::node::AMDArchitecture::RDNA4 &&
            currentDevice.computeBlockCount() == 64u && workload.hi == 320u && workload.wi == 320u &&
            workload.co == 64u;
        if (navi48Large)
        {
            plan.wgM = 64u;
            plan.wgN = 16u;
            plan.panelK = 112u;
            plan.accType = 1u;
        }
        else
        {
            plan.wgM = 32u;
            plan.wgN = 32u;
            plan.panelK = 32u;
            plan.accType = vai::node::selectAccType(currentDevice, caps) == vai::Dtype::Fp16 ? 0u : 1u;
        }
        plan.wgSize = 128u;
        return plan;
    }

    const bool tuned640 = workload.hi == 320u && workload.wi == 320u;
    if (tuned640 && workload.co == 32u)
    {
        plan.wgSize = 64u;
        plan.wgM = 32u;
        plan.wgK = 4u;
    }
    else if (tuned640 && workload.co == 64u)
    {
        plan.wgSize = 128u;
        plan.wgM = 64u;
        plan.wgK = 4u;
    }

    const uint32_t reduction = workload.ci * 9u;
    plan.alignM = (workload.co % plan.wgM) == 0u;
    plan.alignN = (workload.spatialSize() % plan.wgN) == 0u;
    plan.alignK = (reduction % plan.wgK) == 0u;
    return plan;
}

inline void FocusConv::preparePipeline(const vai::node::Workload& workload, const Plan& nextPlan)
{
    const vai::node::ShapeSpec shape = vai::node::ShapeSpec::fromWorkload(workload);
    if (pipelinePrepared_ && plan_ == nextPlan && preparedShape_.matchShape(workload) &&
        preparedIDtype_ == workload.iDtype)
        return;

    if (nextPlan.cooperativeMatrix)
    {
        eva::SpecializationConstant specialization{
            eva::constant_id<0>(nextPlan.wgSize), eva::constant_id<1>(nextPlan.wgM), eva::constant_id<2>(nextPlan.wgN),
            eva::constant_id<3>(nextPlan.panelK), eva::constant_id<4>(nextPlan.sgM), eva::constant_id<5>(nextPlan.sgN),
        };
        vai::node::withShape(specialization, shape, true);
        vai::node::withDTypes(specialization, workload.oDtype, workload.iDtype, workload.wDtype, vai::Dtype::Fp32,
                              nextPlan.accType, {.includeAcc = true});
        vai::node::withFusion(specialization, uint32_t(vai::Epilogue::Silu), 1u, 1u);
        gemm_ = vai::node::makePipelineBinding(device().createComputePipeline({
            .csStage = {focusConvCMSpv, std::move(specialization)},
            .requiredSubgroupSize = 32u,
        }));

        if (!transform_.pipeline)
        {
            transform_ = vai::node::makePipelineBinding(device().createComputePipeline({
                .csStage = focusConvTransformSpv,
                .requiredSubgroupSize = 32u,
            }));
        }
    }
    else
    {
        eva::SpecializationConstant specialization{
            eva::constant_id<0>(nextPlan.wgSize), eva::constant_id<1>(nextPlan.wgM),
            eva::constant_id<2>(nextPlan.wgN),    eva::constant_id<3>(nextPlan.wgK),
            eva::constant_id<4>(nextPlan.tM),     eva::constant_id<5>(nextPlan.alignM),
            eva::constant_id<6>(nextPlan.alignN), eva::constant_id<7>(nextPlan.alignK),
        };
        vai::node::withShape(specialization, shape, true);
        vai::node::withDTypes(specialization, workload.oDtype, workload.iDtype, workload.wDtype, workload.wDtype, 0u,
                              {.includeWeight = true});
        vai::node::withFusion(specialization, uint32_t(vai::Epilogue::Silu), 1u, 1u);
        gemm_ = vai::node::makePipelineBinding(device().createComputePipeline({
            .csStage = {focusConvSpv, std::move(specialization)},
        }));
    }

    plan_ = nextPlan;
    preparedShape_ = shape;
    preparedIDtype_ = workload.iDtype;
    pipelinePrepared_ = true;
}

inline void FocusConv::prepare()
{
    const vai::node::Workload workload = makeFocusWorkload();
    const vai::Tensor& input = (*this)["in0"];
    const vai::Tensor& weight = (*this)["weight"];
    VAI_ASSERT(workload.wDtype == vai::Dtype::Fp32 || workload.wDtype == vai::Dtype::Fp16);
    VAI_ASSERT(workload.wDtype == vai::Dtype::Fp16 || workload.oDtype != vai::Dtype::Fp16);
    VAI_ASSERT(workload.wDtype == vai::Dtype::Fp16 ||
               workload.iDtype != vai::Dtype::Fp16);   // fp16 activations need fp16 weights
    VAI_ASSERT(weight.isShapeOf(workload.co, workload.ci, 3u, 3u));
    VAI_ASSERT((*this)["scale"].dtype() == workload.wDtype && (*this)["scale"].isShapeOf(workload.co));
    VAI_ASSERT((*this)["bias"].dtype() == workload.wDtype && (*this)["bias"].isShapeOf(workload.co));

    preparePipeline(workload, selectPlan(workload));
    (*this)["out0"] = vai::Tensor({workload.co, workload.ho, workload.wo}, workload.oDtype);
}

inline void FocusConv::transformWeight(const vai::node::Workload& workload)
{
    if (!plan_.cooperativeMatrix || weightTransformed_)
        return;

    const uint32_t rawK = workload.ci * 9u;
    const uint32_t paddedK = CEIL_DIV(rawK, 16u) * 16u;
    const uint32_t paddedM = CEIL_DIV(workload.co, 16u) * 16u;
    auto transformCmd = device().newCommandBuffer(eva::queue_compute);
    transformCmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT);
    const bool recorded = vai::node::ensureWeightTransform(
        device(), transformCmd, transform_, (*this)["weight"].buffer(), transformedWeight_,
        vai::node::makePushConstants(workload), size_t(paddedM) * paddedK * sizeof(uint16_t), paddedM * paddedK,
        weightTransformed_);
    VAI_ASSERT(recorded);
    transformCmd.end();
    device().queue() << transformCmd << eva::waiting;
}

inline void FocusConv::run(eva::CommandBuffer cmdBuff)
{
    const vai::node::Workload workload = makeFocusWorkload();
    transformWeight(workload);

    vai::Tensor& output = (*this)["out0"];
    const eva::Descriptor weight =
        plan_.cooperativeMatrix ? eva::Descriptor(transformedWeight_) : eva::Descriptor((*this)["weight"].buffer());
    gemm_.descriptorSet.write({
        output.buffer(),
        (*this)["in0"].buffer(),
        weight,
        (*this)["bias"].buffer(),
        (*this)["scale"].buffer(),
    });
    vai::node::recordDispatch(cmdBuff, gemm_, vai::node::makePushConstants(workload),
                              CEIL_DIV(workload.spatialSize(), plan_.wgN), CEIL_DIV(workload.co, plan_.wgM));
    cmdBuff.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / output.buffer() / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace model::experimental::yolox::node

#endif   // EXPERIMENTAL_YOLOX_NODES_FOCUS_CONV_H
