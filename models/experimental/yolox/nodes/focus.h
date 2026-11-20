#ifndef EXPERIMENTAL_YOLOX_NODES_FOCUS_H
#define EXPERIMENTAL_YOLOX_NODES_FOCUS_H

#include "vai/nodes/convolution.h"

namespace model::experimental::yolox::node {

/*=============================================================================
  Focus - YOLOX Focus folded into its stem Conv-BN-SiLU.
  Input : in0  [..., Ci, 2H, 2W]
  Params: weight [Co, 4Ci, 3, 3], official Megvii BatchNorm tensors [Co]
  Output: out0 [..., Co, H, W]

  It preserves the standard convolution parameter-slot/weight-loader contract,
  while its prepare/run implementation and shaders remain model-local.
=============================================================================*/
class Focus : public vai::Node
{
public:
    struct Config
    {
        uint32_t inChannels;
        uint32_t outChannels;
    } config_;

    Focus(uint32_t inChannels, uint32_t outChannels, uint32_t kernelSize = 3u);
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
        vai::Dtype accType = vai::Dtype::Fp32;
        bool sharedInputPatch = false;
        bool batched = false;
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
    bool weightTransformed_ = false;

    vai::node::Workload makeFocusWorkload() const;
    Plan selectPlan(const vai::node::Workload& workload) const;
    void preparePipeline(const vai::node::Workload& workload, const Plan& plan);
    void transformWeight(const vai::node::Workload& workload);
};

inline Focus::Focus(uint32_t inChannels, uint32_t outChannels, uint32_t kernelSize)
: config_({
      .inChannels = 4u * inChannels,
      .outChannels = outChannels,
  })
, inputChannels_(inChannels)
{
    VAI_ASSERT(inChannels != 0u && outChannels != 0u);
    VAI_ASSERT(kernelSize == 3u);
    addSlot("out0", vai::NodeSlot::output);
    addSlot("in0", vai::NodeSlot::input);
    addSlot("weight", vai::NodeSlot::persistent);
    addSlot("scale", vai::NodeSlot::persistent);
    addSlot("bias", vai::NodeSlot::persistent);
}

inline vai::node::Workload Focus::makeFocusWorkload() const
{
    const vai::Dtype oDtype = (oDtype_ == vai::Dtype::None) ? oDefaultDtype() : oDtype_;
    const vai::Tensor& input = (*this)["in0"];
    const std::vector<uint32_t>& shape = input.shape();
    const size_t r = shape.size();
    VAI_ASSERT(r == 3 || r == 4);
    const uint32_t batch = vai::outerSize(shape, 3);
    VAI_ASSERT(batch && batch <= device().maxComputeWorkGroupCount()[2]);
    VAI_ASSERT(shape[r - 3] == inputChannels_);
    const uint32_t height = shape[r - 2] / 2u, width = shape.back() / 2u;
    VAI_ASSERT(height && width && height * 2u == shape[r - 2] && width * 2u == shape.back());
    return {
        .ci = config_.inChannels,
        .co = config_.outChannels,
        .kernel = 3u,
        .stride = 1u,
        .hi = height,
        .wi = width,
        .ho = height,
        .wo = width,
        .batch = batch,
        .iDtype = input.dtype(),
        .oDtype = oDtype,
        .wDtype = (*this)["weight"].dtype(),
        .fusion = vai::node::ConvolutionFusion::Scale,
    };
}

inline Focus::Plan Focus::selectPlan(const vai::node::Workload& workload) const
{
    const eva::Device currentDevice = device();
    const vai::node::ConvolutionCaps caps = vai::node::convolutionCaps(currentDevice);

    Plan plan;
    plan.batched = workload.batch > 1;
    const bool rtx5090 = (currentDevice.deviceID() == 0x2b85u);
    const bool nvidiaModelFocus =
        rtx5090 && workload.ci == 12u && workload.hi == 320u && workload.wi == 320u &&
        (workload.co == 32u || workload.co == 48u || workload.co == 64u || workload.co == 80u) &&
        workload.wDtype == vai::Dtype::Fp16 &&
        vai::node::supportsCMShape(currentDevice, vai::Dtype::Fp16, vai::Dtype::Fp32);
    plan.cooperativeMatrix =
        workload.wDtype == vai::Dtype::Fp16 && vai::node::enableCM(currentDevice, caps) &&
        (!rtx5090 || nvidiaModelFocus);

    if (plan.cooperativeMatrix)
    {
        if (nvidiaModelFocus)
        {
            // Keeping FP32 accumulation avoids the specialization-dependent failures seen on 5090.
            plan.wgM = workload.co;
            plan.wgN = 16u;
            plan.wgSize = 32u * (workload.co / 16u);
            plan.panelK = workload.co == 48u ? 128u : workload.co == 64u ? 112u : 64u;
            plan.accType = vai::Dtype::Fp32;
            return plan;
        }
        // Validated model-local presets for RGB 640x640 FP16 inference.
        // Keep other input sizes, dtypes and devices on their existing paths.
        const bool strixHalo640 =
            currentDevice.deviceID() == vai::node::kAMDStrixHaloDeviceId && workload.ci == 12u && workload.hi == 320u &&
            workload.wi == 320u;
        if (strixHalo640 && (workload.co == 32u || workload.co == 48u || workload.co == 64u || workload.co == 80u))
        {
            plan.wgM = workload.co;
            plan.wgN = 16u;
            plan.wgSize = 32u * (plan.wgM / 16u);
            plan.panelK = workload.co <= 48u ? 64u : 112u;
            plan.accType = vai::Dtype::Fp32;
            plan.sharedInputPatch = true;
            return plan;
        }
        const bool navi48Large =
            currentDevice.vendorID() == eva::VENDOR_ID::AMD &&
            vai::node::classifyAMDArchitecture(currentDevice.vendorID(), currentDevice.deviceID()) ==
                vai::node::AMDArchitecture::RDNA4 &&
            workload.hi == 320u && workload.wi == 320u && workload.co == 64u;
        if (navi48Large)
        {
            plan.wgM = 64u;
            plan.wgN = 16u;
            plan.panelK = 112u;
            plan.accType = vai::Dtype::Fp32;
        }
        else
        {
            plan.wgM = 32u;
            plan.wgN = 32u;
            plan.panelK = 32u;
            plan.accType = vai::node::selectAccType(currentDevice, caps);
        }
        plan.wgSize = 128u;
        return plan;
    }

    if (rtx5090 && workload.ci == 12u && workload.hi == 320u && workload.wi == 320u)
    {
        if (workload.wDtype == vai::Dtype::Fp32)
        {
            if (workload.co == 32u)
            {
                plan.wgM = 32u;
                plan.wgN = 64u;
                plan.wgK = 12u;
                plan.tM = 8u;
                plan.wgSize = 64u;
                plan.alignM = true;
                plan.alignN = true;
                plan.alignK = true;
                return plan;
            }
            else if (workload.co == 48u)
            {
                plan.wgM = 48u;
                plan.wgN = 64u;
                plan.wgK = 36u;
                plan.tM = 8u;
                plan.wgSize = 96u;
                plan.alignM = true;
                plan.alignN = true;
                plan.alignK = true;
                return plan;
            }
            else if (workload.co == 80u)
            {
                plan.wgM = 80u;
                plan.wgN = 32u;
                plan.wgK = 4u;
                plan.tM = 4u;
                plan.wgSize = 160u;
                plan.alignM = true;
                plan.alignN = true;
                plan.alignK = true;
                return plan;
            }
        }
        else if (workload.wDtype == vai::Dtype::Fp16)
        {
            if (workload.co == 32u)
            {
                plan.wgM = 32u;
                plan.wgN = 64u;
                plan.wgK = 36u;
                plan.tM = 8u;
                plan.wgSize = 64u;
                plan.alignM = true;
                plan.alignN = true;
                plan.alignK = true;
                return plan;
            }
            else if (workload.co == 48u)
            {
                plan.wgM = 48u;
                plan.wgN = 64u;
                plan.wgK = 27u;
                plan.tM = 6u;
                plan.wgSize = 128u;
                plan.alignM = true;
                plan.alignN = true;
                plan.alignK = true;
                return plan;
            }
            else if (workload.co == 64u)
            {
                plan.wgM = 64u;
                plan.wgN = 64u;
                plan.wgK = 36u;
                plan.tM = 8u;
                plan.wgSize = 128u;
                plan.alignM = true;
                plan.alignN = true;
                plan.alignK = true;
                return plan;
            }
            else if (workload.co == 80u)
            {
                plan.wgM = 80u;
                plan.wgN = 64u;
                plan.wgK = 36u;
                plan.tM = 8u;
                plan.wgSize = 160u;
                plan.alignM = true;
                plan.alignN = true;
                plan.alignK = true;
                return plan;
            }
        }
    }

    if (workload.hi == 320u && workload.wi == 320u && workload.co == 32u)
    {
        plan.wgSize = 64u;
        plan.wgM = 32u;
        plan.wgK = 4u;
    }
    else if (workload.hi == 320u && workload.wi == 320u && workload.co == 64u)
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

inline void Focus::preparePipeline(const vai::node::Workload& workload, const Plan& nextPlan)
{
    const vai::node::ShapeSpec shape = vai::node::ShapeSpec::fromWorkload(workload);
    if (gemm_.pipeline && plan_ == nextPlan && preparedShape_.matchShape(workload) &&
        preparedIDtype_ == workload.iDtype)
        return;

    eva::SpecializationConstant specialization{
        eva::constant_id<20>(nextPlan.batched),
        eva::constant_id<0>(nextPlan.wgSize),
        eva::constant_id<1>(nextPlan.wgM),
        eva::constant_id<2>(nextPlan.wgN),
    };
    vai::node::withShape(specialization, shape, true);
    vai::node::withFusion(specialization, uint32_t(vai::ActType::Silu), 1u, 1u);

    if (nextPlan.cooperativeMatrix)
    {
        specialization.addConstant(eva::constant_id<3>(nextPlan.panelK));
        specialization.addConstant(eva::constant_id<4>(nextPlan.sgM));
        specialization.addConstant(eva::constant_id<5>(nextPlan.sgN));
        specialization.addConstant(eva::constant_id<6>(false));
        specialization.addConstant(eva::constant_id<7>(nextPlan.sharedInputPatch));
        vai::node::withDTypes(
            specialization,
            workload.oDtype,
            workload.iDtype,
            workload.wDtype,
            vai::Dtype::Fp32,
            uint32_t(nextPlan.accType),
            { .includeAcc = true });
        gemm_ = vai::node::makePipelineBinding(pipelinePool().request(
            EXPERIMENTAL_YOLOX_SHADER_OUTPUT_DIR,
            "focus-cm",
            std::move(specialization),
            { .requiredSubgroupSize = 32u }));

        if (!transform_.pipeline)
        {
            transform_ = vai::node::makePipelineBinding(pipelinePool().request(
                EXPERIMENTAL_YOLOX_SHADER_OUTPUT_DIR, "focus-transform", {}, { .requiredSubgroupSize = 32u }));
        }
    }
    else
    {
        specialization.addConstant(eva::constant_id<3>(nextPlan.wgK));
        specialization.addConstant(eva::constant_id<4>(nextPlan.tM));
        specialization.addConstant(eva::constant_id<5>(nextPlan.alignM));
        specialization.addConstant(eva::constant_id<6>(nextPlan.alignN));
        specialization.addConstant(eva::constant_id<7>(nextPlan.alignK));
        vai::node::withDTypes(
            specialization,
            workload.oDtype,
            workload.iDtype,
            workload.wDtype,
            workload.wDtype,
            uint32_t(vai::Dtype::Fp32),
            { .includeWeight = true });
        gemm_ = vai::node::makePipelineBinding(
            pipelinePool().request(EXPERIMENTAL_YOLOX_SHADER_OUTPUT_DIR, "focus", std::move(specialization)));
    }

    plan_ = nextPlan;
    preparedShape_ = shape;
    preparedIDtype_ = workload.iDtype;
}

inline void Focus::prepare()
{
    const vai::node::Workload workload = makeFocusWorkload();
    const vai::Tensor& weight = (*this)["weight"];
    VAI_ASSERT(workload.wDtype == vai::Dtype::Fp32 || workload.wDtype == vai::Dtype::Fp16);
    VAI_ASSERT(workload.wDtype == vai::Dtype::Fp16 || workload.oDtype != vai::Dtype::Fp16);
    VAI_ASSERT(
        workload.wDtype == vai::Dtype::Fp16 ||
        workload.iDtype != vai::Dtype::Fp16);   // fp16 activations need fp16 weights
    VAI_ASSERT(weight.isShapeOf(workload.co, workload.ci, 3u, 3u));
    VAI_ASSERT((*this)["scale"].dtype() == workload.wDtype && (*this)["scale"].isShapeOf(workload.co));
    VAI_ASSERT((*this)["bias"].dtype() == workload.wDtype && (*this)["bias"].isShapeOf(workload.co));

    preparePipeline(workload, selectPlan(workload));
    auto shape = (*this)["in0"].shape();
    shape[shape.size() - 3] = workload.co;
    shape[shape.size() - 2] = workload.ho;
    shape.back() = workload.wo;
    (*this)["out0"] = vai::Tensor(shape, workload.oDtype);
}

inline void Focus::transformWeight(const vai::node::Workload& workload)
{
    if (!plan_.cooperativeMatrix || weightTransformed_)
        return;

    const uint32_t rawK = workload.ci * 9u;
    const uint32_t paddedK = ROUNDUP(rawK, 16u);
    const uint32_t paddedM = ROUNDUP(workload.co, 16u);
    auto transformCmd = device().newCommandBuffer(eva::queue_compute);
    transformCmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT);
    vai::node::ensureWeightTransform(
        device(),
        transformCmd,
        transform_,
        (*this)["weight"].buffer(),
        transformedWeight_,
        vai::node::makePushConstants(workload),
        size_t(paddedM) * paddedK * sizeof(uint16_t),
        paddedM * paddedK,
        weightTransformed_);
    VAI_ASSERT(weightTransformed_);
    transformCmd.end();
    device().queue() << transformCmd << eva::waiting;
}

inline void Focus::run(eva::CommandBuffer cmdBuff)
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
    vai::node::recordDispatch(
        cmdBuff,
        gemm_,
        vai::node::makePushConstants(workload),
        CEIL_DIV(workload.spatialSize(), plan_.wgN),
        CEIL_DIV(workload.co, plan_.wgM));
    cmdBuff.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / output.buffer() / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace model::experimental::yolox::node

#endif   // EXPERIMENTAL_YOLOX_NODES_FOCUS_H
