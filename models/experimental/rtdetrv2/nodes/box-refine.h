#ifndef EXPERIMENTAL_RTDETRV2_NODES_BOX_REFINE_H
#define EXPERIMENTAL_RTDETRV2_NODES_BOX_REFINE_H

#include "vai/nodes/common.h"

namespace model::experimental::rtdetrv2::node {

/*=============================================================================
  BoxRefine - Refine normalized boxes with predicted logit-space deltas
  Input:  boxes [Q, 4], delta [Q, 4] (fp32)
  Output: out0 [Q, 4] (fp32), sigmoid(delta + inverse_sigmoid(boxes))
=============================================================================*/
class BoxRefine : public vai::Node
{
    eva::ComputePipeline pipe_;
    eva::DescriptorSet descSet_;

public:
    BoxRefine();
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;
};

/////////////////////////////////////////////////////////////////////////////////////////
// BoxRefine
/////////////////////////////////////////////////////////////////////////////////////////
inline BoxRefine::BoxRefine()
{
    addSlot("boxes", vai::NodeSlot::input);
    addSlot("delta", vai::NodeSlot::input);
    addSlot("out0", vai::NodeSlot::output);
    pipe_ = pipelinePool().request(EXPERIMENTAL_RTDETRV2_SHADER_OUTPUT_DIR, "box-refine", {}, {});
    descSet_ = vai::gVaiDestSetPool(pipe_.descSetLayout(0));
}

inline void BoxRefine::prepare()
{
    const auto& boxes = (*this)["boxes"];
    const auto& delta = (*this)["delta"];
    VAI_ASSERT(
        boxes.dtype() == vai::Dtype::Fp32 && delta.dtype() == boxes.dtype() && delta.shape() == boxes.shape() &&
        boxes.isShapeOf(-1, 4));
    (*this)["out0"] = vai::Tensor(boxes.shape(), vai::Dtype::Fp32);
}

inline void BoxRefine::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    uint32_t numElements = (*this)["out0"].numElements();
    descSet_.write({ outBuf, (*this)["boxes"].buffer(), (*this)["delta"].buffer() });
    cmdBuff.bindPipeline(pipe_)
        .bindDescSets({ descSet_ })
        .setPushConstants(0, sizeof(numElements), &numElements)
        .dispatch2(numElements)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace model::experimental::rtdetrv2::node

#endif   // EXPERIMENTAL_RTDETRV2_NODES_BOX_REFINE_H
