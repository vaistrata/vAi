#ifndef EXPERIMENTAL_RTDETRV2_NODES_ADD_H
#define EXPERIMENTAL_RTDETRV2_NODES_ADD_H

#include "vai/nodes/common.h"

namespace model::experimental::rtdetrv2::node {


/////////////////////////////////////////////////////////////////////////////////////////
// Add - element-wise addition for residual connections
// Input: in0, in1 [same shape]
// Output: in0 + in1
//
// Inputs must have identical shapes and dtypes.
// With constantIn1, assign in1 before inference; it is uploaded and retained
// as a persistent tensor instead of receiving a graph connection.
/////////////////////////////////////////////////////////////////////////////////////////
class Add : public vai::Node
{
    eva::ComputePipeline pipe_;
    eva::DescriptorSet descSet_;

public:
    struct Config
    {
        bool constantIn1 = false;
    } config_;

    Add();
    Add(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;
};


inline Add::Add()
: Add(Config())
{
}

inline Add::Add(Config config)
: config_(config)
{
    addSlot("out0", vai::NodeSlot::output);
    addSlot("in0", vai::NodeSlot::input);
    addSlot("in1", config_.constantIn1 ? vai::NodeSlot::persistent : vai::NodeSlot::input);
}

inline void Add::prepare()
{
    const vai::Tensor& in0Ten = (*this)["in0"];
    const vai::Tensor& in1Ten = (*this)["in1"];

    const vai::Dtype oDtype = oDtype_ == vai::Dtype::None ? oDefaultDtype() : oDtype_;
    pipe_ = pipelinePool().request(
        "add",
        {
            eva::constant_id<SPEC_ID_TYPE_O>(oDtype),          // typeO
            eva::constant_id<SPEC_ID_TYPE_I>(in0Ten.dtype())   // typeI (in0 & in1)
        });
    if (!descSet_)
        descSet_ = vai::gVaiDestSetPool(pipe_.descSetLayout(0));
    VAI_ASSERT(in1Ten.dtype() == in0Ten.dtype());
    VAI_ASSERT(in1Ten.shape() == in0Ten.shape());

    (*this)["out0"] = vai::Tensor(in0Ten.shape(), oDtype);
}

inline void Add::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto in0Buf = (*this)["in0"].buffer();
    auto in1Buf = (*this)["in1"].buffer();

    descSet_.write({ outBuf, in0Buf, in1Buf });

    uint32_t N = (*this)["in0"].numElements();

    cmdBuff.bindPipeline(pipe_)
        .setPushConstants(0, sizeof(N), &N)
        .bindDescSets({ descSet_ })
        .dispatch2(N)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace model::experimental::rtdetrv2::node

#endif   // EXPERIMENTAL_RTDETRV2_NODES_ADD_H
