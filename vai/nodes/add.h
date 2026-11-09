#ifndef VAI_NODES_ADD_H
#define VAI_NODES_ADD_H

#include "vai/nodes/node-common.h"

namespace vai::node {


/*=============================================================================
  Add - element-wise addition for residual connections
  Input: in0, in1 [same shape]
  Output: in0 + in1

  Inputs must have identical shapes and dtypes.
=============================================================================*/
class Add : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        vai::Dtype oDtype = vai::Dtype::None;   // None: oDefaultDtype
    } config_;

    Add();
    Add(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};


inline Add::Add() : Add(Config()) {}

inline Add::Add(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
    addSlot("in1", NodeSlot::input);

    config_.oDtype = (config_.oDtype == Dtype::None) ? oDefaultDtype() : config_.oDtype;
}

inline void Add::prepare()
{
    const Tensor& in0Ten = (*this)["in0"];
    const Tensor& in1Ten = (*this)["in1"];

    const Dtype iDtype = in0Ten.dtype();

    const PipeParams p = { .iDtype = iDtype };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        pipe = pipelinePool().request("add", {
            eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),   // typeO
            eva::constant_id<SPEC_ID_TYPE_I>(iDtype)            // typeI (in0 & in1)
        });
        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }

    VAI_ASSERT(in1Ten.dtype() == in0Ten.dtype());
    VAI_ASSERT(in1Ten.shape() == in0Ten.shape());

    (*this)["out0"] = Tensor(in0Ten.shape(), config_.oDtype);
}

inline void Add::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto in0Buf = (*this)["in0"].buffer();
    auto in1Buf = (*this)["in1"].buffer();

    descSet.write({ outBuf, in0Buf, in1Buf });

    uint32_t N = (*this)["in0"].numElements();

    cmdBuff
        .bindPipeline(pipe)
        .setPushConstants(0, sizeof(N), &N)
        .bindDescSets({descSet})
        .dispatch2(N)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace vai::node

#endif // VAI_NODES_ADD_H
