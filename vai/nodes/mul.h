#ifndef VAI_NODES_MUL_H
#define VAI_NODES_MUL_H

#include "vai/nodes/node-common.h"

namespace vai::node {


/*=============================================================================
  Mul - Element-wise multiplication
  Input: in0 [any shape], in1 [same shape]
  Output: out0 [same shape] = in0 * in1

  Inputs must have identical shapes and dtypes.
=============================================================================*/
class Mul : public vai::Node
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

    Mul();
    Mul(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};


inline Mul::Mul() : Mul(Config()) {}

inline Mul::Mul(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
    addSlot("in1", NodeSlot::input);

    config_.oDtype = (config_.oDtype == Dtype::None) ? oDefaultDtype() : config_.oDtype;
}

inline void Mul::prepare()
{
    const Tensor& in0Ten = (*this)["in0"];
    const Tensor& in1Ten = (*this)["in1"];

    const Dtype iDtype = in0Ten.dtype();

    const PipeParams p = { .iDtype = iDtype };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        pipe = pipelinePool().request("mul", {
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

inline void Mul::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto in0Buf = (*this)["in0"].buffer();
    auto in1Buf = (*this)["in1"].buffer();

    descSet.write({ outBuf, in0Buf, in1Buf });

    uint32_t N = (*this)["in0"].numElements();

    struct {
        uint32_t N;
    } constants = { N };

    cmdBuff
        .bindPipeline(pipe)
        .setPushConstants(0, sizeof(constants), &constants)
        .bindDescSets({descSet})
        .dispatch2(N)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace vai::node

#endif // VAI_NODES_MUL_H
