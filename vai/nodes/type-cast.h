#ifndef VAI_NODES_TYPE_CAST_H
#define VAI_NODES_TYPE_CAST_H

// TypeCast - element-wise dtype cast.
#include "node-common.h"

namespace vai::node {


/*=============================================================================
  TypeCast - Casts each tensor element to another dtype.
  Input : in0  [any shape], any dtype
  Output: out0 [same shape] (oDtype)

  in0 already at oDtype makes out0 the input tensor itself: no pipeline, no
  dispatch.
=============================================================================*/
class TypeCast : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;
    bool skip = false;   // out0 is in0 itself -> nothing for run() to record

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        vai::Dtype oDtype;                      // required: the dtype to cast to
    } config_;

    TypeCast(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};


/////////////////////////////////////////////////////////////////////////////////////////
// TypeCast
/////////////////////////////////////////////////////////////////////////////////////////
inline TypeCast::TypeCast(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);

    VAI_ASSERT(config_.oDtype != Dtype::None);
}

inline void TypeCast::prepare()
{
    const Tensor& inTen = (*this)["in0"];
    VAI_ASSERT(inTen.validShape());

    skip = (config_.oDtype == inTen.dtype());
    if (skip)
    {
        (*this)["out0"] = inTen;
        return;
    }

    const Dtype iDtype = inTen.dtype();
    const PipeParams p = { .iDtype = iDtype };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        // clang-format off
        pipe = pipelinePool().request("cast", {
            eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),   // typeO
            eva::constant_id<SPEC_ID_TYPE_I>(iDtype)            // typeI
        });
        // clang-format on
        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }

    (*this)["out0"] = Tensor(inTen.shape(), config_.oDtype);
}

inline void TypeCast::run(eva::CommandBuffer cmdBuff)
{
    if (skip) return;

    auto outBuf = (*this)["out0"].buffer();
    auto inBuf  = (*this)["in0"].buffer();

    descSet.write({ outBuf, inBuf });

    uint32_t N = (*this)["in0"].numElements();

    struct {
        uint32_t N;
    } constants = { N };

    cmdBuff
        .bindPipeline(pipe)
        .setPushConstants(0, sizeof(constants), &constants)
        .bindDescSets({descSet})
        .dispatch2(N)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / (*this)["out0"].buffer() / eva::SYNC_SCOPE::COMPUTE_READ);
}


}   // namespace vai::node

#endif // VAI_NODES_TYPE_CAST_H
