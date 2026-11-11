#ifndef VAI_NODES_ACTIVATION_H
#define VAI_NODES_ACTIVATION_H

#include "node-common.h"

namespace vai::node {


/*=============================================================================
  Activation - Applies an element-wise activation.
  Input: in0 [any shape]
  Output: epilogue(in0); optionally in-place

  The inPlace option aliases the output to the input and requires matching input
  and output dtypes. Otherwise, the output uses separate storage.
=============================================================================*/
class Activation : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        vai::Epilogue epilogue = vai::Epilogue::None;
        bool inPlace = false;
    } config_;

    Activation(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};


inline Activation::Activation(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
}

inline void Activation::prepare()
{
    const Dtype oDtype = (oDtype_ == Dtype::None) ? oDefaultDtype() : oDtype_;
    const Tensor& inTen = (*this)["in0"];
    VAI_ASSERT(inTen.validShape());

    const Dtype iDtype = inTen.dtype();
    if (config_.inPlace)   // in-place reads and writes the same buffer
        VAI_ASSERT(oDtype == iDtype);

    const PipeParams p = { .iDtype = iDtype };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        pipe = pipelinePool().request("activation",
                                      {
                                          eva::constant_id<0>(config_.inPlace ? 1u : 0u),                  // inPlace
                                          eva::constant_id<SPEC_ID_TYPE_O>(oDtype),                        // typeO
                                          eva::constant_id<SPEC_ID_TYPE_I>(iDtype),                        // typeI
                                          eva::constant_id<SPEC_ID_EPILOGUE>((uint32_t)config_.epilogue)   // epilogue
                                      });
        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }

    (*this)["out0"] = config_.inPlace ? inTen : Tensor(inTen.shape(), oDtype);
}

inline void Activation::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf  = config_.inPlace ? eva::Buffer() : (*this)["in0"].buffer();

    descSet.write({ outBuf, inBuf });

    uint32_t N = (*this)["out0"].numElements();

    struct {
        uint32_t N;
    } constants = { N };

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch2(N)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace vai::node

#endif // VAI_NODES_ACTIVATION_H
