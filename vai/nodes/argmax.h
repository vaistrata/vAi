#ifndef VAI_NODES_ARGMAX_H
#define VAI_NODES_ARGMAX_H

// Argmax - row-wise argmax (greedy decode) over a logits tensor.
#include "node-common.h"

namespace vai::node {


/*=============================================================================
  Argmax - Row-wise argmax / greedy sampling
  Input:  in0  [M, N] (fp32 logits)
  Output: out0 [M]    (one raw uint32 index per row, in 4-byte slots)

  Each row is reduced independently. out0[m] is the index of the largest value in
  in0[m, :]. Output elements are raw uint32 values, not fp32 numeric values.
=============================================================================*/
class Argmax : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
    } config_;

    Argmax();
    Argmax(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};


/////////////////////////////////////////////////////////////////////////////////////////
// Argmax
/////////////////////////////////////////////////////////////////////////////////////////
inline Argmax::Argmax() : Argmax(Config()) {}

inline Argmax::Argmax(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0",  NodeSlot::input);
}

inline void Argmax::prepare()
{
    auto& inTen = (*this)["in0"];
    VAI_ASSERT(inTen.validShape());
    VAI_ASSERT(inTen.shape().size() == 2);

    const Dtype iDtype = inTen.dtype();
    const PipeParams p = { .iDtype = iDtype };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        pipe = pipelinePool().request("argmax", {
            eva::constant_id<SPEC_ID_TYPE_I>(iDtype)   // typeI (logits)
        });
        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }

    uint32_t M = inTen.shape()[0];

    // [M] of raw uint32 indices (Dtype has no integer kind yet, so we reuse
    // fp32 as a generic 4-byte container).
    (*this)["out0"] = Tensor({M}, Dtype::Fp32);
}

inline void Argmax::run(eva::CommandBuffer cmdBuff)
{
    auto inBuf  = (*this)["in0"].buffer();
    auto outBuf = (*this)["out0"].buffer();

    // Single set: binding 0 = input logits (fp32), binding 1 = output indices.
    descSet.write({ inBuf, outBuf });

    uint32_t M = (*this)["in0"].shape()[0];
    uint32_t N = (*this)["in0"].shape()[1];

    struct {
        uint32_t M, N;
    } constants = { M, N };

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch(M)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}


}   // namespace vai::node

#endif // VAI_NODES_ARGMAX_H
