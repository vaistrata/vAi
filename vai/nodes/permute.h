#ifndef VAI_NODES_PERMUTE_H
#define VAI_NODES_PERMUTE_H

#include "node-common.h"

namespace vai::node {


/*=============================================================================
  Permute - Permutes the axes of a tensor of any rank.
  Input: in0 [D0, .., Dr-1], r = perms.size()
  Output: axes reordered by perms; reshape-only fast path when layout is unchanged

  Output dimension i is input dimension perms[i], and perms must be a permutation of
  {0, .., r-1} with r at most PERMUTE_MAX_RANK. When the permutation preserves memory
  order and the dtype is unchanged, the output aliases the input; otherwise it uses new
  contiguous storage.

  The rank is a specialization constant, so the pool holds one pipeline per rank.
=============================================================================*/
class Permute : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;
    std::vector<uint32_t> perms_;
    bool skip = false;

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config
    {
    } config_;

    Permute(std::vector<uint32_t> perms, Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};


inline Permute::Permute(std::vector<uint32_t> perms, Config config = Config())
: perms_(std::move(perms))
, config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);

    VAI_ASSERT(!perms_.empty() && perms_.size() <= PERMUTE_MAX_RANK);
}

inline void Permute::prepare()
{
    const Dtype oDtype = (oDtype_ == Dtype::None) ? oDefaultDtype() : oDtype_;
    const Tensor& inTen = (*this)["in0"];

    const Dtype iDtype = inTen.dtype();

    const PipeParams p = { .iDtype = iDtype };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        pipe = pipelinePool().request("permute", {
                                                     eva::constant_id<SPEC_ID_TYPE_O>(oDtype),      // typeO
                                                     eva::constant_id<SPEC_ID_TYPE_I>(iDtype),      // typeI
                                                     eva::constant_id<0>((uint32_t)perms_.size())   // rank
                                                 });
        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }
    VAI_ASSERT(inTen.shape().size() == perms_.size());

    const auto& D = inTen.shape();
    const uint32_t rank = (uint32_t) perms_.size();

    // Size-1 axes carry no offset, so only the rest fix the layout: a reshape reproduces
    // the permutation exactly when those axes keep their relative order.
    bool canSkip = true;
    bool seen = false;
    uint32_t prevAxis = 0;
    for (uint32_t i = 0; i < rank && canSkip; ++i)
    {
        uint32_t axis = perms_[i];
        if (D[axis] == 1)
            continue;
        if (seen && axis < prevAxis)
            canSkip = false;
        prevAxis = axis;
        seen = true;
    }

    std::vector<uint32_t> outShape(rank);
    for (uint32_t i = 0; i < rank; ++i)
        outShape[i] = D[perms_[i]];

    // Pass-through requires the dtype to stay unchanged.
    if (canSkip && iDtype == oDtype)
    {
        (*this)["out0"] = inTen.reshape(outShape);
        skip = true;
    }
    else
    {
        (*this)["out0"] = Tensor(outShape, oDtype);
        skip = false;
    }
}

inline void Permute::run(eva::CommandBuffer cmdBuff)
{
    if (skip)
        return;

    auto outBuf = (*this)["out0"].buffer();
    auto inBuf  = (*this)["in0"].buffer();

    descSet.write({ outBuf, inBuf });

    const auto& D = (*this)["in0"].shape();
    const uint32_t rank = (uint32_t) perms_.size();

    uint32_t inStride[PERMUTE_MAX_RANK];        // input stride per input dim
    inStride[rank - 1] = 1;
    for (uint32_t i = rank - 1; i-- > 0; )
        inStride[i] = inStride[i + 1] * D[i + 1];

    // output dims (contiguous) and the input stride each output axis contributes
    struct {
        uint32_t E[PERMUTE_MAX_RANK];
        uint32_t IS[PERMUTE_MAX_RANK];
        uint32_t total;
    } constants = {};

    uint32_t total = 1;
    for (uint32_t i = 0; i < rank; ++i)
    {
        constants.E[i]  = D[perms_[i]];
        constants.IS[i] = inStride[perms_[i]];
        total *= constants.E[i];
    }
    constants.total = total;

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch2(total)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace vai::node

#endif // VAI_NODES_PERMUTE_H
