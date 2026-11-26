#ifndef VAI_NODES_PERMUTE_H
#define VAI_NODES_PERMUTE_H

#include "common.h"

namespace vai::node {


/*=============================================================================
  Permute - Permutes the axes of a tensor of any rank.
  Input: in0 [..., D0, .., Dk-1], k = perms.size()
  Output: [..., the last k axes reordered by perms]; reshape-only fast path when
          layout is unchanged

  perms reorders the last k axes of in0 and the axes ahead of them pass through, so a
  leading batch axis may be there or not. Among the last k, output dimension i is input
  dimension perms[i], and perms must be a permutation of {0, .., k-1}. The leading axes
  fold into one, so k (plus one when there are leading axes) is at most
  PERMUTE_MAX_RANK. When the permutation preserves memory order and the dtype is
  unchanged, the output aliases the input; otherwise it uses new contiguous storage.

  The rank is a specialization constant, so the pool holds one pipeline per rank.
=============================================================================*/
class Permute : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;
    std::vector<uint32_t> perms_;
    std::vector<uint32_t> inDims_;   // in0 as the kernel sees it: size-1 axes dropped, runs merged
    std::vector<uint32_t> axes_;     // perms over inDims_, the folded axis kept first
    bool skip = false;

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;
        uint32_t rank = 0;

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

    // The axes ahead of the last k pass through, folded into one outer axis that
    // stays first.
    const auto& shape = inTen.shape();
    const size_t k = perms_.size();
    VAI_ASSERT(shape.size() >= k);
    const size_t lead = shape.size() - k;

    std::vector<uint32_t> dims, order;
    if (lead > 0)
    {
        uint32_t outer = 1;
        for (size_t i = 0; i < lead; ++i)
            outer *= shape[i];
        dims.push_back(outer);
        order.push_back(0);
    }
    const uint32_t first = lead > 0 ? 1u : 0u;
    for (size_t i = 0; i < k; ++i)
    {
        dims.push_back(shape[lead + i]);
        order.push_back(first + perms_[i]);
    }

    // Size-1 axes carry no offset, so the kernel sees only the rest; the output keeps
    // its full shape.
    std::vector<uint32_t> kept(dims.size());   // input axis -> its index among the kept
    inDims_.clear();
    for (size_t i = 0; i < dims.size(); ++i)
    {
        kept[i] = (uint32_t)inDims_.size();
        if (dims[i] > 1)
            inDims_.push_back(dims[i]);
    }
    axes_.clear();
    for (uint32_t axis : order)
        if (dims[axis] > 1)
            axes_.push_back(kept[axis]);
    if (axes_.empty())   // a single element
    {
        inDims_.push_back(1);
        axes_.push_back(0);
    }

    // Axes adjacent in in0 that stay adjacent, in order, in out0 move as one axis.
    std::vector<uint32_t> runFirst, runLast;   // in0 axes of each run, in out0 order
    for (uint32_t axis : axes_)
    {
        if (!runLast.empty() && axis == runLast.back() + 1)
            runLast.back() = axis;
        else
        {
            runFirst.push_back(axis);
            runLast.push_back(axis);
        }
    }
    std::vector<int> runAt(inDims_.size(), -1);   // in0 axis -> the run it starts
    for (size_t r = 0; r < runFirst.size(); ++r)
        runAt[runFirst[r]] = (int)r;
    std::vector<uint32_t> mergedDims, runIndex(runFirst.size());
    for (uint32_t axis = 0; axis < inDims_.size(); ++axis)
    {
        if (runAt[axis] < 0)
            continue;
        const uint32_t r = (uint32_t)runAt[axis];
        uint32_t size = 1;
        for (uint32_t a = runFirst[r]; a <= runLast[r]; ++a)
            size *= inDims_[a];
        runIndex[r] = (uint32_t)mergedDims.size();
        mergedDims.push_back(size);
    }
    inDims_ = std::move(mergedDims);
    axes_ = std::move(runIndex);

    const uint32_t rank = (uint32_t)axes_.size();
    VAI_ASSERT(rank <= PERMUTE_MAX_RANK);

    const PipeParams p = { .iDtype = iDtype, .rank = rank };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        // clang-format off
        pipe = pipelinePool().request("permute", {
            eva::constant_id<SPEC_ID_TYPE_O>(oDtype),          // typeO
            eva::constant_id<SPEC_ID_TYPE_I>(iDtype),                  // typeI
            eva::constant_id<0>(rank)                                  // rank
        });
        // clang-format on
        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }

    // With the size-1 axes gone and the runs merged, a reshape reproduces the
    // permutation exactly when the rest keep their relative order.
    bool canSkip = true;
    for (uint32_t i = 1; i < rank && canSkip; ++i)
        if (axes_[i] < axes_[i - 1])
            canSkip = false;

    std::vector<uint32_t> outShape(shape.begin(), shape.begin() + lead);
    for (size_t i = 0; i < k; ++i)
        outShape.push_back(shape[lead + perms_[i]]);

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

    const auto& D = inDims_;
    const uint32_t rank = (uint32_t)axes_.size();

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
        constants.E[i] = D[axes_[i]];
        constants.IS[i] = inStride[axes_[i]];
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
