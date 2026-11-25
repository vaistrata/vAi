#ifndef VAI_NODES_INDEX_COPY_H
#define VAI_NODES_INDEX_COPY_H

#include "common.h"

namespace vai::node {

/*=============================================================================
  IndexCopy - write source slices at listed indices, as torch.index_copy
  Input:  in0  [d0, .., d(r-1)]
          in1  in0's shape with d(axis) replaced by N
  Output: out0 in0 with out0[.., index[n], ..] = in1[.., n, ..] along axis

  The index is N uint32 in a host-visible buffer the model owns and hands over
  with setIndex. It is read at dispatch time, so rewriting it after sync() needs
  no re-recording. An index past d(axis) leaves its slice unwritten.

  out0 aliases in0 and only the N slices are written (= index_copy_), so no
  other node may read in0 after this one. in1 may be left out of a recording,
  and out0 then aliases in0 with nothing recorded. in1 is converted to in0's
  dtype when the two differ.

  With accumulate, in1 is added instead, and the indices must be distinct.
=============================================================================*/
class IndexCopy : public Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;
    eva::Buffer index;
    uint32_t vecElems = 0;      // elements one thread moves: 4, 2 or 1
    uint32_t rowUnits = 0;      // a slice row (the axes behind axis) in vecElems units
    uint32_t count = 0;         // N
    uint32_t axisSize = 0;      // d(axis)
    uint32_t rows = 0;          // source rows: outer * N
    bool passthrough_ = false;

public:
    struct Config {
        uint32_t axis = 0;
        bool accumulate = false;   // out0[.., index[n], ..] += in1[.., n, ..], as torch.index_put_(accumulate=True)
    } config_;

    IndexCopy();
    IndexCopy(Config config);
    void setIndex(eva::Buffer buffer) { index = buffer; }
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;
};


inline IndexCopy::IndexCopy() : IndexCopy(Config()) {}

inline IndexCopy::IndexCopy(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0",  NodeSlot::input);
    addSlot("in1",  NodeSlot::input);
    flags_ |= partialInputs;
}

inline void IndexCopy::prepare()
{
    const Tensor& dst = (*this)["in0"];
    const Tensor& src = (*this)["in1"];
    VAI_ASSERT(dst.validShape());

    (*this)["out0"] = dst;

    passthrough_ = !src.validShape();
    if (passthrough_)
        return;

    VAI_ASSERT(index && "IndexCopy: setIndex() was never called");

    const auto& dstShape = dst.shape();
    const auto& srcShape = src.shape();
    const uint32_t axis = config_.axis;
    VAI_ASSERT(axis < dstShape.size() && srcShape.size() == dstShape.size());
    for (size_t i = 0; i < dstShape.size(); ++i)
        VAI_ASSERT(i == axis || srcShape[i] == dstShape[i]);

    count    = srcShape[axis];
    axisSize = dstShape[axis];
    rows     = outerSize(srcShape, srcShape.size() - axis - 1);
    VAI_ASSERT(index.size() >= uint64_t(count) * sizeof(uint32_t));

    const uint64_t rowElems = src.numElements() / rows;
    vecElems = rowElems % 4 == 0 ? 4 : rowElems % 2 == 0 ? 2 : 1;
    rowUnits = uint32_t(rowElems / vecElems);

    // clang-format off
    pipe = pipelinePool().request("index-copy", {
        eva::constant_id<0>(vecElems),                  // vecElems
        eva::constant_id<1>(config_.accumulate),        // accumulate
        eva::constant_id<SPEC_ID_TYPE_O>(dst.dtype()),  // typeO
        eva::constant_id<SPEC_ID_TYPE_I>(src.dtype())   // typeI
    });
    // clang-format on
    if (!descSet)
        descSet = gVaiDestSetPool(pipe.descSetLayout(0));
}

inline void IndexCopy::run(eva::CommandBuffer cmdBuff)
{
    if (passthrough_)
        return;

    auto outBuf = (*this)["out0"].buffer();
    auto srcBuf = (*this)["in1"].buffer();

    struct {
        uint32_t rowUnits, count, axisSize, rows;
    } constants = { rowUnits, count, axisSize, rows };

    descSet.write({ outBuf, srcBuf, index });
    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch2(rowUnits, rows)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace vai::node

#endif // VAI_NODES_INDEX_COPY_H
