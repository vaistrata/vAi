#ifndef VAI_NODES_SLICE_H
#define VAI_NODES_SLICE_H

#include "node-common.h"

namespace vai::node {

/*=============================================================================
  Slice - Take a run of consecutive indices along one axis
  Input:  in0  [d0, .., d(r-1)]   (rank >= 2)
  Output: out0 the same shape with d(axis) replaced by count

  A negative start counts from the end, with -1 selecting the last index.

  What the axes ahead of the cut multiply to decides the cost, not the axis number.
  At 1 the run is one contiguous block, so out0 aliases it when the offset satisfies
  device alignment and is one buffer copy otherwise — axis 0 is always this case, and
  so is a deeper axis with nothing but size-1 axes in front. Past 1 the run is strided,
  count indices every d(axis), and a gather kernel reads it out. The shape decides the
  path; the caller writes the same node either way.
=============================================================================*/
class Slice : public Node
{
    eva::ComputePipeline pipe;      // strided path only (axis > 0, outer > 1)
    eva::DescriptorSet descSet;
    uint32_t startIdx = 0;  // resolved (non-negative) start, set in prepare()
    uint32_t outer = 0;     // elements ahead of the sliced axis, flattened
    uint32_t inner = 0;     // elements behind it, flattened
    uint32_t axisSize = 0;  // the sliced axis as the input has it
    bool aliased = false;   // out0 views in0 -> nothing for run() to record

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        uint32_t axis = 0;      // the axis to cut; 0 takes a contiguous block
        int64_t  start;         // first index on the axis; negative counts from the end
        uint32_t count = 1;     // how many consecutive indices to take
    } config_;

    Slice(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};


inline Slice::Slice(Config config)
: config_(config)
{
    VAI_ASSERT(config_.count >= 1);
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
}

inline void Slice::prepare()
{
    const Dtype oDtype = (oDtype_ == Dtype::None) ? oDefaultDtype() : oDtype_;
    const Tensor& inTen = (*this)["in0"];
    VAI_ASSERT(inTen.validShape());

    const Dtype iDtype = inTen.dtype();

    const std::vector<uint32_t>& shape = inTen.shape();
    VAI_ASSERT(shape.size() >= 2 && config_.axis < shape.size());
    axisSize = shape[config_.axis];

    int64_t start = config_.start;
    if (start < 0)
        start += axisSize;
    VAI_ASSERT(start >= 0 && uint64_t(start) + config_.count <= axisSize);
    startIdx = uint32_t(start);

    std::vector<uint32_t> outShape = shape;
    outShape[config_.axis] = config_.count;

    outer = outerSize(shape, shape.size() - config_.axis);
    inner = 1;
    for (size_t i = size_t(config_.axis) + 1; i < shape.size(); ++i)
        inner *= shape[i];

    // Only a run with whole blocks ahead of it is strided; outer == 1 leaves the
    // selected indices contiguous, whatever axis they were cut from.
    if (outer > 1)
    {
        const PipeParams p = { .iDtype = iDtype };
        if (!pipe || p != pipeParams)
        {
            pipeParams = p;
            // clang-format off
            pipe = pipelinePool().request("slice", {
                eva::constant_id<SPEC_ID_TYPE_O>(oDtype),   // typeO
                eva::constant_id<SPEC_ID_TYPE_I>(iDtype),           // typeI
            });
            // clang-format on
            if (!descSet)
                descSet = gVaiDestSetPool(pipe.descSetLayout(0));
        }

        (*this)["out0"] = Tensor(outShape, oDtype);
        aliased = false;
        return;
    }

    // Bytes move as they are on this path, so a declared cast has no kernel to
    // happen in.
    VAI_ASSERT(oDtype == iDtype);

    Tensor sliced = inTen.view(outShape, inTen.baseOffset() + startIdx * inner);

    aliased = (sliced.byteOffset() % device().minStorageBufferOffsetAlignment() == 0);
    if (aliased)
        (*this)["out0"] = std::move(sliced);
    else
        (*this)["out0"] = Tensor(outShape, inTen.dtype());
}

inline void Slice::run(eva::CommandBuffer cmdBuff)
{
    if (aliased) return;

    const Tensor& inTen = (*this)["in0"];
    auto inBuf  = inTen.buffer();
    auto outBuf = (*this)["out0"].buffer();

    if (outer > 1)
    {
        struct {
            uint32_t inner, count, axisSize, start, total;
        } constants = {
            inner, config_.count, axisSize, startIdx, outer * config_.count * inner,
        };

        descSet.write({ outBuf, inBuf });
        cmdBuff
            .bindPipeline(pipe)
            .bindDescSets({descSet})
            .setPushConstants(0, sizeof(constants), &constants)
            .dispatch2(constants.total)
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
        return;
    }

    // outer == 1, so one index along the axis is exactly the trailing block.
    uint64_t rowBytes   = inTen.numBytes() / axisSize;
    uint64_t offset     = uint64_t(startIdx) * rowBytes;
    uint64_t sliceBytes = uint64_t(config_.count) * rowBytes;

    cmdBuff
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / inBuf / eva::SYNC_SCOPE::TRANSFER_SRC)
        .copyBuffer(inBuf(offset, sliceBytes), outBuf)
        .barrier(eva::SYNC_SCOPE::TRANSFER_DST / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace vai::node

#endif // VAI_NODES_SLICE_H
