#ifndef VAI_NODES_ROW_SLICE_SCATTER_H
#define VAI_NODES_ROW_SLICE_SCATTER_H

#include "node-common.h"
#include <utility>
#include <vector>

namespace vai::node {

/*=============================================================================
  RowSliceScatter - write slices into rows, as torch.slice_scatter on dim 0
  Input:  in0  [T, D]
          in1  [B, S, D] or [S, D]   (S == sliceSize; B slices, one when rank 2)
  Output: out0 [T, D], in0 with rows [rowStarts[b], rowStarts[b] + S) taken from slice b

  in1 may be left out of a recording, and out0 then aliases in0 with nothing recorded.
  Rows move as bytes, so in0 and in1 share a dtype. The row starts are host state
  that the recorded copies bake in: setRowStarts re-records only when they change.
=============================================================================*/
class RowSliceScatter : public Node
{
    std::vector<uint32_t> rowStarts_ = {0};
    bool passthrough_ = false;

public:
    struct Config {
        uint32_t sliceSize;     // S: rows in one slice; 0 when in1 is never fed
    } config_;

    RowSliceScatter(Config config);
    void setRowStarts(std::vector<uint32_t> rows);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;
};


inline RowSliceScatter::RowSliceScatter(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0",  NodeSlot::input);
    addSlot("in1",  NodeSlot::input);
    flags_ |= partialInputs;
}

inline void RowSliceScatter::setRowStarts(std::vector<uint32_t> rows)
{
    if (rows == rowStarts_)
        return;
    rowStarts_ = std::move(rows);
    ownedNet_->invalidateCommandBuffer();
}

inline void RowSliceScatter::prepare()
{
    const Tensor& dst = (*this)["in0"];
    const Tensor& src = (*this)["in1"];
    VAI_ASSERT(dst.validShape());

    passthrough_ = !src.validShape();
    if (passthrough_)
    {
        (*this)["out0"] = dst;
        return;
    }

    VAI_ASSERT(dst.dtype() == src.dtype());

    const auto& dstShape = dst.shape();
    const auto& srcShape = src.shape();
    VAI_ASSERT(dstShape.size() == 2);
    VAI_ASSERT(srcShape.size() == 2 || srcShape.size() == 3);
    VAI_ASSERT(srcShape.back() == dstShape[1]);

    const uint32_t S = srcShape[srcShape.size() - 2];
    const uint32_t B = srcShape.size() == 3 ? srcShape[0] : 1;
    VAI_ASSERT(config_.sliceSize == S);
    VAI_ASSERT(rowStarts_.size() == B);
    for (uint32_t start : rowStarts_)
        VAI_ASSERT(start + S <= dstShape[0]);

    (*this)["out0"] = Tensor(dstShape, dst.dtype());
}

inline void RowSliceScatter::run(eva::CommandBuffer cmdBuff)
{
    if (passthrough_)
        return;

    const Tensor& dst = (*this)["in0"];
    const Tensor& src = (*this)["in1"];
    auto dstBuf = dst.buffer();
    auto srcBuf = src.buffer();
    auto outBuf = (*this)["out0"].buffer();

    const uint64_t rowBytes   = dst.numBytes() / dst.shape()[0];
    const uint64_t sliceBytes = uint64_t(config_.sliceSize) * rowBytes;

    // The slices land after the whole of in0, so the second batch of copies waits
    // on the first to finish writing out0.
    cmdBuff
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / dstBuf / eva::SYNC_SCOPE::TRANSFER_SRC)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / srcBuf / eva::SYNC_SCOPE::TRANSFER_SRC)
        .copyBuffer(dstBuf, outBuf)
        .barrier(eva::SYNC_SCOPE::TRANSFER_DST / outBuf / eva::SYNC_SCOPE::TRANSFER_DST);

    for (size_t b = 0; b < rowStarts_.size(); ++b)
        cmdBuff.copyBuffer(srcBuf(b * sliceBytes, sliceBytes),
                           outBuf(rowStarts_[b] * rowBytes, sliceBytes));

    cmdBuff.barrier(eva::SYNC_SCOPE::TRANSFER_DST / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace vai::node

#endif // VAI_NODES_ROW_SLICE_SCATTER_H
