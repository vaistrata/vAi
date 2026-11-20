#ifndef EXPERIMENTAL_RTDETRV2_NODES_MUL_H
#define EXPERIMENTAL_RTDETRV2_NODES_MUL_H

#include "vai/nodes/common.h"
#include <algorithm>
#include <cstring>
#include <vector>

namespace model::experimental::rtdetrv2::node {


/////////////////////////////////////////////////////////////////////////////////////////
// Mul - Element-wise multiplication
// Input: in0 / in1 with broadcast-compatible, contiguous shapes.
// Output: out0 [broadcast shape] = in0 * in1
//
// Align axes from the right: dimensions must match or one must be 1.
// Inputs share a dtype. Like vai::Tensor, this node requires nonempty, positive
// shapes; use [1] for a scalar. No input data is replicated for broadcasting.
// With constantIn1, assign in1 directly before the first inference instead of
// connecting a producer. It is uploaded and retained as a persistent tensor.
/////////////////////////////////////////////////////////////////////////////////////////
class Mul : public vai::Node
{
    eva::ComputePipeline pipe_;
    eva::DescriptorSet descSet_;
    uint32_t broadcastRank_ = 0;   // 0 selects the same-shape direct indexing path.
    std::vector<uint32_t> cachedStrides_;
    eva::Buffer strideBuffer_;

public:
    struct Config
    {
        bool constantIn1 = false;   // in1 is a stored tensor instead of a graph input
    } config_;

    Mul(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;
};


inline Mul::Mul(Config config)
: config_(config)
{
    addSlot("out0", vai::NodeSlot::output);
    addSlot("in0", vai::NodeSlot::input);
    addSlot("in1", config_.constantIn1 ? vai::NodeSlot::persistent : vai::NodeSlot::input);
}

inline void Mul::prepare()
{
    const vai::Tensor& in0Ten = (*this)["in0"];
    const vai::Tensor& in1Ten = (*this)["in1"];

    VAI_ASSERT(in1Ten.dtype() == in0Ten.dtype());
    VAI_ASSERT(in0Ten.validShape() && in1Ten.validShape());

    const vai::Dtype oDtype = oDtype_ == vai::Dtype::None ? oDefaultDtype() : oDtype_;
    pipe_ = pipelinePool().request(
        EXPERIMENTAL_RTDETRV2_SHADER_OUTPUT_DIR,
        "mul",
        { eva::constant_id<SPEC_ID_TYPE_O>(oDtype), eva::constant_id<SPEC_ID_TYPE_I>(in0Ten.dtype()) });
    if (!descSet_)
        descSet_ = vai::gVaiDestSetPool(pipe_.descSetLayout(0));

    const auto& in0Shape = in0Ten.shape();
    const auto& in1Shape = in1Ten.shape();
    const size_t rank = std::max(in0Shape.size(), in1Shape.size());
    const size_t in0LeadingDims = rank - in0Shape.size();
    const size_t in1LeadingDims = rank - in1Shape.size();
    std::vector<uint32_t> outShape(rank);
    // Three uint words per axis: output stride, in0 stride, in1 stride.
    std::vector<uint32_t> strideData(3 * rank);
    uint64_t outStride = 1, in0Stride = 1, in1Stride = 1;
    for (size_t axis = rank; axis-- > 0;)
    {
        const uint32_t in0Dim = axis < in0LeadingDims ? 1 : in0Shape[axis - in0LeadingDims];
        const uint32_t in1Dim = axis < in1LeadingDims ? 1 : in1Shape[axis - in1LeadingDims];
        VAI_ASSERT(in0Dim == in1Dim || in0Dim == 1 || in1Dim == 1);
        outShape[axis] = std::max(in0Dim, in1Dim);
        strideData[3 * axis] = uint32_t(outStride);
        strideData[3 * axis + 1] = in0Dim == 1 ? 0 : uint32_t(in0Stride);
        strideData[3 * axis + 2] = in1Dim == 1 ? 0 : uint32_t(in1Stride);
        outStride *= outShape[axis];
        in0Stride *= in0Dim;
        in1Stride *= in1Dim;
    }

    broadcastRank_ = (in0Shape == in1Shape) ? 0 : uint32_t(rank);
    if (broadcastRank_ && strideData != cachedStrides_)
    {
        // Shape changes re-record commands without re-uploading persistent
        // tensors. Keep metadata in a coherent host-written storage buffer.
        const size_t bytes = strideData.size() * sizeof(uint32_t);
        if (!strideBuffer_ || strideBuffer_.size() < bytes)
        {
            if (strideBuffer_)
                strideBuffer_.destroy();
            strideBuffer_ = device().createBuffer(
                { .size = bytes,
                  .usage = eva::BUFFER_USAGE::STORAGE_BUFFER,
                  .reqMemProps = eva::MEMORY_PROPERTY::HOST_VISIBLE | eva::MEMORY_PROPERTY::HOST_COHERENT });
        }
        std::memcpy(strideBuffer_.map(), strideData.data(), bytes);
        cachedStrides_ = std::move(strideData);
    }

    (*this)["out0"] = vai::Tensor(outShape, oDtype);
}

inline void Mul::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto in0Buf = (*this)["in0"].buffer();
    auto in1Buf = (*this)["in1"].buffer();

    // The same-shape path never reads binding 3; bind a valid buffer anyway.
    auto strideBuf = broadcastRank_ ? strideBuffer_() : outBuf;
    descSet_.write({ outBuf, in0Buf, in1Buf, strideBuf });

    uint32_t N = uint32_t((*this)["out0"].numElements());

    struct
    {
        uint32_t N;
        uint32_t rank;
    } constants = { N, broadcastRank_ };

    cmdBuff.bindPipeline(pipe_)
        .setPushConstants(0, sizeof(constants), &constants)
        .bindDescSets({ descSet_ })
        .dispatch2(N)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace model::experimental::rtdetrv2::node

#endif   // EXPERIMENTAL_RTDETRV2_NODES_MUL_H
