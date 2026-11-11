#ifndef VAI_NODES_CONCAT_H
#define VAI_NODES_CONCAT_H

#include "node-common.h"

namespace vai::node {


/*=============================================================================
  Concat - join inputs along one axis
  Input: in0 .. inN [same shape except along the concat axis]
  Output: out0 the inputs joined along axis, in input order

  All inputs must have identical dtypes, ranks, and dimensions except along axis. A
  negative axis counts from the end.

  What the axes ahead of the cut multiply to decides the cost, not the axis number. At
  1 each input is one contiguous run of the output, so buffer copies place them and no
  shader runs — axis 0 is always this case, and so is a deeper axis with nothing but
  size-1 axes in front. One input arriving is then the whole output and out0 aliases it.
  Past 1 the runs interleave and a gather kernel reads them, which is where the 8-input
  limit and the dtype conversion live.

  allowPartialInputs lets a recording bring only some of them; the rest are not joined.
  Off, every input is required, so a recording that cannot supply one drops this node.
=============================================================================*/
class Concat : public vai::Node
{
    static constexpr uint32_t maxInputs = 8; // limited by hardcoded layout in shader
    eva::ComputePipeline pipe;               // gather path only
    eva::DescriptorSet descSet;
    uint32_t axisIdx = 0;          // resolved (non-negative) axis, set in prepare()
    bool contiguous = false;       // each input is one run of the output: copy, no kernel
    bool skip = false;             // out0 views the one input that arrived
    std::vector<uint32_t> arrived; // the inputs this recording brought, in input order

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        int axis = 0;          // negative counts from the end
        uint32_t inputCount = 2;

        // A recording may leave some inputs out — the caller names the ones it
        // brings, and the rest are not joined. Off, every input is required, so
        // a recording that cannot supply one drops this node instead.
        bool allowPartialInputs = false;

    } config_;

    Concat();
    Concat(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};


inline Concat::Concat() : Concat(Config()) {}

inline Concat::Concat(Config config)
: config_(config)
{
    VAI_ASSERT(config_.inputCount >= 2);
    addSlot("out0", NodeSlot::output);
    for (uint32_t i = 0; i < config_.inputCount; ++i)
        addSlot("in" + std::to_string(i), NodeSlot::input);

    if (config_.allowPartialInputs)
        flags_ |= partialInputs;
}

inline void Concat::prepare()
{
    const Dtype oDtype = (oDtype_ == Dtype::None) ? oDefaultDtype() : oDtype_;
    const Tensor* first = nullptr;
    uint32_t concatDim = 0;

    arrived.clear();
    for (uint32_t i = 0; i < config_.inputCount; ++i)
    {
        const Tensor& in = (*this)["in" + std::to_string(i)];
        if (!in.validShape())
            continue;
        if (!first)
        {
            first = &in;
            int axis = config_.axis;
            if (axis < 0)
                axis += int(in.shape().size());
            VAI_ASSERT(axis >= 0 && axis < int(in.shape().size()));
            axisIdx = uint32_t(axis);
        }

        VAI_ASSERT(in.dtype() == first->dtype());
        VAI_ASSERT(in.shape().size() == first->shape().size());
        for (size_t d = 0; d < in.shape().size(); ++d)
            VAI_ASSERT(d == axisIdx || in.shape()[d] == first->shape()[d]);

        concatDim += in.shape()[axisIdx];
        arrived.push_back(i);
    }
    VAI_ASSERT(first);

    const Dtype iDtype = first->dtype();

    // Nothing ahead of the axis means nothing to step over: each input lands as one
    // run. A dtype change has to pass through the kernel whatever the shape.
    uint32_t outer = 1;
    for (uint32_t d = 0; d < axisIdx; ++d)
        outer *= first->shape()[d];
    contiguous = (outer == 1 && iDtype == oDtype);

    std::vector<uint32_t> outShape = first->shape();
    outShape[axisIdx] = concatDim;

    skip = contiguous && arrived.size() == 1;
    if (skip)
        (*this)["out0"] = *first;
    else
        (*this)["out0"] = Tensor(outShape, oDtype);

    if (contiguous)
        return;

    // The gather path reads its inputs through the shader's fixed bindings.
    VAI_ASSERT(arrived.size() <= maxInputs);
    const PipeParams p = { .iDtype = iDtype };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        pipe = pipelinePool().request("concat", {
                                                    eva::constant_id<SPEC_ID_TYPE_O>(oDtype),   // typeO
                                                    eva::constant_id<SPEC_ID_TYPE_I>(iDtype)    // typeI (all inputs)
                                                });
        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }
}

inline void Concat::run(eva::CommandBuffer cmdBuff)
{
    if (skip)
        return;

    auto outBuf = (*this)["out0"].buffer();

    if (contiguous)
    {
        uint64_t dstOffset = 0;
        for (uint32_t i : arrived)
        {
            const Tensor& in = (*this)["in" + std::to_string(i)];
            uint64_t byteSize = in.numBytes();
            cmdBuff
                .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / in.buffer() / eva::SYNC_SCOPE::TRANSFER_SRC)
                .copyBuffer(in.buffer(), outBuf(dstOffset, byteSize));
            dstOffset += byteSize;
        }

        cmdBuff.barrier(eva::SYNC_SCOPE::TRANSFER_DST / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
        return;
    }

    const auto& outShape = (*this)["out0"].shape();

    uint32_t innerSize = 1;
    for (uint32_t i = axisIdx + 1; i < outShape.size(); ++i)
        innerSize *= outShape[i];

    struct {
        uint32_t totalElements;
        uint32_t innerSize;
        uint32_t concatDimOut;
        uint32_t inputCount;
        uint32_t prefixSum[maxInputs];
    } constants = {(uint32_t) (*this)["out0"].numElements(), innerSize, outShape[axisIdx],
                   (uint32_t) arrived.size(), {}};

    // The inputs that arrived, packed into the leading bindings: the kernel reads
    // inputCount of them and never learns which slots they came from.
    std::vector<eva::Descriptor> descs(maxInputs + 1, eva::Buffer());
    descs[0] = outBuf;

    // inclusive prefix sum: prefixSum[i] = dim0 + dim1 + ... + dim_i
    uint32_t sum = 0;
    for (uint32_t k = 0; k < arrived.size(); ++k)
    {
        const Tensor& in = (*this)["in" + std::to_string(arrived[k])];
        descs[k + 1] = in.buffer();
        sum += in.shape()[axisIdx];
        constants.prefixSum[k] = sum;
    }

    descSet.write(descs);

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch2(constants.totalElements)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace vai::node

#endif // VAI_NODES_CONCAT_H
