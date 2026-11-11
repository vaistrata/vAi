#ifndef ZIMAGE_PAD_ROWS_H
#define ZIMAGE_PAD_ROWS_H

#include "vai/nodes/node-common.h"

namespace zimage {

/*=============================================================================
  PadRowsNode - pad a token sequence to a row multiple with a learned token
  Input : in0 [N, D]
  Params: padToken [D] (or [1, D])
  Output: out0 [ceil(N / multiple) * multiple, D]; rows past N hold padToken

  Z-Image pads both the image and the caption sequence to a multiple of 32
  and overwrites the pad rows with x_pad_token / cap_pad_token. Batch-1
  attention masks nothing, so the pad rows are attended like real tokens —
  they must hold exactly these values.
=============================================================================*/
class PadRowsNode : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;
        vai::Dtype wDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        uint32_t multiple = 32;
        vai::Dtype wDtype = vai::Dtype::None;   // None: wDefaultDtype
        vai::Dtype oDtype = vai::Dtype::None;   // None: oDefaultDtype
    } config_;

    PadRowsNode(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};


inline PadRowsNode::PadRowsNode(Config config)
: config_(config)
{
    addSlot("out0", vai::NodeSlot::output);
    addSlot("in0", vai::NodeSlot::input);
    addSlot("padToken", vai::NodeSlot::persistent);

    config_.wDtype = (config_.wDtype == vai::Dtype::None) ? wDefaultDtype() : config_.wDtype;
    config_.oDtype = (config_.oDtype == vai::Dtype::None) ? oDefaultDtype() : config_.oDtype;
}

inline void PadRowsNode::prepare()
{
    const vai::Tensor& inTen = (*this)["in0"];
    const vai::Tensor& pTen  = (*this)["padToken"];

    VAI_ASSERT(inTen.shape().size() == 2);
    const uint32_t N = inTen.shape()[0], D = inTen.shape()[1];
    VAI_ASSERT(pTen.dtype() == config_.wDtype);
    VAI_ASSERT(pTen.numElements() == D);

    const PipeParams p = { .iDtype = inTen.dtype(), .wDtype = pTen.dtype() };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        pipe = pipelinePool().request(ZIMAGE_SHADER_OUTPUT_DIR, "zimage-pad-rows", {
            eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),          // typeO
            eva::constant_id<SPEC_ID_TYPE_I>(p.iDtype),                // typeI
            eva::constant_id<SPEC_ID_TYPE_W>(p.wDtype)                 // typeW (pad token)
        });
        if (!descSet)
            descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
    }

    const uint32_t Npad = CEIL_DIV(N, config_.multiple) * config_.multiple;
    (*this)["out0"] = vai::Tensor({Npad, D}, config_.oDtype);
}

inline void PadRowsNode::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf  = (*this)["in0"].buffer();
    auto pBuf   = (*this)["padToken"].buffer();

    descSet.write({ outBuf, inBuf, pBuf });

    struct {
        uint32_t N, Npad, D;
    } constants = { (*this)["in0"].shape()[0], (*this)["out0"].shape()[0], (*this)["out0"].shape()[1] };

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch(constants.Npad)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

} // namespace zimage

#endif // ZIMAGE_PAD_ROWS_H
