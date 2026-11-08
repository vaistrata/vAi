#ifndef YOLOX_SPPF_H
#define YOLOX_SPPF_H

#include "vai/nodes/node-common.h"

namespace yolox {

inline eva::SpvBlob sppfSpv = eva::SpvBlob::readFrom(
    YOLOX_SHADER_OUTPUT_DIR "/sppf.comp.spv");

/////////////////////////////////////////////////////////////////////////////////////////
// SPPFNode - Three-stage 5x5 max pooling with channel concatenation
// Input : in0  [C, H, W], with H*W bounded by the shader shared-memory capacity
// Output: out0 [4*C, H, W] = concat(in0, pool1, pool2, pool3)
/////////////////////////////////////////////////////////////////////////////////////////
class SPPFNode : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        vai::Dtype oDtype = vai::Dtype::None;   // None: oDefaultDtype
    } config_;

    SPPFNode();
    SPPFNode(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};

inline SPPFNode::SPPFNode() : SPPFNode(Config()) {}

inline SPPFNode::SPPFNode(Config config)
: config_(config)
{
    addSlot("out0", vai::NodeSlot::output);
    addSlot("in0", vai::NodeSlot::input);

    config_.oDtype = (config_.oDtype == vai::Dtype::None) ? oDefaultDtype() : config_.oDtype;
}

inline void SPPFNode::prepare()
{
    const vai::Tensor& inTen = (*this)["in0"];
    VAI_ASSERT(inTen.validShape());
    VAI_ASSERT(inTen.shape().size() == 3);

    const PipeParams p = { .iDtype = inTen.dtype() };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        static eva::ComputePipeline cached[2][2];   // [typeI][typeO]
        eva::ComputePipeline& c = cached[uint32_t(p.iDtype)][uint32_t(config_.oDtype)];
        if (!c)
        {
            c = vai::getVaiDevice().createComputePipeline({
                .csStage = sppfSpv
                    + eva::constant_id<SPEC_ID_TYPE_I>(p.iDtype)
                    + eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),
            });
        }
        pipe = c;
        if (!descSet)
            descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
    }

    const auto& shape = inTen.shape();
    VAI_ASSERT(shape[1] * shape[2] <= 1024u);
    (*this)["out0"] = vai::Tensor(
        {4u * shape[0], shape[1], shape[2]},
        config_.oDtype);
}

inline void SPPFNode::run(eva::CommandBuffer cmdBuff)
{
    vai::Tensor& inTen = (*this)["in0"];
    vai::Tensor& outTen = (*this)["out0"];
    const auto& shape = inTen.shape();

    descSet.write({outTen.buffer(), inTen.buffer()});

    struct {
        uint32_t C, H, W;
    } constants = {shape[0], shape[1], shape[2]};

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch(constants.C, 1u, 1u)
        .barrier(
            eva::SYNC_SCOPE::COMPUTE_WRITE /
            outTen.buffer() /
            eva::SYNC_SCOPE::COMPUTE_READ);
}

} // namespace yolox

#endif // YOLOX_SPPF_H
