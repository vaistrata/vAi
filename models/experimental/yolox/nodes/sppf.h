#ifndef EXPERIMENTAL_YOLOX_NODES_SPPF_H
#define EXPERIMENTAL_YOLOX_NODES_SPPF_H

#include "vai/nodes/node-common.h"

namespace model::experimental::yolox::node {

inline eva::SpvBlob sppfSpv = eva::SpvBlob::readFrom(EXPERIMENTAL_YOLOX_SHADER_OUTPUT_DIR "/sppf.comp.spv");

/////////////////////////////////////////////////////////////////////////////////////////
// SPPF - Three-stage 5x5 max pooling with channel concatenation
// Input : in0  [C, H, W], with H*W bounded by the shader shared-memory capacity
// Output: out0 [4*C, H, W] = concat(in0, pool1, pool2, pool3)
/////////////////////////////////////////////////////////////////////////////////////////
class SPPF : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams
    {
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config
    {
    } config_;

    SPPF();
    SPPF(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};

inline SPPF::SPPF() : SPPF(Config()) {}

inline SPPF::SPPF(Config config) : config_(config)
{
    addSlot("out0", vai::NodeSlot::output);
    addSlot("in0", vai::NodeSlot::input);
}

inline void SPPF::prepare()
{
    const vai::Dtype oDtype = (oDtype_ == vai::Dtype::None) ? oDefaultDtype() : oDtype_;
    const vai::Tensor& inTen = (*this)["in0"];
    VAI_ASSERT(inTen.validShape());
    VAI_ASSERT(inTen.shape().size() == 3);

    const PipeParams p = {.iDtype = inTen.dtype()};
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        static eva::ComputePipeline cached[2][2];   // [typeI][typeO]
        eva::ComputePipeline& c = cached[uint32_t(p.iDtype)][uint32_t(oDtype)];
        if (!c)
        {
            c = vai::getVaiDevice().createComputePipeline({
                .csStage =
                    sppfSpv + eva::constant_id<SPEC_ID_TYPE_I>(p.iDtype) + eva::constant_id<SPEC_ID_TYPE_O>(oDtype),
            });
        }
        pipe = c;
        if (!descSet)
            descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
    }

    const auto& shape = inTen.shape();
    VAI_ASSERT(shape[1] * shape[2] <= 1024u);
    (*this)["out0"] = vai::Tensor({4u * shape[0], shape[1], shape[2]}, oDtype);
}

inline void SPPF::run(eva::CommandBuffer cmdBuff)
{
    vai::Tensor& inTen = (*this)["in0"];
    vai::Tensor& outTen = (*this)["out0"];
    const auto& shape = inTen.shape();

    descSet.write({outTen.buffer(), inTen.buffer()});

    struct
    {
        uint32_t C, H, W;
    } constants = {shape[0], shape[1], shape[2]};

    cmdBuff.bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch(constants.C, 1u, 1u)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outTen.buffer() / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace model::experimental::yolox::node

#endif   // EXPERIMENTAL_YOLOX_NODES_SPPF_H
