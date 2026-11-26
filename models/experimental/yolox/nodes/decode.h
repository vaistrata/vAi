#ifndef EXPERIMENTAL_YOLOX_NODES_DECODE_H
#define EXPERIMENTAL_YOLOX_NODES_DECODE_H

#include "vai/nodes/common.h"
#include <cstdint>
#include <string>

namespace model::experimental::yolox::node {

/*=============================================================================
  Decode - Decode and interleave the three YOLOX prediction scales
  Inputs per scale: regobj [...,5,H,W], cls [...,numClasses,H,W]
  Output: out0 [...,N,5 + numClasses], rows are cx, cy, w, h, obj, class probabilities
=============================================================================*/
class Decode : public vai::Node
{
    eva::ComputePipeline pipe_;
    eva::DescriptorSet descSet_;

public:
    struct PipeParams
    {
        vai::Dtype iDtype = vai::Dtype::None;
        bool batched = false;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config
    {
        uint32_t imageSize = 640;
        uint32_t numClasses = 80;
    } config_;

    Decode(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams_;
};

inline Decode::Decode(Config config)
: config_(config)
{
    VAI_ASSERT(config_.imageSize >= 32u && (config_.imageSize % 32u) == 0u);
    VAI_ASSERT(config_.numClasses > 0u);

    addSlot("out0", vai::NodeSlot::output);
    for (const char* scale : { "p3", "p4", "p5" })
    {
        addSlot(std::string("regobj_") + scale, vai::NodeSlot::input);
        addSlot(std::string("cls_") + scale, vai::NodeSlot::input);
    }
}

inline void Decode::prepare()
{
    const vai::Dtype oDtype = (oDtype_ == vai::Dtype::None) ? oDefaultDtype() : oDtype_;
    const auto& first = (*this)["regobj_p3"];
    const auto& shape = first.shape();
    VAI_ASSERT(shape.size() == 3 || shape.size() == 4);
    const uint32_t batch = vai::outerSize(shape, 3);
    uint32_t n = 0;
    for (uint32_t i = 0; i < 3; ++i)
    {
        const auto scale = "p" + std::to_string(i + 3);
        const uint32_t side = config_.imageSize / (8u << i);
        auto expected = shape;
        expected[shape.size() - 2] = expected.back() = side;
        expected[shape.size() - 3] = 5u;
        const auto& reg = (*this)["regobj_" + scale];
        const auto& cls = (*this)["cls_" + scale];
        VAI_ASSERT(reg.shape() == expected && reg.dtype() == first.dtype());
        expected[shape.size() - 3] = config_.numClasses;
        VAI_ASSERT(cls.shape() == expected && cls.dtype() == first.dtype());
        n += side * side;
    }
    VAI_ASSERT(batch && batch <= device().maxComputeWorkGroupCount()[2]);
    const PipeParams p = { .iDtype = first.dtype(), .batched = batch > 1 };
    if (!pipe_ || p != pipeParams_)
    {
        pipeParams_ = p;
        pipe_ = pipelinePool().request(
            EXPERIMENTAL_YOLOX_SHADER_OUTPUT_DIR,
            "decode",
            {
                eva::constant_id<0>(config_.imageSize),
                eva::constant_id<1>(config_.numClasses),
                eva::constant_id<2>(p.batched),
                eva::constant_id<SPEC_ID_TYPE_I>(p.iDtype),
                eva::constant_id<SPEC_ID_TYPE_O>(oDtype),
            });
        if (!descSet_)
            descSet_ = vai::gVaiDestSetPool(pipe_.descSetLayout(0));
    }

    std::vector<uint32_t> outShape;
    if (shape.size() == 4)
        outShape.push_back(batch);
    outShape.insert(outShape.end(), { n, 5u + config_.numClasses });
    (*this)["out0"] = vai::Tensor(outShape, oDtype);
}

inline void Decode::run(eva::CommandBuffer cmdBuff)
{
    vai::Tensor& outTen = (*this)["out0"];
    descSet_.write({
        (*this)["regobj_p3"].buffer(),
        (*this)["cls_p3"].buffer(),
        (*this)["regobj_p4"].buffer(),
        (*this)["cls_p4"].buffer(),
        (*this)["regobj_p5"].buffer(),
        (*this)["cls_p5"].buffer(),
        outTen.buffer(),
    });

    cmdBuff.bindPipeline(pipe_)
        .bindDescSets({ descSet_ })
        .dispatch2(
            outTen.shape()[outTen.shape().size() - 2] * outTen.shape().back(), 1u, vai::outerSize(outTen.shape(), 2))
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outTen.buffer() / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace model::experimental::yolox::node

#endif   // EXPERIMENTAL_YOLOX_NODES_DECODE_H
