#ifndef EXPERIMENTAL_YOLOX_NODES_DECODE_H
#define EXPERIMENTAL_YOLOX_NODES_DECODE_H

#include "vai/nodes/node-common.h"
#include <cstdint>
#include <string>
#include <unordered_map>

namespace model::experimental::yolox::node {

inline eva::SpvBlob decodeSpv = eva::SpvBlob::readFrom(EXPERIMENTAL_YOLOX_SHADER_OUTPUT_DIR "/decode.comp.spv");

/////////////////////////////////////////////////////////////////////////////////////////
// Decode - Decode and interleave the three YOLOX prediction scales
// Inputs per scale: regobj [5,H,W], cls [numClasses,H,W]
// Output: out0 [N, 5 + numClasses], rows are cx, cy, w, h, obj, class probabilities
/////////////////////////////////////////////////////////////////////////////////////////
class Decode : public vai::Node
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
        uint32_t imageSize = 640;
        uint32_t numClasses = 80;
    } config_;

    Decode(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};

inline Decode::Decode(Config config) : config_(config)
{
    VAI_ASSERT(config_.imageSize >= 32u && (config_.imageSize % 32u) == 0u);
    VAI_ASSERT(config_.numClasses > 0u);

    addSlot("out0", vai::NodeSlot::output);
    for (const char* scale : {"p3", "p4", "p5"})
    {
        addSlot(std::string("regobj_") + scale, vai::NodeSlot::input);
        addSlot(std::string("cls_") + scale, vai::NodeSlot::input);
    }
}

inline void Decode::prepare()
{
    const vai::Dtype oDtype = (oDtype_ == vai::Dtype::None) ? oDefaultDtype() : oDtype_;
    const uint32_t channels = 5u + config_.numClasses;
    const uint32_t sizes[3] = {
        config_.imageSize / 8u,
        config_.imageSize / 16u,
        config_.imageSize / 32u,
    };
    const char* scales[3] = {"p3", "p4", "p5"};

    const vai::Dtype iDtype = (*this)["regobj_p3"].dtype();
    uint32_t n = 0u;
    for (uint32_t i = 0; i < 3u; ++i)
    {
        const vai::Tensor& regobj = (*this)[std::string("regobj_") + scales[i]];
        const vai::Tensor& cls = (*this)[std::string("cls_") + scales[i]];
        VAI_ASSERT(regobj.validShape() && cls.validShape());
        VAI_ASSERT(regobj.dtype() == iDtype);
        VAI_ASSERT(cls.dtype() == iDtype);
        VAI_ASSERT(regobj.isShapeOf(5u, sizes[i], sizes[i]));
        VAI_ASSERT(cls.isShapeOf(config_.numClasses, sizes[i], sizes[i]));
        n += sizes[i] * sizes[i];
    }

    const PipeParams p = {.iDtype = iDtype};
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        const uint32_t p3Size = config_.imageSize / 8u;
        const uint32_t p4Size = config_.imageSize / 16u;
        const uint32_t p5Size = config_.imageSize / 32u;
        const uint32_t p3N = p3Size * p3Size;
        const uint32_t p4N = p4Size * p4Size;
        const uint32_t p5N = p5Size * p5Size;
        const uint32_t n = p3N + p4N + p5N;

        const uint64_t cacheKey = uint64_t(p.iDtype) | (uint64_t(oDtype) << 4u) | (uint64_t(config_.imageSize) << 8u) |
                                  (uint64_t(config_.numClasses) << 32u);
        static std::unordered_map<uint64_t, eva::ComputePipeline> cached;
        auto it = cached.find(cacheKey);
        if (it == cached.end())
        {
            eva::ComputePipeline created = vai::getVaiDevice().createComputePipeline({
                .csStage = decodeSpv + eva::constant_id<0>(n) + eva::constant_id<1>(config_.numClasses) +
                           eva::constant_id<2>(p3N) + eva::constant_id<3>(p4N) + eva::constant_id<4>(p5N) +
                           eva::constant_id<5>(p3Size) + eva::constant_id<6>(p4Size) + eva::constant_id<7>(p5Size) +
                           eva::constant_id<8>(config_.imageSize) + eva::constant_id<SPEC_ID_TYPE_I>(p.iDtype) +
                           eva::constant_id<SPEC_ID_TYPE_O>(oDtype),
            });
            it = cached.emplace(cacheKey, std::move(created)).first;
        }
        pipe = it->second;
        if (!descSet)
            descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
    }

    (*this)["out0"] = vai::Tensor({n, channels}, oDtype);
}

inline void Decode::run(eva::CommandBuffer cmdBuff)
{
    vai::Tensor& outTen = (*this)["out0"];
    descSet.write({
        (*this)["regobj_p3"].buffer(),
        (*this)["cls_p3"].buffer(),
        (*this)["regobj_p4"].buffer(),
        (*this)["cls_p4"].buffer(),
        (*this)["regobj_p5"].buffer(),
        (*this)["cls_p5"].buffer(),
        outTen.buffer(),
    });

    cmdBuff.bindPipeline(pipe)
        .bindDescSets({descSet})
        .dispatch2(outTen.numElements())
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outTen.buffer() / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace model::experimental::yolox::node

#endif   // EXPERIMENTAL_YOLOX_NODES_DECODE_H
