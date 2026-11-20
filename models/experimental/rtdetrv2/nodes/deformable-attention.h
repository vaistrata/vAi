#ifndef EXPERIMENTAL_RTDETRV2_NODES_DEFORMABLE_ATTENTION_H
#define EXPERIMENTAL_RTDETRV2_NODES_DEFORMABLE_ATTENTION_H

#include "vai/nodes/common.h"
#include <limits>

namespace model::experimental::rtdetrv2::node {

/////////////////////////////////////////////////////////////////////////////////////////
// DeformableAttention - Sampling coordinates, softmax, and weighted bilinear sampling
// Input:  values [B, S, M, D] (FP32 or FP16), boxes [B, Q, R, C],
//         offsets [B, Q, M, P, 2], logits [B, Q, M, P] (remaining inputs FP32)
// Output: out0 [B, Q, M*D] (FP32)
//
// S=sum(H_l*W_l), P=sum(pointCounts_l), in spatialShapes/pointCounts level order.
// R=1 shares boxes across levels; R=L supplies a box per level.
// For a level with P_l points, C=4 boxes use normalized (cx, cy, width, height):
//   locations = boxes.xy + offsets / P_l * boxes.wh * offsetScale.
// C=2 also accepts point coordinates: boxes.xy + offsets / (W_l, H_l).
// weights = softmax(logits, axis=P), over ALL levels for each (batch, query, head).
// out[b,q,m,d] = sum_p(weights[b,q,m,p] * bilinear(values, locations[b,q,m,p], m,d)).
//
// Two kernels share temporary locations/weights buffers within this node. The second
// indexes each level directly, with no Split/Concatenate buffers. Pixel coordinates
// are (u*W_l-0.5, v*H_l-0.5); out-of-bounds neighbors are zero (align_corners=false).
// Coordinates, interpolation, accumulation, and output use FP32. Output is query-major,
// so merging M*D needs no permute.
// All tensors are contiguous with positive dimensions; inputs must be finite.
/////////////////////////////////////////////////////////////////////////////////////////
class DeformableAttention : public vai::Node
{
    eva::ComputePipeline paramsPipe_, samplePipe_;
    eva::DescriptorSet paramsDs_, sampleDs_;
    uint32_t numTokens_ = 0, numPoints_ = 0;
    uint32_t numQueries_ = 0, numHeads_ = 0, headDim_ = 0, outElems_ = 0;
    uint32_t rows_ = 0, refLevels_ = 0, refDims_ = 0;
    std::array<uint32_t, 2> paramsGroups_{}, sampleGroups_{};

    static uint32_t elementCount(const std::vector<uint32_t>& shape);
    static std::array<uint32_t, 2> dispatchGroups(eva::Device device, uint32_t count);

public:
    struct Config
    {
        std::vector<std::array<uint32_t, 2>> spatialShapes;   // (height, width)
        std::vector<uint32_t> pointCounts;
        float offsetScale = .5f;   // used only for 4D box references
    };
    const Config config_;

    explicit DeformableAttention(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmd) override;
};

inline uint32_t DeformableAttention::elementCount(const std::vector<uint32_t>& shape)
{
    uint64_t count = 1;
    for (uint32_t dim : shape)
    {
        VAI_ASSERT(dim > 0 && count <= uint64_t(UINT32_MAX) / dim);
        count *= dim;
    }
    return uint32_t(count);
}

inline std::array<uint32_t, 2> DeformableAttention::dispatchGroups(eva::Device device, uint32_t count)
{
    const auto limits = device.maxComputeWorkGroupCount();
    const uint64_t groups = (uint64_t(count) + 63) / 64;
    const uint32_t gx = uint32_t(std::min(groups, uint64_t(limits[0])));
    const uint32_t gy = uint32_t((groups + gx - 1) / gx);
    VAI_ASSERT(gy <= limits[1]);
    return { gx, gy };
}

inline DeformableAttention::DeformableAttention(Config config)
: config_{
    .spatialShapes = std::move(config.spatialShapes),
    .pointCounts = std::move(config.pointCounts),
    .offsetScale = config.offsetScale
}
{
    VAI_ASSERT(std::isfinite(config_.offsetScale));
    const size_t levels = config_.spatialShapes.size();
    constexpr uint64_t maxIndex = std::numeric_limits<uint32_t>::max();
    VAI_ASSERT(levels > 0 && levels == config_.pointCounts.size() && levels <= maxIndex / 5);
    std::vector<uint32_t> paramsMetadata, sampleMetadata;
    uint64_t tokens = 0, points = 0;
    for (size_t l = 0; l < levels; ++l)
    {
        const auto [h, w] = config_.spatialShapes[l];
        const uint32_t count = config_.pointCounts[l];
        VAI_ASSERT(h > 0 && w > 0 && count > 0);
        VAI_ASSERT(h <= uint32_t(INT32_MAX) && w <= uint32_t(INT32_MAX));
        paramsMetadata.insert(paramsMetadata.end(), { h, w, count });
        sampleMetadata.insert(sampleMetadata.end(), { h, w, uint32_t(tokens), uint32_t(points), count });
        tokens += uint64_t(h) * w;
        points += count;
        VAI_ASSERT(tokens <= maxIndex && points <= maxIndex);
    }
    numTokens_ = uint32_t(tokens);
    numPoints_ = uint32_t(points);

    for (auto name : { "values", "boxes", "offsets", "logits" })
        addSlot(name, vai::NodeSlot::input);
    addSlot("out0", vai::NodeSlot::output);
    addSlot("locations", vai::NodeSlot::temporary);
    addSlot("weights", vai::NodeSlot::temporary);
    addSlot("paramsMetadata", vai::NodeSlot::persistent);
    addSlot("sampleMetadata", vai::NodeSlot::persistent);
    (*this)["paramsMetadata"] = vai::Tensor({ uint32_t(levels), 3 }, vai::Dtype::Fp32).setData(paramsMetadata.data());
    (*this)["sampleMetadata"] = vai::Tensor({ uint32_t(levels), 5 }, vai::Dtype::Fp32).setData(sampleMetadata.data());
    paramsPipe_ =
        pipelinePool().request(EXPERIMENTAL_RTDETRV2_SHADER_OUTPUT_DIR, "ms-deformable-sampling-params", {}, {});

    paramsDs_ = vai::gVaiDestSetPool(paramsPipe_.descSetLayout(0));
}

inline void DeformableAttention::prepare()
{
    const auto& values = (*this)["values"];
    VAI_ASSERT(values.dtype() == vai::Dtype::Fp32 || values.dtype() == vai::Dtype::Fp16);
    samplePipe_ = pipelinePool().request(
        EXPERIMENTAL_RTDETRV2_SHADER_OUTPUT_DIR,
        "grid-sample-fused",
        { eva::constant_id<SPEC_ID_TYPE_I>(values.dtype()) },
        {});
    if (!sampleDs_)
        sampleDs_ = vai::gVaiDestSetPool(samplePipe_.descSetLayout(0));
    const auto& boxes = (*this)["boxes"];
    const auto& offsets = (*this)["offsets"];
    const auto& logits = (*this)["logits"];
    VAI_ASSERT(values.validShape() && values.shape().size() == 4);
    VAI_ASSERT(boxes.validShape() && boxes.shape().size() == 4);
    VAI_ASSERT(offsets.validShape() && offsets.shape().size() == 5);
    VAI_ASSERT(logits.validShape() && logits.shape().size() == 4);
    VAI_ASSERT(
        boxes.dtype() == vai::Dtype::Fp32 && offsets.dtype() == vai::Dtype::Fp32 && logits.dtype() == vai::Dtype::Fp32);
    const uint32_t batch = values.shape()[0];
    numHeads_ = values.shape()[2];
    headDim_ = values.shape()[3];
    numQueries_ = offsets.shape()[1];
    refLevels_ = boxes.shape()[2];
    refDims_ = boxes.shape()[3];
    VAI_ASSERT(refLevels_ == 1 || refLevels_ == config_.spatialShapes.size());
    VAI_ASSERT(refDims_ == 2 || refDims_ == 4);
    VAI_ASSERT(values.shape()[1] == numTokens_);
    VAI_ASSERT(boxes.isShapeOf(batch, numQueries_, refLevels_, refDims_));
    VAI_ASSERT(offsets.isShapeOf(batch, numQueries_, numHeads_, numPoints_, 2));
    VAI_ASSERT(logits.isShapeOf(batch, numQueries_, numHeads_, numPoints_));
    elementCount(values.shape());
    elementCount(boxes.shape());
    rows_ = elementCount(offsets.shape()) / 2 / numPoints_;
    const std::vector<uint32_t> outShape{ batch, numQueries_, numHeads_ * headDim_ };
    outElems_ = elementCount(outShape);
    paramsGroups_ = dispatchGroups(device(), rows_);
    sampleGroups_ = dispatchGroups(device(), outElems_);
    (*this)["locations"] = vai::Tensor(offsets.shape(), vai::Dtype::Fp32);
    (*this)["weights"] = vai::Tensor(logits.shape(), vai::Dtype::Fp32);
    (*this)["out0"] = vai::Tensor(outShape, vai::Dtype::Fp32);
}

inline void DeformableAttention::run(eva::CommandBuffer cmd)
{
    auto output = (*this)["out0"].buffer();
    auto values = (*this)["values"].buffer();
    auto locations = (*this)["locations"].buffer();
    auto weights = (*this)["weights"].buffer();
    auto boxes = (*this)["boxes"].buffer();
    auto offsets = (*this)["offsets"].buffer();
    auto logits = (*this)["logits"].buffer();
    const struct
    {
        uint32_t rows, heads, points, refLevels, refDims, levels;
        float scale;
    } paramsConstants{
        rows_, numHeads_, numPoints_, refLevels_, refDims_, uint32_t(config_.spatialShapes.size()), config_.offsetScale
    };
    const uint32_t sampleConstants[] = {
        outElems_, numQueries_, numTokens_, numHeads_, headDim_, numPoints_, uint32_t(config_.spatialShapes.size())
    };
    const eva::SYNC_SCOPE outputReads(
        { eva::PIPELINE_STAGE::COMPUTE_SHADER | eva::PIPELINE_STAGE::TRANSFER,
          eva::ACCESS::SHADER_READ | eva::ACCESS::TRANSFER_READ });
    paramsDs_.write({ locations, weights, boxes, offsets, logits, (*this)["paramsMetadata"].buffer() });
    sampleDs_.write({ output, values, locations, weights, (*this)["sampleMetadata"].buffer() });
    cmd.bindPipeline(paramsPipe_)
        .bindDescSets({ paramsDs_ })
        .setPushConstants(0, sizeof(paramsConstants), &paramsConstants)
        .dispatch(paramsGroups_[0], paramsGroups_[1])
        // Make the first kernel's temporary results visible to the sampling kernel.
        .barrier({ eva::SYNC_SCOPE::COMPUTE_WRITE / locations / eva::SYNC_SCOPE::COMPUTE_READ,
                   eva::SYNC_SCOPE::COMPUTE_WRITE / weights / eva::SYNC_SCOPE::COMPUTE_READ })
        .bindPipeline(samplePipe_)
        .bindDescSets({ sampleDs_ })
        .setPushConstants(0, sizeof(sampleConstants), sampleConstants)
        .dispatch(sampleGroups_[0], sampleGroups_[1])
        // The graph may reuse every input and temporary allocation after this node.
        .barrier(
            { eva::SYNC_SCOPE::COMPUTE_READ / values / eva::SYNC_SCOPE::COMPUTE_WRITE,
              eva::SYNC_SCOPE::COMPUTE_READ / boxes / eva::SYNC_SCOPE::COMPUTE_WRITE,
              eva::SYNC_SCOPE::COMPUTE_READ / offsets / eva::SYNC_SCOPE::COMPUTE_WRITE,
              eva::SYNC_SCOPE::COMPUTE_READ / logits / eva::SYNC_SCOPE::COMPUTE_WRITE,
              eva::SYNC_SCOPE::COMPUTE_READ / locations / eva::SYNC_SCOPE::COMPUTE_WRITE,
              eva::SYNC_SCOPE::COMPUTE_READ / weights / eva::SYNC_SCOPE::COMPUTE_WRITE,
              eva::SYNC_SCOPE::COMPUTE_WRITE / output / outputReads });
}

}   // namespace model::experimental::rtdetrv2::node

#endif   // EXPERIMENTAL_RTDETRV2_NODES_DEFORMABLE_ATTENTION_H
