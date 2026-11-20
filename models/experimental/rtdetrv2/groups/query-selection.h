#ifndef EXPERIMENTAL_RTDETRV2_GROUPS_QUERY_SELECTION_H
#define EXPERIMENTAL_RTDETRV2_GROUPS_QUERY_SELECTION_H

#include "../nodes/add.h"
#include "../nodes/index-select.h"
#include "../nodes/mul.h"
#include "../nodes/query-score.h"
#include "../nodes/topk.h"
#include "vai/nodes/layer-norm.h"
#include "vai/nodes/linear.h"
#include <iterator>

namespace model::experimental::rtdetrv2::group {

/////////////////////////////////////////////////////////////////////////////////////////
// QuerySelectionConfig - candidate features, selected query count, and anchor shapes
/////////////////////////////////////////////////////////////////////////////////////////
struct QuerySelectionConfig
{
    uint32_t embedDim;
    uint32_t numQueries;
    uint32_t numClasses;
    float normEps;
    uint32_t featureShapes[3][2];
};

/////////////////////////////////////////////////////////////////////////////////////////
// generateAnchors - Fixed anchors and validity mask for the configured feature pyramid
// Returns [S, 4] FP32 anchor logits and [S, 1] binary values in maskDtype.
// S is the sum of the feature areas, in the same level order as sampling.
// Invalid anchors have infinite logits and a zero mask value.
/////////////////////////////////////////////////////////////////////////////////////////
inline std::pair<vai::Tensor, vai::Tensor> generateAnchors(
    const uint32_t (&spatialShapes)[3][2], vai::Dtype maskDtype = vai::Dtype::Fp32)
{
    uint32_t numTokens = 0;
    for (const auto& shape : spatialShapes)
    {
        numTokens += shape[0] * shape[1];   // 8400
    }
    std::vector<float> anchors, validMask;
    anchors.reserve(size_t(numTokens) * 4);
    validMask.reserve(numTokens);
    for (uint32_t level = 0; level < std::size(spatialShapes); ++level)
    {
        const auto [height, width] = spatialShapes[level];
        const float size = .05f * float(1u << level);
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x)
            {
                float b[] = { (x + 0.5f) / width, (y + 0.5f) / height, size, size };
                bool valid = true;
                for (float v : b)
                    valid &= v > 0.01f && v < 0.99f;
                validMask.push_back(valid ? 1.0f : 0.0f);
                for (float v : b)
                    anchors.push_back(valid ? std::log(v / (1 - v)) : std::numeric_limits<float>::infinity());
            }
    }
    // Broadcast one validity value over each feature row. Binary 0/1 values
    // remain exact in FP16; anchor logits themselves always remain FP32.
    return { vai::Tensor({ numTokens, 4 }, vai::Dtype::Fp32).setFloats(std::move(anchors)),
             vai::Tensor({ numTokens, 1 }, maskDtype).setFloats(std::move(validMask)) };
}

/////////////////////////////////////////////////////////////////////////////////////////
// QuerySelection - Select candidate features before predicting their boxes
// Input:  memory [S, D] (oDefaultDtype), in featureShapes level order
// Output: queries [Q, D] (oDefaultDtype), boxLogits [Q, 4] (FP32),
//         indices [Q] (raw uint32 in fp32 slots)
// Q is numQueries; D is embedDim. boxLogits are inverse-sigmoid-space cx, cy, w, h.
// The decoder applies sigmoid to boxLogits before its first block.
/////////////////////////////////////////////////////////////////////////////////////////
struct QuerySelection : public vai::NodeGroup
{
    node::Mul mask;
    vai::node::Linear proj, classHead;
    vai::node::LayerNorm norm;
    vai::node::Linear bbox0, bbox1, bbox2;
    node::Add anchorAdd;
    node::QueryScore queryScore;
    node::TopK topk;
    node::IndexSelect querySelect, anchorSelect;

    QuerySelection(QuerySelectionConfig config)
    : mask({ .constantIn1 = true })
    , proj({ .inDim = config.embedDim, .outDim = config.embedDim })
    , classHead({ .inDim = config.embedDim, .outDim = config.numClasses })
    , norm({ .eps = config.normEps })
    , bbox0({ .inDim = config.embedDim, .outDim = config.embedDim, .epilogue = vai::Epilogue::Relu })
    , bbox1({ .inDim = config.embedDim, .outDim = config.embedDim, .epilogue = vai::Epilogue::Relu })
    , bbox2({ .inDim = config.embedDim, .outDim = 4 })
    , topk({ .k = config.numQueries })
    , querySelect({})
    , anchorSelect({ .constantInput = true })
    {
        auto [anchors, validMask] = generateAnchors(config.featureShapes, proj.oDefaultDtype());
        mask["in1"] = validMask;
        anchorSelect["in0"] = anchors;

        mask - proj - norm;
        norm - classHead - queryScore - "in0" / topk;
        norm - querySelect - bbox0 - bbox1 - bbox2 - "in0" / anchorAdd;
        topk / "indices" - "indices" / querySelect;
        topk / "indices" - "indices" / anchorSelect;
        anchorSelect - "in1" / anchorAdd;

        defineSlot("memory", mask.slot("in0"));
        defineSlot("queries", querySelect.slot("out0"));
        defineSlot("boxLogits", anchorAdd.slot("out0"));
        defineSlot("indices", topk.slot("indices"));
    }
};

}   // namespace model::experimental::rtdetrv2::group

#endif   // EXPERIMENTAL_RTDETRV2_GROUPS_QUERY_SELECTION_H
