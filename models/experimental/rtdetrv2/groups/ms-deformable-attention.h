#ifndef EXPERIMENTAL_RTDETRV2_GROUPS_MS_DEFORMABLE_ATTENTION_H
#define EXPERIMENTAL_RTDETRV2_GROUPS_MS_DEFORMABLE_ATTENTION_H

#include "../nodes/add.h"
#include "../nodes/deformable-attention.h"
#include "vai/nodes/linear.h"
#include "vai/nodes/reshape.h"
#include <iterator>
#include <numeric>

namespace model::experimental::rtdetrv2::group {

/////////////////////////////////////////////////////////////////////////////////////////
// MS-Deformable Attention Config
/////////////////////////////////////////////////////////////////////////////////////////

struct MSDeformableAttnConfig
{
    uint32_t inDim;
    uint32_t outDim;
    uint32_t headDim;
    uint32_t numHeads;
    uint32_t featureShapes[3][2];    // [h, w] for each feature level, P3, P4, P5
    uint32_t numSamplingPoints[3];   // number of sampling points for each feature level

    uint32_t totalPoints() const   // sum of numSamplingPoints
    {
        return std::accumulate(std::begin(numSamplingPoints), std::end(numSamplingPoints), 0u);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////
// MSDeformableAttention - Multi-scale deformable attention over three feature levels
// Input:  in0 [1, Q, inDim], position [1, Q, inDim], values [1, S, outDim],
//         boxes [1, Q, 1, 4] (FP32 normalized center-x, center-y, width, height)
// Output: out0 [1, Q, outDim]
// S is the sum of the configured feature areas. Offsets and attention logits
// are projected from in0 + position. DeformableAttention computes sampling coordinates
// and softmax weights, then samples values and sums over points within one node.
/////////////////////////////////////////////////////////////////////////////////////////
struct MSDeformableAttention : public vai::NodeGroup
{
    node::Add queryPosAdd;   // query + position embedding
    vai::node::Linear offsets, logits, out;
    vai::node::Reshape valueShape, offsetShape, logitShape;
    node::DeformableAttention attnCore;

    MSDeformableAttention(MSDeformableAttnConfig config)
    : offsets({ .inDim = config.inDim, .outDim = config.numHeads * config.totalPoints() * 2 })
    , logits({ .inDim = config.inDim, .outDim = config.numHeads * config.totalPoints() })
    , out({ .inDim = config.outDim, .outDim = config.outDim })
    , valueShape({ 1, -1, config.numHeads, config.headDim })
    , offsetShape({ 1, -1, config.numHeads, config.totalPoints(), 2 })
    , logitShape({ 1, -1, config.numHeads, config.totalPoints() })
    , attnCore(
          { .spatialShapes = { { config.featureShapes[0][0], config.featureShapes[0][1] },
                               { config.featureShapes[1][0], config.featureShapes[1][1] },
                               { config.featureShapes[2][0], config.featureShapes[2][1] } },
            .pointCounts = { std::begin(config.numSamplingPoints), std::end(config.numSamplingPoints) } })
    {
        queryPosAdd - logits - logitShape - "logits" / attnCore;
        queryPosAdd - offsets - offsetShape - "offsets" / attnCore;
        valueShape - "values" / attnCore;
        attnCore - out;

        defineSlot("in0", queryPosAdd.slot("in0"));
        defineSlot("position", queryPosAdd.slot("in1"));
        defineSlot("values", valueShape.slot("in0"));
        defineSlot("boxes", attnCore.slot("boxes"));
        defineSlot("out0", out.slot("out0"));
    }
};

}   // namespace model::experimental::rtdetrv2::group

#endif   // EXPERIMENTAL_RTDETRV2_GROUPS_MS_DEFORMABLE_ATTENTION_H
