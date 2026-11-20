#ifndef EXPERIMENTAL_RTDETRV2_GROUPS_DECODER_BLOCK_H
#define EXPERIMENTAL_RTDETRV2_GROUPS_DECODER_BLOCK_H

#include "../nodes/box-refine.h"
#include "vai/nodes/identity.h"
#include "vai/nodes/linear.h"
#include "vai/nodes/reshape.h"
#include "decoder-layer.h"

namespace model::experimental::rtdetrv2::group {

/////////////////////////////////////////////////////////////////////////////////////////
// DecoderBlockConfig - position encoding, decoder layer, and prediction heads
/////////////////////////////////////////////////////////////////////////////////////////
struct DecoderBlockConfig
{
    uint32_t positionHiddenDim;
    uint32_t numClasses;
    DecoderLayerConfig layerCfg;
    bool hasClassHead = false;   // last block only, for the final prediction
};

/////////////////////////////////////////////////////////////////////////////////////////
// DecoderBlock - One query update and box refinement stage
// Input:  in0 [Q, D], values [1, S, D] (oDefaultDtype), boxes [Q, 4] (FP32)
// Output: out0 [Q, D] (oDefaultDtype), boxesOut [Q, 4] (FP32 normalized cx, cy, w, h)
// Optional output: logits [Q, numClasses] (FP32), enabled by hasClassHead.
// D is layerCfg.crossCfg.inDim. Position encoding maps 4 -> positionHiddenDim -> D.
// The layer receives queries/positions [1, Q, D] and boxes [1, Q, 1, 4].
// Restore its output to [Q, D] before the prediction heads and the next block.
// The bbox0/1/2 linears predict delta; refined boxes are sigmoid(inverse_sigmoid(boxes) + delta).
/////////////////////////////////////////////////////////////////////////////////////////
struct DecoderBlock : public vai::NodeGroup
{
    vai::node::Identity boxes;
    vai::node::Reshape queryShape, positionShape, boxShape;
    vai::node::Linear pos0, pos1;
    DecoderLayer layer;
    vai::node::Reshape outputShape;
    vai::node::Linear bbox0, bbox1, bbox2;
    node::BoxRefine refine;
    std::optional<vai::node::Linear> classHead;

    DecoderBlock(DecoderBlockConfig config)
    : queryShape({ 1, -1, config.layerCfg.crossCfg.inDim })
    , positionShape({ 1, -1, config.layerCfg.crossCfg.inDim })
    , boxShape({ 1, -1, 1, 4 })
    , pos0({ .inDim = 4, .outDim = config.positionHiddenDim, .epilogue = vai::Epilogue::Relu })
    , pos1({ .inDim = config.positionHiddenDim, .outDim = config.layerCfg.crossCfg.inDim })
    , layer(config.layerCfg)
    , outputShape({ -1, config.layerCfg.crossCfg.inDim })
    , bbox0({ .inDim = config.layerCfg.crossCfg.inDim,
              .outDim = config.layerCfg.crossCfg.inDim,
              .epilogue = vai::Epilogue::Relu })
    , bbox1({ .inDim = config.layerCfg.crossCfg.inDim,
              .outDim = config.layerCfg.crossCfg.inDim,
              .epilogue = vai::Epilogue::Relu })
    , bbox2({ .inDim = config.layerCfg.crossCfg.inDim, .outDim = 4 })
    {
        boxes - pos0 - pos1 - positionShape - "position" / layer;   // query position head
        boxes - boxShape - "boxes" / layer;
        queryShape - layer - outputShape - bbox0 - bbox1 - bbox2 - "delta" / refine;
        boxes - "boxes" / refine;

        if (config.hasClassHead)
        {
            classHead.emplace(
                vai::node::Linear::Config{ .inDim = config.layerCfg.crossCfg.inDim, .outDim = config.numClasses });
            outputShape - classHead;
            defineSlot("logits", classHead->slot("out0"));
        }

        defineSlot("in0", queryShape.slot("in0"));
        defineSlot("values", layer.slot("values"));
        defineSlot("boxes", boxes.slot("in0"));
        defineSlot("out0", outputShape.slot("out0"));
        defineSlot("boxesOut", refine.slot("out0"));
    }
};

}   // namespace model::experimental::rtdetrv2::group

#endif   // EXPERIMENTAL_RTDETRV2_GROUPS_DECODER_BLOCK_H
