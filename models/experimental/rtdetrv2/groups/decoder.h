#ifndef EXPERIMENTAL_RTDETRV2_GROUPS_DECODER_H
#define EXPERIMENTAL_RTDETRV2_GROUPS_DECODER_H

#include "vai/nodes/activation.h"
#include "vai/nodes/linear.h"
#include "vai/nodes/permute.h"
#include "vai/nodes/reshape.h"
#include "vai/nodes/slice.h"
#include "decoder-block.h"

namespace model::experimental::rtdetrv2::group {

/////////////////////////////////////////////////////////////////////////////////////////
// DecoderConfig - block count, prediction heads, and decoder layer settings
/////////////////////////////////////////////////////////////////////////////////////////
struct DecoderConfig
{
    uint32_t numLayers;
    uint32_t numClasses;
    uint32_t positionHiddenDim;
    DecoderLayerConfig layerCfg;
};

/////////////////////////////////////////////////////////////////////////////////////////
// Decoder - Sequential query updates and box refinement
// Input:  queries [Q, D], memory [S, D] (oDefaultDtype), boxLogits [Q, 4] (FP32)
// Output: pred_logits [Q, numClasses], pred_boxes [Q, 4] (both FP32)
// Memory is in crossCfg.featureShapes level order; D is crossCfg.inDim.
// Apply sigmoid to the initial box logits, then pass queries and normalized boxes
// through the blocks. Independent value projections share one GEMM; each
// block receives a contiguous view of its projected memory.
/////////////////////////////////////////////////////////////////////////////////////////
struct Decoder : public vai::NodeGroup
{
    vai::node::Linear valueProjection;
    vai::node::Reshape packedValues;
    vai::node::Permute perValues;
    std::vector<vai::node::Slice> valueViews;
    vai::node::Activation initBoxes;
    std::vector<DecoderBlock> blocks;

    Decoder(DecoderConfig config)
    : valueProjection(
          { .inDim = config.layerCfg.crossCfg.inDim, .outDim = config.numLayers * config.layerCfg.crossCfg.outDim })
    , packedValues({ -1, config.numLayers, config.layerCfg.crossCfg.outDim })
    , perValues({ 1, 0, 2 })
    , initBoxes({ .actType = vai::ActType::Sigmoid })
    {
        valueProjection - packedValues - perValues;
        valueViews.reserve(config.numLayers);
        blocks.reserve(config.numLayers);
        for (uint32_t i = 0; i < config.numLayers; ++i)
        {
            blocks.emplace_back(DecoderBlockConfig{
                .positionHiddenDim = config.positionHiddenDim,
                .numClasses = config.numClasses,
                .layerCfg = config.layerCfg,
                .hasClassHead = i + 1 == config.numLayers });
            valueViews.emplace_back(vai::node::Slice::Config{ .axis = 0, .start = i });
            perValues - valueViews.back() - "values" / blocks[i];
        }

        initBoxes - "boxes" / blocks.front();
        for (size_t i = 1; i < blocks.size(); ++i)
        {
            blocks[i - 1] - blocks[i];   // query 300, 256
            blocks[i - 1] / "boxesOut" - "boxes" / blocks[i];
        }

        defineSlot("memory", valueProjection.slot("in0"));
        defineSlot("queries", blocks.front().slot("in0"));
        defineSlot("boxLogits", initBoxes.slot("in0"));
        defineSlot("pred_logits", blocks.back().slot("logits"));
        defineSlot("pred_boxes", blocks.back().slot("boxesOut"));
    }
};

}   // namespace model::experimental::rtdetrv2::group

#endif   // EXPERIMENTAL_RTDETRV2_GROUPS_DECODER_H
