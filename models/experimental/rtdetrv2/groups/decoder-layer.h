#ifndef EXPERIMENTAL_RTDETRV2_GROUPS_DECODER_LAYER_H
#define EXPERIMENTAL_RTDETRV2_GROUPS_DECODER_LAYER_H

#include "../nodes/add.h"
#include "vai/nodes/identity.h"
#include "vai/nodes/layer-norm.h"
#include "vai/nodes/linear.h"
#include "ms-deformable-attention.h"
#include "multi-head-attention.h"

namespace model::experimental::rtdetrv2::group {

/////////////////////////////////////////////////////////////////////////////////////////
// DecoderLayerConfig - attention, feed-forward, and normalization settings
/////////////////////////////////////////////////////////////////////////////////////////
struct DecoderLayerConfig
{
    uint32_t ffnHiddenDim;
    float normEps;
    MSDeformableAttnConfig crossCfg;
};

/////////////////////////////////////////////////////////////////////////////////////////
// DecoderLayer - Self attention, deformable cross attention, and FFN
// Input:  in0 [1, Q, D], position [1, Q, D], values [1, S, D], boxes [1, Q, 1, 4] (FP32)
// Output: out0 [1, Q, D]
// Each attention/FFN sublayer uses a residual sum and LayerNorm.
// D is crossCfg.inDim = crossCfg.outDim.
/////////////////////////////////////////////////////////////////////////////////////////
struct DecoderLayer : public vai::NodeGroup
{
    vai::node::Identity input, position, values, boxes;
    MultiHeadAttention<QkvLayout::Separate> self;
    MSDeformableAttention cross;
    node::Add sum1, sum2, sum3;
    vai::node::LayerNorm norm1, norm2, norm3;
    vai::node::Linear fc1, fc2;

    DecoderLayer(DecoderLayerConfig config)
    : self({ .embedDim = config.crossCfg.inDim,
             .headDim = config.crossCfg.headDim,
             .numQHeads = config.crossCfg.numHeads,
             .posCfg = node::Add::Config{} })
    , cross(config.crossCfg)
    , norm1({ .eps = config.normEps })
    , norm2({ .eps = config.normEps })
    , norm3({ .eps = config.normEps })
    , fc1({ .inDim = config.crossCfg.inDim, .outDim = config.ffnHiddenDim, .actType = vai::ActType::Relu })
    , fc2({ .inDim = config.ffnHiddenDim, .outDim = config.crossCfg.inDim })
    {
        position - "position" / self;
        position - "position" / cross;

        input - self - "in0" / sum1;
        input - "in1" / sum1;
        sum1 - norm1 - cross - "in0" / sum2;
        norm1 - "in1" / sum2;
        boxes - "boxes" / cross;
        values - "values" / cross;

        sum2 - norm2 - fc1 - fc2 - "in0" / sum3;
        norm2 - "in1" / sum3;
        sum3 - norm3;

        defineSlot("in0", input.slot("in0"));
        defineSlot("position", position.slot("in0"));
        defineSlot("boxes", boxes.slot("in0"));
        defineSlot("values", values.slot("in0"));
        defineSlot("out0", norm3.slot("out0"));
    }
};

}   // namespace model::experimental::rtdetrv2::group

#endif   // EXPERIMENTAL_RTDETRV2_GROUPS_DECODER_LAYER_H
