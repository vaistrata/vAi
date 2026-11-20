#ifndef EXPERIMENTAL_RTDETRV2_GROUPS_ENCODER_LAYER_H
#define EXPERIMENTAL_RTDETRV2_GROUPS_ENCODER_LAYER_H

#include "../nodes/add.h"
#include "vai/nodes/activation.h"
#include "vai/nodes/identity.h"
#include "vai/nodes/layer-norm.h"
#include "vai/nodes/linear.h"
#include "multi-head-attention.h"

namespace model::experimental::rtdetrv2::group {

/////////////////////////////////////////////////////////////////////////////////////////
// EncoderLayerConfig - attention dimensions and feed-forward settings
/////////////////////////////////////////////////////////////////////////////////////////
struct EncoderLayerConfig
{
    uint32_t embedDim = 256;
    uint32_t headDim = 32;
    uint32_t numHeads = 8;
    uint32_t ffnHiddenDim = 1024;
    float eps = 1e-5f;
};

/////////////////////////////////////////////////////////////////////////////////////////
// EncoderLayer - Self attention and GELU feed-forward network with post-normalization
// Input:  in0 [B, T, D]
// Params: position [B, T, D] (fixed positional embedding)
// Output: out0 [B, T, D]
// Each sublayer adds its residual before LayerNorm.
/////////////////////////////////////////////////////////////////////////////////////////
struct EncoderLayer : public vai::NodeGroup
{
    vai::node::Identity input;
    MultiHeadAttention<QkvLayout::Separate> attention;
    node::Add sum1, sum2;
    vai::node::LayerNorm norm1, norm2;
    vai::node::Linear fc1, fc2;
    vai::node::Activation gelu;

    EncoderLayer(EncoderLayerConfig config)
    : attention(
          { .embedDim = config.embedDim,
            .headDim = config.headDim,
            .numQHeads = config.numHeads,
            .posCfg = node::Add::Config{ .constantIn1 = true } })
    , norm1({ .eps = config.eps })
    , norm2({ .eps = config.eps })
    , fc1({ .inDim = config.embedDim, .outDim = config.ffnHiddenDim })
    , fc2({ .inDim = config.ffnHiddenDim, .outDim = config.embedDim })
    , gelu({ .epilogue = vai::Epilogue::GeluTanh })
    {
        input - attention - "in0" / sum1;
        input - "in1" / sum1;
        sum1 - norm1 - fc1 - gelu - fc2 - "in0" / sum2;
        norm1 - "in1" / sum2;
        sum2 - norm2;

        defineSlot("in0", input.slot("in0"));
        defineSlot("position", attention.slot("position"));
        defineSlot("out0", norm2.slot("out0"));
    }
};

}   // namespace model::experimental::rtdetrv2::group

#endif   // EXPERIMENTAL_RTDETRV2_GROUPS_ENCODER_LAYER_H
