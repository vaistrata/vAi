#ifndef PRODUCTION_ACT_GROUPS_ENCODER_BLOCK_H
#define PRODUCTION_ACT_GROUPS_ENCODER_BLOCK_H

#include "vai/groups/feed-forward.h"
#include "vai/groups/multi-head-attention.h"
#include "vai/nodes/identity.h"
#include "vai/nodes/layer-norm.h"


namespace model::production::act::group {


/////////////////////////////////////////////////////////////////////////////////////////
// EncoderBlock - DETR post-norm block, the position added to Q and K only
// Input : in0  [T, D]
// Output: out0 [T, D]
//
// h = norm1(x + attn(q = k = x + pos, v = x)),  out0 = norm2(h + fc2(relu(fc1(h))))
//
// pos rides in projQKV's bias [T, 3D], on its Q and K thirds: (x + pos) Wᵀ + b =
// x Wᵀ + (b + pos Wᵀ), one row per token, which the loader computes. Q, K and V then
// read x alone, so one projection makes all three.
/////////////////////////////////////////////////////////////////////////////////////////
struct EncoderBlock : public vai::NodeGroup
{
    vai::node::Identity input;
    vai::group::MultiHeadAttention<vai::QkvLayout::Fused> mhAttn;
    vai::node::LayerNorm norm1;
    vai::group::FeedForward ffn;
    vai::node::LayerNorm norm2;

    struct Config
    {
        uint32_t embedDim;
        uint32_t numHeads;
        uint32_t ffnHiddenDim;
        float normEps;
    };

    EncoderBlock(Config config)
    : mhAttn({
          .embedDim = config.embedDim,
          .headDim = config.embedDim / config.numHeads,
          .numQHeads = config.numHeads,
          .acceptResidual = true,
      })
    , norm1({ .eps = config.normEps })
    , ffn({
          .inDim = config.embedDim,
          .hiddenDim = config.ffnHiddenDim,
          .outDim = config.embedDim,
          .activation = vai::Epilogue::Relu,
          .acceptResidual = true,
      })
    , norm2({ .eps = config.normEps })
    {
        // clang-format off
        input -            mhAttn - norm1 -            ffn - norm2;
        input - "residual"/mhAttn;  norm1 - "residual"/ffn;
        // clang-format on

        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", norm2.slot("out0"));
    }
};


}   // namespace model::production::act::group

#endif   // PRODUCTION_ACT_GROUPS_ENCODER_BLOCK_H
