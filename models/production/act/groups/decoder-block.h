#ifndef PRODUCTION_ACT_GROUPS_DECODER_BLOCK_H
#define PRODUCTION_ACT_GROUPS_DECODER_BLOCK_H

#include "vai/groups/feed-forward.h"
#include "vai/nodes/identity.h"
#include "vai/nodes/linear.h"
#include "vai/nodes/reshape.h"
#include "vai/nodes/permute.h"
#include "vai/nodes/slice.h"
#include "vai/nodes/attention.h"
#include "vai/nodes/layer-norm.h"


namespace model::production::act::group {


/////////////////////////////////////////////////////////////////////////////////////////
// DecoderBlock - the cross-attention and FFN half of ACT's one decoder block
// Input : memory [Tk, D], the encoder output
//         query  [H, Tq, Dh], the projected queries split per head
// Output: out0   [Tq, D]
//
// h = norm1(x1 + attn(q, k = memory + pos, v = memory)),  out0 = norm2(h + fc2(relu(fc1(h))))
//
// pos rides in projKV's bias [Tk, 2D], on its K half, as in EncoderBlock. K and V then
// read memory alone, so one projection makes both.
//
// The decoder starts from zeros, so its self-attention and the norm after it depend on nothing
// but weights: the loader computes their output x1 once, projects x1 + the query
// positions into "query", and folds x1 into projOut's bias.
/////////////////////////////////////////////////////////////////////////////////////////
struct DecoderBlock : public vai::NodeGroup
{
    vai::node::Identity memory;
    vai::node::Identity query;
    vai::node::Linear projKV;          // [Tk, 2D]
    vai::node::Reshape splitKV;        // [Tk, 2H, Dh]
    vai::node::Permute perKV;          // [2H, Tk, Dh]
    vai::node::Slice sliceK, sliceV;   // [H, Tk, Dh]
    vai::node::Attention attn;         // [Tq, H, Dh]
    vai::node::Reshape mergeOut;       // [Tq, D]
    vai::node::Linear projOut;
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

    DecoderBlock(Config config)
    : projKV({ .inDim = config.embedDim, .outDim = 2 * config.embedDim })
    , splitKV({ -1, int64_t(2 * config.numHeads), int64_t(config.embedDim / config.numHeads) })
    , perKV({ 1, 0, 2 })
    , sliceK({ .axis = 0, .start = 0, .count = config.numHeads })
    , sliceV({ .axis = 0, .start = config.numHeads, .count = config.numHeads })
    , attn({ .headDim = config.embedDim / config.numHeads, .numQHeads = config.numHeads })
    , mergeOut({ -1, int64_t(config.embedDim) })
    , projOut({ .inDim = config.embedDim, .outDim = config.embedDim })
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
        query                                      - "Q"/attn;
        memory - projKV - splitKV - perKV - sliceK - "K"/attn;
        perKV - sliceV - "V"/attn - mergeOut - projOut - norm1 -            ffn - norm2;
                                                         norm1 - "residual"/ffn;
        // clang-format on

        defineSlot("memory", memory.slot("in0"));
        defineSlot("query", query.slot("in0"));
        defineSlot("out0", norm2.slot("out0"));
    }
};


}   // namespace model::production::act::group

#endif   // PRODUCTION_ACT_GROUPS_DECODER_BLOCK_H
