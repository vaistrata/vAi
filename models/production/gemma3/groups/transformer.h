#ifndef PRODUCTION_GEMMA3_GROUPS_TRANSFORMER_H
#define PRODUCTION_GEMMA3_GROUPS_TRANSFORMER_H

#include "vai/groups/multi-head-attention.h"
#include "vai/groups/gated-feed-forward.h"
#include "vai/nodes/identity.h"
#include "vai/nodes/rms-norm.h"
#include "vai/nodes/slice.h"
#include "vai/nodes/gate.h"
#include <algorithm>
#include <optional>
#include <vector>

namespace model::production::gemma3 {

struct TransformerConfig
{
    uint32_t nBlocks;
    uint32_t embedDim;
    uint32_t numQHeads;
    uint32_t numKVHeads;
    uint32_t ffnHiddenDim;
    uint32_t slidingWindow;   // a local block's attention window size
    float globalRopeFactor;   // a global block's linear RoPE scale if > 1.0, else 1.0 (default RoPE)
    static constexpr uint32_t headDim = 256;
    static constexpr float normEps = 1e-6f;
};

}   // namespace model::production::gemma3
namespace model::production::gemma3::group {

constexpr bool isGlobalBlock(uint32_t block)
{
    return (block + 1) % 6 == 0;
}


/////////////////////////////////////////////////////////////////////////////////////////
// TransformerBlock - each branch normalized on both ends, residuals outside
// Input: in0 [Tq, embedDim]
// Output: out0 the shape of in0, its Tq 1 under sliceLastRow
//
// out0 = norm4(ffn(norm3(h))) + h,
//     h = norm2(mhAttn(norm1(in0))) + in0
//
// The post-norms sit on the branch output, before it rejoins the residual, so
// the residual stream itself is never rescaled. Each post-norm takes that
// residual and adds it after normalizing; a projection that folded the add in
// would add to a value the norm has not seen yet. Q and K are RMS-normed per
// head, and the FFN gates with GELU (tanh).
/////////////////////////////////////////////////////////////////////////////////////////
struct TransformerBlock : public vai::NodeGroup
{
    vai::node::Identity input;
    vai::node::RmsNorm norm1;
    vai::group::MultiHeadAttention<> mhAttn;
    vai::node::RmsNorm norm2;
    std::optional<vai::node::Slice> lastRow;
    std::optional<vai::node::Gate> gate;
    vai::node::RmsNorm norm3;
    vai::group::GatedFeedForward ffn;
    vai::node::RmsNorm norm4;

    // base, and what differs from block to block.
    struct Config
    {
        TransformerConfig base;
        vai::RopeConfig rope;
        uint32_t kvCacheSize;
        vai::Dtype kvCacheDtype;
        bool kvCacheRing;     // a local block: the cache keeps the last kvCacheSize positions
        bool sliceLastRow;    // carry only the last row past the attention
        bool gatingLastOut;   // a closed gate drops the FFN and everything downstream
    };

    TransformerBlock(Config config)
    : norm1({ .eps = config.base.normEps })
    , mhAttn({
          .embedDim = config.base.embedDim,
          .headDim = config.base.headDim,
          .numQHeads = config.base.numQHeads,
          .numKVHeads = config.base.numKVHeads,
          .kvCacheSize = config.kvCacheSize,
          .kvCacheDtype = config.kvCacheDtype,
          .kvCacheRing = config.kvCacheRing,
          .qkNormEps = config.base.normEps,
          .ropeCfg = config.rope,
      })
    , norm2({ .eps = config.base.normEps, .acceptResidual = true })
    , norm3({ .eps = config.base.normEps })
    , ffn({
          .inDim = config.base.embedDim,
          .hiddenDim = config.base.ffnHiddenDim,
          .outDim = config.base.embedDim,
          .activation = vai::Epilogue::GeluTanh,
      })
    , norm4({ .eps = config.base.normEps, .acceptResidual = true })
    {
        if (config.sliceLastRow)
            lastRow.emplace(vai::node::Slice::Config{ .axis = 0, .start = -1 });
        if (config.gatingLastOut)
            gate.emplace();

        // clang-format off
        input - norm1 - mhAttn - norm2 - lastRow - gate/"then" - norm3 - ffn - norm4;
        input -       "residual"/norm2 - lastRow - gate/"then" -    "residual"/norm4;
        // clang-format on

        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", norm4.slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// Transformer - Gemma 3's decoder: the blocks and the norm that closes it
// Input:  in0 [T, embedDim]
// Output: out0 [1, embedDim], the last row
//
// A global block keeps kvCacheSize positions and reads the global mask; a local
// one keeps the window and a forward past it in a ring (or the whole context
// when that is less) and reads the local mask.
/////////////////////////////////////////////////////////////////////////////////////////
struct Transformer : public vai::NodeGroup
{
    std::vector<TransformerBlock> blocks;
    vai::node::RmsNorm finalNorm;

    struct Config
    {
        TransformerConfig base;
        uint32_t kvCacheSize;
        vai::Dtype kvCacheDtype;
        uint32_t maxForwardTokens;   // the most tokens one forward appends
        bool gatingLastOut;          // slightly faster when the prefill is chunked
    };

    Transformer(Config config)
    : finalNorm({ .eps = config.base.normEps })
    {
        const TransformerConfig& base = config.base;
        const uint32_t ringSize = std::min(base.slidingWindow + config.maxForwardTokens - 1, config.kvCacheSize);
        const vai::RopeConfig localRope = { .type = vai::RopeType::Default, .base = 10000.0f };
        const vai::RopeConfig globalRope =
            base.globalRopeFactor == 1.0f
                ? vai::RopeConfig{ .type = vai::RopeType::Default, .base = 1000000.0f }
                : vai::RopeConfig{ .type = vai::RopeType::Linear, .base = 1000000.0f, .factor = base.globalRopeFactor };

        blocks.reserve(base.nBlocks);
        for (uint32_t i = 0; i < base.nBlocks; ++i)
        {
            const bool global = isGlobalBlock(i);
            const bool last = i == base.nBlocks - 1;
            blocks.emplace_back(TransformerBlock::Config{
                .base = base,
                .rope = global ? globalRope : localRope,
                .kvCacheSize = global ? config.kvCacheSize : ringSize,
                .kvCacheDtype = config.kvCacheDtype,
                .kvCacheRing = !global,
                .sliceLastRow = last,
                .gatingLastOut = last && config.gatingLastOut,
            });
        }

        // clang-format off
        for (size_t i = 1; i < blocks.size(); ++i)
            blocks[i - 1] - blocks[i];
        blocks.back() - finalNorm;
        // clang-format on

        defineSlot("in0", blocks[0].slot("in0"));
        defineSlot("out0", finalNorm.slot("out0"));
    }

    void setKvCacheCounter(eva::Buffer buffer)
    {
        for (auto& block : blocks)
            block.mhAttn.setKvCacheCounter(buffer);
    }

    void setCausalMask(eva::Buffer globalMask, eva::Buffer localMask)
    {
        for (auto& block : blocks)
            block.mhAttn.attn.setCausalMask(block.mhAttn.attn.config_.kvCacheRing ? localMask : globalMask);
    }

    // Closing it drops the last block's FFN and everything past it, finalNorm included.
    void setGateOpen(bool open)
    {
        if (auto& gate = blocks.back().gate)
            gate->setOpen(open);
    }
};


}   // namespace model::production::gemma3::group

#endif   // PRODUCTION_GEMMA3_GROUPS_TRANSFORMER_H
