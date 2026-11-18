#ifndef VAI_GROUPS_TRANSFORMER_H
#define VAI_GROUPS_TRANSFORMER_H

#include "multi-head-attention.h"
#include "gated-feed-forward.h"
#include "feed-forward.h"
#include "vai/nodes/identity.h"
#include "vai/nodes/rms-norm.h"
#include "vai/nodes/layer-norm.h"
#include "vai/nodes/gate.h"
#include "vai/nodes/slice.h"
#include "vai/nodes/add.h"
#include <optional>
#include <vector>

namespace vai {


// Where a checkpoint puts its normalizations and its residual adds. The two
// branches (attention, FFN) always exist; only their frame differs, so the
// layout is a type rather than a runtime option.
enum class TransformerType {
    Classic,   // pre-LayerNorm per branch, a plain MLP, the residual folded in
    Default,   // pre-RmsNorm per branch, a gated MLP, the residual folded in
};


// A checkpoint's stack: everything its weights decide.
struct TransformerConfig {
    uint32_t nBlocks;
    uint32_t embedDim;
    uint32_t headDim;
    uint32_t numQHeads;
    uint32_t numKVHeads;   // fewer than numQHeads for GQA
    uint32_t ffnHiddenDim;
    vai::Epilogue ffnActivation;              // the FFN's activation, its GLU variant when gated
    float normEps = 1e-5f;                    // every norm, QK included
    bool withQkNorm = false;                  // per-head RMS norm on Q and K, before RoPE
    std::optional<vai::RopeConfig> ropeCfg;   // absent: no position encoding
    bool ffnFuseGlu = false;                  // gated FFN: the GLU inside its first projection (GatedFeedForward::fuseGlu)
};


}   // namespace vai


namespace vai::group {


// The QKV layout is the attention's business, so it passes straight through to it.
template <TransformerType layout = TransformerType::Default, QkvLayout qkv = QkvLayout::Separate>
struct TransformerBlock;


/////////////////////////////////////////////////////////////////////////////////////////
// TransformerBlock<Classic>, TransformerBlock<Default> - pre-norm block, both
// residuals folded into projections
// Input: in0 [Tq, embedDim], or [B, Tq, embedDim] with allowBatch
// Output: out0 the shape of in0, its Tq 1 under sliceLastRow
//
// out0 = ffn(norm2(h)) + h,  h = mhAttn(norm1(in0)) + in0
//
// Neither add is its own node: the branch's last projection takes the residual
// on its "residual" (e.g. Llama3, PaliGemma's decoder).
//
// Classic is the same frame with the two choices that predate RMS norm and gated
// MLPs: LayerNorm on each branch's input, and one hidden projection rather than
// a gate and a value (e.g. GPT-2).
//
// sliceLastRow drops every row but the last once the attention has run, so the
// FFN and the head that follow it do one row's work. Only the last block of a
// prefill wants it: attention needs the whole prefix, the rest does not.
/////////////////////////////////////////////////////////////////////////////////////////
// clang-format off
template <TransformerType layout> struct BranchNorm;
template <> struct BranchNorm<TransformerType::Classic> { using Type = vai::node::LayerNorm; };
template <> struct BranchNorm<TransformerType::Default> { using Type = vai::node::RmsNorm; };

template <TransformerType layout> struct BranchFfn;
template <> struct BranchFfn<TransformerType::Classic> { using Type = FeedForward; };
template <> struct BranchFfn<TransformerType::Default> { using Type = GatedFeedForward; };
// clang-format on


template <TransformerType layout, QkvLayout qkv>
struct TransformerBlock : public vai::NodeGroup
{
    struct Config
    {
        TransformerConfig base;
        bool allowBatch = false;    // a leading batch axis, kept the whole way
        uint32_t kvCacheSize = 0;   // 0 = no cache
        vai::Dtype kvCacheDtype = vai::Dtype::None;
        bool sliceLastRow = false;    // carry only the last row past the attention
        bool gatingLastOut = false;   // a closed gate drops the FFN and everything downstream
    };

    vai::node::Identity input;
    typename BranchNorm<layout>::Type norm1;
    MultiHeadAttention<qkv> mhAttn;
    std::optional<vai::node::Slice> lastRow;
    std::optional<vai::node::Gate> gate;
    typename BranchNorm<layout>::Type norm2;
    typename BranchFfn<layout>::Type ffn;

    static auto ffnConfig(const TransformerConfig& base)
    {
        if constexpr (layout == TransformerType::Classic)
            return FeedForward::Config{
                .inDim = base.embedDim,
                .hiddenDim = base.ffnHiddenDim,
                .outDim = base.embedDim,
                .activation = base.ffnActivation,
                .acceptResidual = true,
            };
        else
            return GatedFeedForward::Config{
                .inDim = base.embedDim,
                .hiddenDim = base.ffnHiddenDim,
                .outDim = base.embedDim,
                .activation = base.ffnActivation,
                .acceptResidual = true,
                .fuseGlu = base.ffnFuseGlu,
            };
    }

    TransformerBlock(Config config)
    : norm1({ .eps = config.base.normEps })
    , mhAttn({
          .embedDim = config.base.embedDim,
          .headDim = config.base.headDim,
          .numQHeads = config.base.numQHeads,
          .numKVHeads = config.base.numKVHeads,
          .kvCacheSize = config.kvCacheSize,
          .kvCacheDtype = config.kvCacheDtype,
          .qkNormEps = config.base.withQkNorm ? config.base.normEps : 0.0f,
          .ropeCfg = config.base.ropeCfg,
          .allowBatch = config.allowBatch,
          .acceptResidual = true,
      })
    , norm2({ .eps = config.base.normEps })
    , ffn(ffnConfig(config.base))
    {
        if (config.sliceLastRow)
            lastRow.emplace(vai::node::Slice::Config{
                .axis = config.allowBatch ? 1u : 0u,
                .start = -1,
            });
        if (config.gatingLastOut)
            gate.emplace();

        // clang-format off
        input - norm1 -    mhAttn - lastRow - gate/"then" - norm2 -    ffn;
        input - "residual"/mhAttn - lastRow - gate/"then" - "residual"/ffn;
        // clang-format on

        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", ffn.slot("out0"));
    }
};


// The final norm is the one the blocks normalize with.
// clang-format off
template <TransformerType type> struct FinalNorm;
template <> struct FinalNorm<TransformerType::Classic> { using Type = vai::node::LayerNorm; };
template <> struct FinalNorm<TransformerType::Default> { using Type = vai::node::RmsNorm; };
// clang-format on


/////////////////////////////////////////////////////////////////////////////////////////
// Transformer - a stack of blocks and the norm that closes it
// Input:  in0 [T, embedDim], or [B, T, embedDim] with allowBatch
// Output: out0 the shape of in0, its T 1 under sliceLastRow
/////////////////////////////////////////////////////////////////////////////////////////
template<
    TransformerType type = TransformerType::Default,
    QkvLayout qkv = QkvLayout::Separate
>
struct Transformer : public vai::NodeGroup
{
    using Block = TransformerBlock<type, qkv>;

    std::vector<Block> blocks;
    typename FinalNorm<type>::Type finalNorm;

    // sliceLastRow and gatingLastOut apply to the last block.
    struct Config
    {
        TransformerConfig base;
        bool allowBatch = false;
        uint32_t kvCacheSize = 0;
        vai::Dtype kvCacheDtype = vai::Dtype::None;
        bool sliceLastRow = false;
        bool gatingLastOut = false;   // slightly faster when the prefill is chunked
    };

    Transformer(Config config)
    : finalNorm({ .eps = config.base.normEps })
    {
        const uint32_t L = config.base.nBlocks;
        VAI_ASSERT(L >= 1);

        blocks.reserve(L);
        for (uint32_t i = 0; i < L; ++i)
        {
            const bool last = i == L - 1;
            blocks.emplace_back(typename Block::Config{
                .base = config.base,
                .allowBatch = config.allowBatch,
                .kvCacheSize = config.kvCacheSize,
                .kvCacheDtype = config.kvCacheDtype,
                .sliceLastRow = config.sliceLastRow && last,
                .gatingLastOut = config.gatingLastOut && last,
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

    void setCausalMask(eva::Buffer buffer)
    {
        for (auto& block : blocks)
            block.mhAttn.attn.setCausalMask(buffer);
    }

    void setKvCacheCounter(eva::Buffer buffer)
    {
        for (auto& block : blocks)
            block.mhAttn.setKvCacheCounter(buffer);
    }

    // Closing it drops the last block's FFN and everything past it, finalNorm included.
    void setGateOpen(bool open)
    {
        auto& gate = blocks.back().gate;
        if (gate) gate->setOpen(open);
    }
};


}   // namespace vai::group

#endif // VAI_GROUPS_TRANSFORMER_H
