#ifndef VAI_GROUPS_TRANSFORMER_LAYER_H
#define VAI_GROUPS_TRANSFORMER_LAYER_H

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

namespace vai {


// Where a checkpoint puts its normalizations and its residual adds. The two
// branches (attention, FFN) always exist; only their frame differs, so the
// layout is a type rather than a runtime option.
enum class TransformerType {
    Classic,   // pre-LayerNorm per branch, a plain MLP, the residual folded in
    Default,   // pre-RmsNorm per branch, a gated MLP, the residual folded in
};


// Every layout reads the same layer, so one option set serves them all, and a
// model's own layer (e.g. Gemma 3's) reads it too.
struct TransformerLayerConfig {
    bool allowBatch = false;     // true: a leading batch axis, kept the whole way
    uint32_t embedDim;

    uint32_t headDim;
    uint32_t numQHeads;
    uint32_t numKVHeads;         // fewer than numQHeads for GQA
    uint32_t kvCacheSize = 0;    // tokens the KV cache holds; 0 = no cache
    vai::Dtype kvCacheDtype;     // K / V cache
    bool kvCacheRing = false;    // the cache keeps the last kvCacheSize positions (a sliding window's)
    bool withQkNorm = false;     // per-head RMS norm on Q and K, before RoPE
    std::optional<vai::RopeConfig> ropeCfg;   // absent: no position encoding

    uint32_t ffnHiddenDim;
    vai::Epilogue ffnActivation;   // the FFN's activation, its GLU variant when gated

    float normEps = 1e-5f;       // every norm the layer owns, QK included
    bool sliceLastRow = false;   // carry only the last row past the attention
    bool gatingLastOut = false;  // a closed gate drops the FFN and everything downstream
};


}   // namespace vai


namespace vai::group {


// The QKV layout is the attention's business, so it passes straight through to it.
template<
    TransformerType layout = TransformerType::Default,
    QkvLayout qkv = QkvLayout::Separate
> struct TransformerLayer;


/////////////////////////////////////////////////////////////////////////////////////////
// TransformerLayer<Classic>, TransformerLayer<Default> - pre-norm layer, both
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
// FFN and the head that follow it do one row's work. Only the last layer of a
// prefill wants it: attention needs the whole prefix, the rest does not.
/////////////////////////////////////////////////////////////////////////////////////////
template<TransformerType layout> struct BranchNorm;
template<> struct BranchNorm<TransformerType::Classic> { using Type = vai::node::LayerNorm; };
template<> struct BranchNorm<TransformerType::Default> { using Type = vai::node::RmsNorm; };

template<TransformerType layout> struct BranchFfn;
template<> struct BranchFfn<TransformerType::Classic> { using Type = FeedForward; };
template<> struct BranchFfn<TransformerType::Default> { using Type = GatedFeedForward; };


template<TransformerType layout, QkvLayout qkv>
struct TransformerLayer : public vai::NodeGroup
{
    using Config = TransformerLayerConfig;

    vai::node::Identity input;
    typename BranchNorm<layout>::Type norm1;
    MultiHeadAttention<qkv> mhAttn;
    std::optional<vai::node::Slice> lastRow;
    std::optional<vai::node::Gate> gate;
    typename BranchNorm<layout>::Type norm2;
    typename BranchFfn<layout>::Type ffn;

    static auto ffnConfig(const Config& config)
    {
        if constexpr (layout == TransformerType::Classic)
            return FeedForward::Config{
                .inDim = config.embedDim,
                .hiddenDim = config.ffnHiddenDim,
                .outDim = config.embedDim,
                .activation = config.ffnActivation,
                .acceptResidual = true,
            };
        else
            return GatedFeedForward::Config{
                .inDim = config.embedDim,
                .hiddenDim = config.ffnHiddenDim,
                .outDim = config.embedDim,
                .activation = config.ffnActivation,
                .acceptResidual = true,
            };
    }

    TransformerLayer(Config config)
    : norm1({ .eps = config.normEps })
    , mhAttn({
          .embedDim = config.embedDim,
          .headDim = config.headDim,
          .numQHeads = config.numQHeads,
          .numKVHeads = config.numKVHeads,
          .kvCacheSize = config.kvCacheSize,
          .kvCacheDtype = config.kvCacheDtype,
          .kvCacheRing = config.kvCacheRing,
          .qkNormEps = config.withQkNorm ? config.normEps : 0.0f,
          .ropeCfg = config.ropeCfg,
          .allowBatch = config.allowBatch,
          .acceptResidual = true,
      })
    , norm2({ .eps = config.normEps })
    , ffn(ffnConfig(config))
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


}   // namespace vai::group

#endif // VAI_GROUPS_TRANSFORMER_LAYER_H
