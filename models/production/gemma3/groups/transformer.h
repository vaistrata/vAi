#ifndef PRODUCTION_GEMMA3_GROUPS_TRANSFORMER_H
#define PRODUCTION_GEMMA3_GROUPS_TRANSFORMER_H

#include "vai/groups/multi-head-attention.h"
#include "vai/groups/gated-feed-forward.h"
#include "vai/nodes/identity.h"
#include "vai/nodes/rms-norm.h"
#include "vai/nodes/slice.h"
#include "vai/nodes/gate.h"
#include <optional>

namespace model::production::gemma3::group {


/////////////////////////////////////////////////////////////////////////////////////////
// TransformerLayer - each branch normalized on both ends, residuals outside
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
struct TransformerLayer : public vai::NodeGroup
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

    struct Config
    {
        uint32_t embedDim;
        uint32_t headDim;
        uint32_t numQHeads;
        uint32_t numKVHeads;
        uint32_t ffnHiddenDim;
        float normEps;   // every norm the layer owns, QK included
        vai::RopeConfig rope;
        uint32_t kvCacheSize;
        vai::Dtype kvCacheDtype;
        bool kvCacheRing;     // a local layer: the cache keeps the last kvCacheSize positions
        bool sliceLastRow;    // carry only the last row past the attention
        bool gatingLastOut;   // a closed gate drops the FFN and everything downstream
    };

    TransformerLayer(Config config)
        : norm1({.eps = config.normEps}), mhAttn({.embedDim = config.embedDim,
                                                  .headDim = config.headDim,   // 256
                                                  .numQHeads = config.numQHeads,
                                                  .numKVHeads = config.numKVHeads,
                                                  .kvCacheSize = config.kvCacheSize,
                                                  .kvCacheDtype = config.kvCacheDtype,
                                                  .kvCacheRing = config.kvCacheRing,
                                                  .qkNormEps = config.normEps,   // 1e-6f
                                                  .ropeCfg = config.rope}),
          norm2({.eps = config.normEps, .acceptResidual = true}), norm3({.eps = config.normEps}),
          ffn({.inDim = config.embedDim,
               .hiddenDim = config.ffnHiddenDim,
               .outDim = config.embedDim,
               .activation = vai::Epilogue::GeluTanh}),
          norm4({.eps = config.normEps, .acceptResidual = true})
    {
        if (config.sliceLastRow)
            lastRow.emplace(vai::node::Slice::Config{.axis = 0, .start = -1});
        if (config.gatingLastOut)
            gate.emplace();

        input - norm1 - mhAttn - norm2 - lastRow - gate / "then" - norm3 - ffn - norm4;
        input - "residual" / norm2 - lastRow - gate / "then" - "residual" / norm4;

        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", norm4.slot("out0"));
    }
};


}   // namespace model::production::gemma3::group

#endif   // PRODUCTION_GEMMA3_GROUPS_TRANSFORMER_H
