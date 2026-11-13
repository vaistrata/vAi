#ifndef VAI_GROUPS_MULTI_HEAD_ATTENTION_H
#define VAI_GROUPS_MULTI_HEAD_ATTENTION_H

#include "vai/nodes/identity.h"
#include "vai/nodes/linear.h"
#include "vai/nodes/reshape.h"
#include "vai/nodes/permute.h"
#include "vai/nodes/slice.h"
#include "vai/nodes/rms-norm.h"
#include "vai/nodes/rope.h"
#include "vai/nodes/attention.h"

namespace vai {


// How a checkpoint stores the weights that produce Q, K and V. It decides the
// shape of the projection, not what happens after it, so the two layouts carry
// the same config and differ only in how they arrive at three operands.
enum class QkvLayout {
    Separate,   // one projection per operand
    Fused,      // one projection over the three, sliced by head
};


// Both layouts read the same block. A checkpoint that ties headDim to embedDim /
// numQHeads can leave numQHeads at 0, and one without position encoding (a ViT)
// leaves ropeCfg empty.
struct MultiHeadAttentionConfig
{
    uint32_t embedDim;
    uint32_t headDim;
    uint32_t numQHeads = 0;      // 0 -> embedDim / headDim
    uint32_t numKVHeads = 0;     // for GQA; 0 -> Hkv = numQHeads
    uint32_t kvCacheSize = 0;    // tokens the KV cache holds; 0 = no cache
    vai::Dtype kvCacheDtype = vai::Dtype::None;   // K / V cache; required when kvCacheSize > 0
    bool kvCacheRing = false;                     // the cache keeps the last kvCacheSize positions (a sliding window's)
    float qkNormEps = 0.0f;      // per-head RMS norm on Q and K, before RoPE; 0 = none
    std::optional<vai::RopeConfig> ropeCfg;   // absent: no position encoding (e.g. a ViT)
    bool crossAttention = false;              // true: Q / K / V open as separate inputs in place of in0 (Separate only)
    bool allowBatch = false;                  // true: a leading batch axis, kept the whole way
    bool acceptResidual = false; // open "residual" and fold its addend into the output projection
};


}   // namespace vai


namespace vai::group {


// Declared, never defined: a layout that has no specialization fails to compile
// rather than falling back to something plausible.
template <QkvLayout layout = QkvLayout::Separate>
struct MultiHeadAttention;


/////////////////////////////////////////////////////////////////////////////////////////
// MultiHeadAttention<Separate> - GQA, one projection per operand
// Input: in0 [Tq, embedDim], or [B, Tq, embedDim] with allowBatch;
//        Q / K / V of that shape each in place of in0 with crossAttention;
//        optional residual of the same shape
// Output: out0 the shape of in0
//
// allowBatch's B sequences share the projections and attend apart. A KV cache holds
// one position counter for every sequence, so a cached block wants B = 1.
//
// Q, K and V are projected apart, so RoPE and the QK norm reach Q and K without
// touching V.
//
// The caller owns the KV-cache counter and hands it to attn and the RoPEs:
// they read the same position.
/////////////////////////////////////////////////////////////////////////////////////////
template<>
struct MultiHeadAttention<QkvLayout::Separate> : public vai::NodeGroup
{
    using Config = MultiHeadAttentionConfig;

    uint32_t D;       // embedding dimension
    uint32_t Dh;      // head dimension
    uint32_t Hq;      // number of query heads
    uint32_t Hkv;     // number of KV heads

    std::optional<vai::node::Identity> input;   // in0, absent with crossAttention
    vai::node::Linear projQ, projK, projV;      // Q: [B, Tq, D] -> [B, Tq, Hq*Dh],  K/V: -> [B, Tq, Hkv*Dh]
    vai::node::Reshape splitQ, splitK, splitV;  // Q: [B, Tq, Hq, Dh],               K/V: [B, Tq, Hkv, Dh]
    std::optional<vai::node::RmsNorm> normQ, normK;
    vai::node::Permute perQ, perK, perV;        // Q: -> [B, Hq, Tq, Dh],            K/V: -> [B, Hkv, Tq, Dh]
    std::optional<vai::node::Rope> ropeQ, ropeK;
    vai::node::Attention attn;                  //    -> [B, Tq, Hq, Dh]
    vai::node::Reshape mergeOut;                //    -> [B, Tq, Hq*Dh]
    vai::node::Linear projOut;                  // [B, Tq, Hq*Dh] -> [B, Tq, D]

    MultiHeadAttention(Config config)
    : D(config.embedDim)
    , Dh(config.headDim)
    , Hq(config.numQHeads ? config.numQHeads : config.embedDim / config.headDim)
    , Hkv(config.numKVHeads ? config.numKVHeads : Hq)
    , projQ({ .inDim = D, .outDim = Hq * Dh })
    , projK({ .inDim = D, .outDim = Hkv * Dh })
    , projV({ .inDim = D, .outDim = Hkv * Dh })
    , splitQ(config.allowBatch ? std::vector<int64_t>{ 0, -1, Hq, Dh } : std::vector<int64_t>{ -1, Hq, Dh })
    , splitK(config.allowBatch ? std::vector<int64_t>{ 0, -1, Hkv, Dh } : std::vector<int64_t>{ -1, Hkv, Dh })
    , splitV(config.allowBatch ? std::vector<int64_t>{ 0, -1, Hkv, Dh } : std::vector<int64_t>{ -1, Hkv, Dh })
    , perQ(config.allowBatch ? std::vector<uint32_t>{ 0, 2, 1, 3 } : std::vector<uint32_t>{ 1, 0, 2 })
    , perK(config.allowBatch ? std::vector<uint32_t>{ 0, 2, 1, 3 } : std::vector<uint32_t>{ 1, 0, 2 })
    , perV(config.allowBatch ? std::vector<uint32_t>{ 0, 2, 1, 3 } : std::vector<uint32_t>{ 1, 0, 2 })
    , attn({
          .headDim = Dh,
          .numQHeads = Hq,
          .numKVHeads = Hkv,
          .kvCacheSize = config.kvCacheSize,
          .kvCacheDtype = config.kvCacheDtype,
          .kvCacheRing = config.kvCacheRing,
      })
    , mergeOut(config.allowBatch ? std::vector<int64_t>{ 0, -1, Hq * Dh } : std::vector<int64_t>{ -1, Hq * Dh })
    , projOut({ .inDim = Hq * Dh, .outDim = D, .acceptResidual = config.acceptResidual })
    {
        if (!config.crossAttention)
            input.emplace();

        if (config.qkNormEps > 0.0f)
        {
            normQ.emplace(vai::node::RmsNorm::Config{.eps = config.qkNormEps});
            normK.emplace(vai::node::RmsNorm::Config{.eps = config.qkNormEps});
        }

        if (config.ropeCfg)
        {
            ropeQ.emplace(vai::node::Rope::Config{.cfg = *config.ropeCfg, .headDim = Dh});
            ropeK.emplace(vai::node::Rope::Config{.cfg = *config.ropeCfg, .headDim = Dh});
        }

        // clang-format off
        input - projQ - splitQ - normQ - perQ - ropeQ - "Q"/attn;
        input - projK - splitK - normK - perK - ropeK - "K"/attn;
        input - projV - splitV         - perV         - "V"/attn - mergeOut - projOut;
        // clang-format on

        if (input)
            defineSlot("in0", input->slot("in0"));
        else
        {
            defineSlot("Q", projQ.slot("in0"));
            defineSlot("K", projK.slot("in0"));
            defineSlot("V", projV.slot("in0"));
        }

        if (config.acceptResidual)
            defineSlot("residual", projOut.slot("residual"));

        defineSlot("out0", projOut.slot("out0"));
    }

    void setKvCacheCounter(eva::Buffer buffer)
    {
        attn.setKvCacheCounter(buffer);
        if (ropeQ) ropeQ->setKvCacheCounter(buffer);
        if (ropeK) ropeK->setKvCacheCounter(buffer);
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// MultiHeadAttention<Fused> - GQA, one projection over Q, K and V together
// Input: in0 [Tq, embedDim], or [B, Tq, embedDim] with allowBatch;
//        optional residual of the same shape
// Output: out0 the shape of in0
//
// The checkpoint stacks the three projections into one weight (e.g. GPT-2's
// c_attn), so the heads come out interleaved and a row slice per operand takes
// them apart: Q is [0, Hq), K is [Hq, Hq + Hkv), V the rest.
//
// The split has to happen before RoPE and the QK norm, which touch Q and K
// alone.
//
// The caller owns the KV-cache counter and hands it to attn and the RoPEs:
// they read the same position.
/////////////////////////////////////////////////////////////////////////////////////////
template<>
struct MultiHeadAttention<QkvLayout::Fused> : public vai::NodeGroup
{
    using Config = MultiHeadAttentionConfig;

    uint32_t D;       // embedding dimension
    uint32_t Dh;      // head dimension
    uint32_t Hq;      // number of query heads
    uint32_t Hkv;     // number of KV heads

    vai::node::Linear projQKV;                     // [B, Tq, D] -> [B, Tq, (Hq + 2*Hkv)*Dh]
    vai::node::Reshape splitHeads;                 //            -> [B, Tq, Hq + 2*Hkv, Dh]
    vai::node::Permute perQKV;                     //            -> [B, Hq + 2*Hkv, Tq, Dh]
    vai::node::Slice sliceQ, sliceK, sliceV;       //  Q: [B, Hq, Tq, Dh],  K/V: [B, Hkv, Tq, Dh]
    std::optional<vai::node::RmsNorm> normQ, normK;
    std::optional<vai::node::Rope> ropeQ, ropeK;
    vai::node::Attention attn;                     //            -> [B, Tq, Hq, Dh]
    vai::node::Reshape mergeOut;                   //            -> [B, Tq, Hq*Dh]
    vai::node::Linear projOut;                     // [B, Tq, Hq*Dh] -> [B, Tq, D]

    MultiHeadAttention(Config config)
    : D(config.embedDim)
    , Dh(config.headDim)
    , Hq(config.numQHeads ? config.numQHeads : config.embedDim / config.headDim)
    , Hkv(config.numKVHeads ? config.numKVHeads : Hq)
    , projQKV({ .inDim = D, .outDim = (Hq + 2 * Hkv) * Dh })
    , splitHeads(config.allowBatch ? std::vector<int64_t>{ 0, -1, Hq + 2 * Hkv, Dh }
                                   : std::vector<int64_t>{ -1, Hq + 2 * Hkv, Dh })
    , perQKV(config.allowBatch ? std::vector<uint32_t>{ 0, 2, 1, 3 } : std::vector<uint32_t>{ 1, 0, 2 })
    , sliceQ({ .axis = config.allowBatch ? 1u : 0u, .start = 0, .count = Hq })
    , sliceK({ .axis = config.allowBatch ? 1u : 0u, .start = Hq, .count = Hkv })
    , sliceV({ .axis = config.allowBatch ? 1u : 0u, .start = Hq + Hkv, .count = Hkv })
    , attn({
          .headDim = Dh,
          .numQHeads = Hq,
          .numKVHeads = Hkv,
          .kvCacheSize = config.kvCacheSize,
          .kvCacheDtype = config.kvCacheDtype,
          .kvCacheRing = config.kvCacheRing,
      })
    , mergeOut(config.allowBatch ? std::vector<int64_t>{ 0, -1, Hq * Dh } : std::vector<int64_t>{ -1, Hq * Dh })
    , projOut({ .inDim = Hq * Dh, .outDim = D, .acceptResidual = config.acceptResidual })
    {
        // One projection reads one input, so it has no Q / K / V of its own.
        VAI_ASSERT(!config.crossAttention);

        // Built in place: a node's slots point back at it, so it cannot be moved in.
        if (config.qkNormEps > 0.0f)
        {
            normQ.emplace(vai::node::RmsNorm::Config{.eps = config.qkNormEps});
            normK.emplace(vai::node::RmsNorm::Config{.eps = config.qkNormEps});
        }

        if (config.ropeCfg)
        {
            ropeQ.emplace(vai::node::Rope::Config{.cfg = *config.ropeCfg, .headDim = Dh});
            ropeK.emplace(vai::node::Rope::Config{.cfg = *config.ropeCfg, .headDim = Dh});
        }

        // clang-format off
        projQKV - splitHeads - perQKV - sliceQ - normQ - ropeQ - "Q"/attn;
        perQKV - sliceK - normK - ropeK - "K"/attn;
        perQKV - sliceV                 - "V"/attn - mergeOut - projOut;
        // clang-format on

        defineSlot("in0", projQKV.slot("in0"));

        if (config.acceptResidual)
            defineSlot("residual", projOut.slot("residual"));

        defineSlot("out0", projOut.slot("out0"));
    }

    void setKvCacheCounter(eva::Buffer buffer)
    {
        attn.setKvCacheCounter(buffer);
        if (ropeQ) ropeQ->setKvCacheCounter(buffer);
        if (ropeK) ropeK->setKvCacheCounter(buffer);
    }
};


}   // namespace vai::group

#endif // VAI_GROUPS_MULTI_HEAD_ATTENTION_H
