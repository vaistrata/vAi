#ifndef EXPERIMENTAL_RTDETRV2_GROUPS_MULTI_HEAD_ATTENTION_H
#define EXPERIMENTAL_RTDETRV2_GROUPS_MULTI_HEAD_ATTENTION_H

#include "../nodes/add.h"
#include "vai/nodes/attention.h"
#include "vai/nodes/identity.h"
#include "vai/nodes/linear.h"
#include "vai/nodes/permute.h"
#include "vai/nodes/reshape.h"
#include "vai/nodes/rms-norm.h"
#include "vai/nodes/rope.h"
#include "vai/nodes/slice.h"

namespace model::experimental::rtdetrv2 {


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
    uint32_t numQHeads = 0;     // 0 -> embedDim / headDim
    uint32_t numKVHeads = 0;    // for GQA; 0 -> Hkv = numQHeads
    uint32_t kvCacheSize = 0;   // tokens the KV cache holds; 0 = no cache
    vai::Dtype kvCacheDtype = vai::Dtype::None;
    bool kvCacheRing = false;
    float qkNormEps = 0.0f;                    // per-head RMS norm on Q and K, before RoPE; 0 = none
    std::optional<vai::RopeConfig> ropeCfg;    // absent: no position encoding (e.g. a ViT)
    std::optional<node::Add::Config> posCfg;   // Separate only: add position before Q/K projections
    bool acceptResidual = false;               // open "residual" and fold its addend into the output projection
};
}   // namespace model::experimental::rtdetrv2


namespace model::experimental::rtdetrv2::group {


// Declared, never defined: a layout that has no specialization fails to compile
// rather than falling back to something plausible.
template <QkvLayout layout>
struct MultiHeadAttention;


/////////////////////////////////////////////////////////////////////////////////////////
// MultiHeadAttention<Separate> - GQA, one projection per operand
// Input: in0 [B, Tq, embedDim], optional residual [B, Tq, embedDim]
// Output: out0 [B, Tq, embedDim]
//
// B sequences share the projections and attend apart. A KV cache holds one
// position counter for every sequence, so a cached block wants B = 1.
//
// Q/K share a projection and are split by head; RoPE and the QK norm reach them without
// touching V. Reshape and Permute expose the head-major layout.
//
// With posCfg, position [B, Tq, embedDim] is added only to the Q/K input.
// V keeps the original input. constantIn1 stores position as a persistent tensor;
// otherwise position is a graph input. An absent posCfg bypasses the add.
//
// The caller owns the KV-cache counter and hands it to attn and the RoPEs:
// they read the same position.
/////////////////////////////////////////////////////////////////////////////////////////
template <>
struct MultiHeadAttention<QkvLayout::Separate> : public vai::NodeGroup
{
    using Config = MultiHeadAttentionConfig;

    uint32_t D;     // embedding dimension
    uint32_t Dh;    // head dimension
    uint32_t Hq;    // number of query heads
    uint32_t Hkv;   // number of KV heads

    vai::node::Identity input;
    std::optional<node::Add> posAdd;
    vai::node::Linear toQK, toV;   // Q/K share the position-adjusted input.
    vai::node::Reshape splitQK, splitV;
    vai::node::Permute perQK, perV;
    vai::node::Slice splitQ, splitK;   // Contiguous head ranges when B == 1.
    std::optional<vai::node::RmsNorm> normQ, normK;
    std::optional<vai::node::Rope> ropeQ, ropeK;
    vai::node::Attention attn;     //    -> [B, Tq, Hq, Dh]
    vai::node::Reshape mergeOut;   //    -> [B, Tq, Hq*Dh]
    vai::node::Linear toOut;       // [B, Tq, Hq*Dh] -> [B, Tq, D]

    MultiHeadAttention(Config config)
    : D(config.embedDim)
    , Dh(config.headDim)
    , Hq(config.numQHeads ? config.numQHeads : config.embedDim / config.headDim)
    , Hkv(config.numKVHeads ? config.numKVHeads : Hq)
    , toQK({ .inDim = D, .outDim = (Hq + Hkv) * Dh })
    , toV({ .inDim = D, .outDim = Hkv * Dh })
    , splitQK({ Hq + Hkv, Dh }, 1)
    , splitV({ Hkv, Dh }, 1)
    , perQK({ 0, 2, 1, 3 })
    , perV({ 0, 2, 1, 3 })
    , splitQ({ .axis = 1, .start = 0, .count = Hq })
    , splitK({ .axis = 1, .start = Hq, .count = Hkv })
    , attn({ .headDim = Dh,
             .numQHeads = Hq,
             .numKVHeads = Hkv,
             .kvCacheSize = config.kvCacheSize,
             .kvCacheDtype = config.kvCacheDtype,
             .kvCacheRing = config.kvCacheRing })
    , mergeOut({ Hq * Dh }, 2)
    , toOut({ .inDim = Hq * Dh, .outDim = D, .acceptResidual = config.acceptResidual })
    {
        // Built in place: a node's slots point back at it, so it cannot be moved in.
        if (config.posCfg)
            posAdd.emplace(*config.posCfg);
        if (config.qkNormEps > 0.0f)
        {
            normQ.emplace(vai::node::RmsNorm::Config{ .eps = config.qkNormEps });
            normK.emplace(vai::node::RmsNorm::Config{ .eps = config.qkNormEps });
        }
        if (config.ropeCfg)
        {
            ropeQ.emplace(vai::node::Rope::Config{ .cfg = *config.ropeCfg, .headDim = Dh });
            ropeK.emplace(vai::node::Rope::Config{ .cfg = *config.ropeCfg, .headDim = Dh });
        }

        input - posAdd - toQK - splitQK - perQK;
        perQK - splitQ - normQ - ropeQ - "Q" / attn;
        perQK - splitK - normK - ropeK - "K" / attn;
        input - toV - splitV - perV - "V" / attn - mergeOut - toOut;

        defineSlot("in0", input.slot("in0"));
        if (posAdd)
            defineSlot("position", posAdd->slot("in1"));
        if (config.acceptResidual)
            defineSlot("residual", toOut.slot("residual"));
        defineSlot("out0", toOut.slot("out0"));
    }

    // RoPE and the cache sweep read the same position, so one counter drives them all.
    void setKvCacheCounter(eva::Buffer buffer)
    {
        attn.setKvCacheCounter(buffer);
        if (ropeQ)
            ropeQ->setPosParam(buffer);
        if (ropeK)
            ropeK->setPosParam(buffer);
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// MultiHeadAttention<Fused> - GQA, one projection over Q, K and V together
// Input: in0 [B, Tq, embedDim], optional residual [B, Tq, embedDim]
// Output: out0 [B, Tq, embedDim]
//
// The checkpoint stacks the three projections into one weight (e.g. GPT-2's
// c_attn), so the heads come out interleaved and a row slice per operand takes
// them apart: Q is [0, Hq), K is [Hq, Hq + Hkv), V the rest.
//
// The split happens before RoPE and the QK norm, which touch Q and K alone.
//
// The caller owns the KV-cache counter and hands it to attn and the RoPEs:
// they read the same position.
/////////////////////////////////////////////////////////////////////////////////////////
template <>
struct MultiHeadAttention<QkvLayout::Fused> : public vai::NodeGroup
{
    using Config = MultiHeadAttentionConfig;

    uint32_t D;     // embedding dimension
    uint32_t Dh;    // head dimension
    uint32_t Hq;    // number of query heads
    uint32_t Hkv;   // number of KV heads

    vai::node::Linear toQKV;                   // [B, Tq, D] -> [B, Tq, (Hq + 2*Hkv)*Dh]
    vai::node::Reshape splitHeads;             //            -> [B, Tq, Hq + 2*Hkv, Dh]
    vai::node::Permute perQKV;                 //            -> [B, Hq + 2*Hkv, Tq, Dh]
    vai::node::Slice sliceQ, sliceK, sliceV;   //  Q: [B, Hq, Tq, Dh],  K/V: [B, Hkv, Tq, Dh]
    std::optional<vai::node::RmsNorm> normQ, normK;
    std::optional<vai::node::Rope> ropeQ, ropeK;
    vai::node::Attention attn;     //            -> [B, Tq, Hq, Dh]
    vai::node::Reshape mergeOut;   //            -> [B, Tq, Hq*Dh]
    vai::node::Linear toOut;       // [B, Tq, Hq*Dh] -> [B, Tq, D]

    MultiHeadAttention(Config config)
    : D(config.embedDim)
    , Dh(config.headDim)
    , Hq(config.numQHeads ? config.numQHeads : config.embedDim / config.headDim)
    , Hkv(config.numKVHeads ? config.numKVHeads : Hq)
    , toQKV({ .inDim = D, .outDim = (Hq + 2 * Hkv) * Dh })
    , splitHeads({ Hq + 2 * Hkv, Dh }, 1)
    , perQKV({ 0, 2, 1, 3 })
    , sliceQ({ .axis = 1, .start = 0, .count = Hq })
    , sliceK({ .axis = 1, .start = Hq, .count = Hkv })
    , sliceV({ .axis = 1, .start = Hq + Hkv, .count = Hkv })
    , attn({ .headDim = Dh,
             .numQHeads = Hq,
             .numKVHeads = Hkv,
             .kvCacheSize = config.kvCacheSize,
             .kvCacheDtype = config.kvCacheDtype,
             .kvCacheRing = config.kvCacheRing })
    , mergeOut({ Hq * Dh }, 2)
    , toOut({ .inDim = Hq * Dh, .outDim = D, .acceptResidual = config.acceptResidual })
    {
        // Built in place: a node's slots point back at it, so it cannot be moved in.
        if (config.qkNormEps > 0.0f)
        {
            normQ.emplace(vai::node::RmsNorm::Config{ .eps = config.qkNormEps });
            normK.emplace(vai::node::RmsNorm::Config{ .eps = config.qkNormEps });
        }
        if (config.ropeCfg)
        {
            ropeQ.emplace(vai::node::Rope::Config{ .cfg = *config.ropeCfg, .headDim = Dh });
            ropeK.emplace(vai::node::Rope::Config{ .cfg = *config.ropeCfg, .headDim = Dh });
        }

        toQKV - splitHeads - perQKV - sliceQ - normQ - ropeQ - "Q" / attn;
        perQKV - sliceK - normK - ropeK - "K" / attn;
        perQKV - sliceV - "V" / attn - mergeOut - toOut;

        defineSlot("in0", toQKV.slot("in0"));
        if (config.acceptResidual)
            defineSlot("residual", toOut.slot("residual"));
        defineSlot("out0", toOut.slot("out0"));
    }

    // RoPE and the cache sweep read the same position, so one counter drives them all.
    void setKvCacheCounter(eva::Buffer buffer)
    {
        attn.setKvCacheCounter(buffer);
        if (ropeQ)
            ropeQ->setPosParam(buffer);
        if (ropeK)
            ropeK->setPosParam(buffer);
    }
};
}   // namespace model::experimental::rtdetrv2::group

#endif   // EXPERIMENTAL_RTDETRV2_GROUPS_MULTI_HEAD_ATTENTION_H
