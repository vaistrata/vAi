#ifndef PRODUCTION_PI05_GROUPS_ACTION_DECODER_H
#define PRODUCTION_PI05_GROUPS_ACTION_DECODER_H

#include "vai/groups/transformer.h"
#include "vai/nodes/identity.h"
#include "vai/nodes/rms-norm.h"
#include "vai/nodes/slice.h"
#include "vai/nodes/ele-affine.h"
#include <array>
#include <vector>

namespace model::production::pi05::group {


/////////////////////////////////////////////////////////////////////////////////////////
// ActionTransformerBlock - Gemma's block, its norms and residuals set by a condition
// Input: in0  [Tq, embedDim]
//        cond [6, embedDim]   scale1 shift1 g1 scale2 shift2 g2, one row each, shared by every token
// Output: out0 the shape of in0
//
// out0 = h + g2 * ffn(norm2(h)),  h = in0 + g1 * mhAttn(norm1(in0))
// norm(x) = rms(x) * scale + shift
//
// scale, shift and g come from the flow-matching timestep rather than the checkpoint,
// so the norms read them as inputs and each gated residual is an EleAffine
// (g * branch + x): a projection that folded the residual in would add before the
// gate has scaled the branch.
/////////////////////////////////////////////////////////////////////////////////////////
struct ActionTransformerBlock : public vai::NodeGroup
{
    struct Config
    {
        vai::TransformerConfig base;
        uint32_t kvCacheSize;
        vai::Dtype kvCacheDtype;
    };

    vai::node::Identity input;
    vai::node::Identity cond;
    std::array<vai::node::Slice, 6> condRows;   // [scale1], [shift1], [g1], [scale2], [shift2], [g2]
    vai::node::RmsNorm norm1;
    vai::group::MultiHeadAttention<> mhAttn;
    vai::node::EleAffine fma1;
    vai::node::RmsNorm norm2;
    vai::group::GatedFeedForward ffn;
    vai::node::EleAffine fma2;

    ActionTransformerBlock(Config config)
    : condRows{ { { { .start = 0 } },
                  { { .start = 1 } },
                  { { .start = 2 } },
                  { { .start = 3 } },
                  { { .start = 4 } },
                  { { .start = 5 } } }, }
    , norm1({ .eps = config.base.normEps, .coeffsAsInputs = true })
    , mhAttn({
          .embedDim = config.base.embedDim,
          .headDim = config.base.headDim,
          .numQHeads = config.base.numQHeads,
          .numKVHeads = config.base.numKVHeads,
          .kvCacheSize = config.kvCacheSize,
          .kvCacheDtype = config.kvCacheDtype,
          .qkNormEps = config.base.withQkNorm ? config.base.normEps : 0.0f,
          .ropeCfg = config.base.ropeCfg,
      })
    , fma1({ .scaleAsInput = true, .shiftAsInput = true })
    , norm2({ .eps = config.base.normEps, .coeffsAsInputs = true })
    , ffn({
          .inDim = config.base.embedDim,
          .hiddenDim = config.base.ffnHiddenDim,
          .outDim = config.base.embedDim,
          .actType = config.base.ffnActType,
          .fuseGlu = config.base.ffnFuseGlu,
      })
    , fma2({ .scaleAsInput = true, .shiftAsInput = true })
    {
        // clang-format off
        cond - condRows[0] - "scale"/norm1;
        cond - condRows[1] - "shift"/norm1;
        cond - condRows[2] - "scale"/fma1;
        cond - condRows[3] - "scale"/norm2;
        cond - condRows[4] - "shift"/norm2;
        cond - condRows[5] - "scale"/fma2;

        input - norm1 - mhAttn - "in0"/fma1 - norm2 - ffn - "in0"/fma2;
        input -                "shift"/fma1 -             "shift"/fma2;
        // clang-format on

        defineSlot("in0", input.slot("in0"));
        defineSlot("cond", cond.slot("in0"));
        defineSlot("out0", fma2.slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// ActionDecoder - pi0.5's action expert: a stack of ActionTransformerBlock and
// the conditioned norm that closes it
// Input:  in0  [T, embedDim]
//         cond [6 * nBlocks + 2, embedDim]   each block's six rows in order, then
//              scale and shift of the final norm
// Output: out0 the shape of in0
/////////////////////////////////////////////////////////////////////////////////////////
struct ActionDecoder : public vai::NodeGroup
{
    using Block = ActionTransformerBlock;

    std::vector<Block> blocks;
    vai::node::RmsNorm finalNorm;

    vai::node::Identity cond;
    std::vector<vai::node::Slice> condSubTables;   // each block's [6], then the final norm's [scale], [shift]

    struct Config
    {
        vai::TransformerConfig base;
        uint32_t kvCacheSize;
        vai::Dtype kvCacheDtype;
    };

    ActionDecoder(Config config)
    : finalNorm({ .eps = config.base.normEps, .coeffsAsInputs = true })
    {
        const uint32_t L = config.base.nBlocks;
        VAI_ASSERT(L >= 1);

        blocks.reserve(L);
        for (uint32_t i = 0; i < L; ++i)
            blocks.emplace_back(Block::Config{
                .base = config.base,
                .kvCacheSize = config.kvCacheSize,
                .kvCacheDtype = config.kvCacheDtype,
            });

        // clang-format off
        for (size_t i = 1; i < blocks.size(); ++i)
            blocks[i - 1] - blocks[i];
        blocks.back() - finalNorm;
        // clang-format on

        condSubTables.reserve(L + 2);
        for (uint32_t i = 0; i < L; ++i)
            condSubTables.emplace_back(vai::node::Slice::Config{ .start = 6 * i, .count = 6 });
        condSubTables.emplace_back(vai::node::Slice::Config{ .start = 6 * L });
        condSubTables.emplace_back(vai::node::Slice::Config{ .start = 6 * L + 1 });

        // clang-format off
        for (uint32_t i = 0; i < L; ++i)
            cond - condSubTables[i] - "cond"/blocks[i];
        cond - condSubTables[L]     - "scale"/finalNorm;
        cond - condSubTables[L + 1] - "shift"/finalNorm;
        // clang-format on

        defineSlot("in0", blocks[0].slot("in0"));
        defineSlot("cond", cond.slot("in0"));
        defineSlot("out0", finalNorm.slot("out0"));
    }

    void setKvCacheCounter(eva::Buffer buffer)
    {
        for (auto& block : blocks)
            block.mhAttn.setKvCacheCounter(buffer);
    }
};


}   // namespace model::production::pi05::group

#endif   // PRODUCTION_PI05_GROUPS_ACTION_DECODER_H
