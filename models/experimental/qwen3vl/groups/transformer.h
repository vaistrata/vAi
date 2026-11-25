#ifndef EXPERIMENTAL_QWEN3VL_GROUPS_TRANSFORMER_H
#define EXPERIMENTAL_QWEN3VL_GROUPS_TRANSFORMER_H

#include "vai/groups/transformer.h"
#include "vai/nodes/index-copy.h"
#include "vai/nodes/rms-norm.h"
#include <string>
#include <vector>

namespace model::experimental::qwen3vl {

// Qwen3's decoder: 128-wide heads over 8 KV heads, QK-normed, SwiGLU, and the
// interleaved MRoPE at every size.
constexpr vai::TransformerConfig lm(uint32_t nBlocks, uint32_t embedDim, uint32_t numQHeads, uint32_t ffnHiddenDim)
{
    return {
        .nBlocks = nBlocks,
        .embedDim = embedDim,
        .headDim = 128,
        .numQHeads = numQHeads,
        .numKVHeads = 8,
        .ffnHiddenDim = ffnHiddenDim,
        .ffnActType = vai::ActType::Silu,
        .normEps = 1e-6f,
        .withQkNorm = true,
        .ropeCfg =
            vai::RopeConfig{
                .posType = vai::RopePosType::MRoPE,
                .base = 5000000.0f,
                .mropeSection = { 24, 20, 20 },
            },
    };
}

}   // namespace model::experimental::qwen3vl
namespace model::experimental::qwen3vl::group {

using TransformerBlock = vai::group::TransformerBlock<vai::TransformerType::Default, vai::QkvLayout::Separate>;


/////////////////////////////////////////////////////////////////////////////////////////
// Transformer - Qwen3-VL's decoder: the blocks, DeepStack and the closing norm
// Input: in0 [T, embedDim]
//        deep0.. [B * tokensPerImage, embedDim], one DeepStack tap each
// Output: out0 [1, embedDim], the last row
//
// Block j's output takes tap j at the image tokens' rows before block j + 1 reads
// it. A forward without images leaves the taps unwired and the adds pass through.
//
// RoPE rotates by the positions the model writes per token rather than by the
// cache counter: an image's tokens share a row, so positions run behind it.
/////////////////////////////////////////////////////////////////////////////////////////
struct Transformer : public vai::NodeGroup
{
    std::vector<TransformerBlock> blocks;
    std::vector<vai::node::IndexCopy> deepAdds;
    vai::node::RmsNorm finalNorm;

    struct Config
    {
        vai::TransformerConfig base;
        uint32_t kvCacheSize;
        vai::Dtype kvCacheDtype;
        uint32_t deepstackCount;
    };

    Transformer(Config config)
    : finalNorm({ .eps = config.base.normEps })
    {
        const vai::TransformerConfig& base = config.base;
        VAI_ASSERT(config.deepstackCount < base.nBlocks);

        blocks.reserve(base.nBlocks);
        for (uint32_t i = 0; i < base.nBlocks; ++i)
            blocks.emplace_back(TransformerBlock::Config{
                .base = base,
                .kvCacheSize = config.kvCacheSize,
                .kvCacheDtype = config.kvCacheDtype,
                .sliceLastRow = i == base.nBlocks - 1,
            });

        deepAdds.reserve(config.deepstackCount);
        for (uint32_t j = 0; j < config.deepstackCount; ++j)
            deepAdds.emplace_back(vai::node::IndexCopy::Config{ .accumulate = true });

        // clang-format off
        for (size_t i = 1; i < blocks.size(); ++i)
        {
            if (i - 1 < deepAdds.size())
                blocks[i - 1] - deepAdds[i - 1] - blocks[i];
            else
                blocks[i - 1] - blocks[i];
        }
        blocks.back() - finalNorm;
        // clang-format on

        defineSlot("in0", blocks[0].slot("in0"));
        defineSlot("out0", finalNorm.slot("out0"));
        for (size_t j = 0; j < deepAdds.size(); ++j)
            defineSlot("deep" + std::to_string(j), deepAdds[j].slot("in1"));
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

    // positions [T, 3]: each token's (t, h, w).
    void setPositions(eva::Buffer buffer)
    {
        for (auto& block : blocks)
            block.mhAttn.setPositions(buffer);
    }

    // index [B * tokensPerImage]: the image tokens' rows, in the order the taps hold them.
    void setImageIndex(eva::Buffer buffer)
    {
        for (auto& add : deepAdds)
            add.setIndex(buffer);
    }
};

}   // namespace model::experimental::qwen3vl::group

#endif   // EXPERIMENTAL_QWEN3VL_GROUPS_TRANSFORMER_H
