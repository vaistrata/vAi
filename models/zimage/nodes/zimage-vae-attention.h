#ifndef ZIMAGE_VAE_ATTENTION_H
#define ZIMAGE_VAE_ATTENTION_H

#include "vai/nodes/node-common.h"
#include <algorithm>
#include <cmath>

namespace zimage {

/*=============================================================================
  VaeAttentionNode - single-head self-attention over a feature map
  Input : QKV [3D, H, W] fp32 (one 1x1 conv's q, k, v channels, in that order)
  Output: out0 [D, H, W] fp32 = V softmax(K^T Q / sqrt(D)), per position

  The VAE decoder's mid-block attention (diffusers' Attention with one head
  of all D channels) over the N = H*W latent positions. It reads and writes
  the channel-major layout the convs around it use, so the q/k/v and output
  projections run as 1x1 convs with no transposes.

  Three passes per block of queries, over a scores buffer [N, block]:
  zimage-vae-attn-scores (S = K^T Q / sqrt(D)), zimage-vae-attn-softmax
  (down each column) and zimage-vae-attn-values (O = V S), the products as
  register-tiled GEMMs (zimage-vae-attn-gemm.glsl). Materializing S costs
  N * block * 4 bytes; blocks keep it within kScoresBudget. One query row
  per lane-group flash attention ran at ~34 GFLOPS here (1 s at 512x512).
=============================================================================*/
class VaeAttentionNode : public vai::Node
{
    eva::ComputePipeline pipeScores, pipeSoftmax, pipeValues;
    eva::DescriptorSet descScores, descSoftmax, descValues;

public:
    static constexpr uint32_t kTileM = 8, kTileN = 4, kWgSize = 64;   // zimage-vae-attn-gemm.glsl
    static constexpr size_t kScoresBudget = size_t(8) << 20;          // floats in the scores buffer (32 MB)

    VaeAttentionNode();
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    uint32_t block_ = 0;   // queries per block
};


inline VaeAttentionNode::VaeAttentionNode()
{
    addSlot("out0", vai::NodeSlot::output);
    addSlot("in0", vai::NodeSlot::input);
    addSlot("scores", vai::NodeSlot::temporary);
}

inline void VaeAttentionNode::prepare()
{
    const vai::Tensor& qkv = (*this)["in0"];
    VAI_ASSERT(qkv.shape().size() == 3 && qkv.shape()[0] % 3 == 0 && qkv.dtype() == vai::Dtype::Fp32);
    const uint32_t D = qkv.shape()[0] / 3, N = qkv.shape()[1] * qkv.shape()[2];
    VAI_ASSERT(D % 4 == 0 && N % 4 == 0);

    if (!pipeScores)
    {
        pipeScores = pipelinePool().request(ZIMAGE_SHADER_OUTPUT_DIR, "zimage-vae-attn-scores", eva::SpecializationConstant{});
        pipeSoftmax = pipelinePool().request(ZIMAGE_SHADER_OUTPUT_DIR, "zimage-vae-attn-softmax", eva::SpecializationConstant{});
        pipeValues = pipelinePool().request(ZIMAGE_SHADER_OUTPUT_DIR, "zimage-vae-attn-values", eva::SpecializationConstant{});
        descScores = vai::gVaiDestSetPool(pipeScores.descSetLayout(0));
        descSoftmax = vai::gVaiDestSetPool(pipeSoftmax.descSetLayout(0));
        descValues = vai::gVaiDestSetPool(pipeValues.descSetLayout(0));
    }

    block_ = std::min<uint32_t>(N, std::max<uint32_t>(4, uint32_t(kScoresBudget / N) / 4 * 4));
    (*this)["scores"] = vai::Tensor({N, block_}, vai::Dtype::Fp32);
    (*this)["out0"] = vai::Tensor({D, qkv.shape()[1], qkv.shape()[2]}, vai::Dtype::Fp32);
}

inline void VaeAttentionNode::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto qkvBuf = (*this)["in0"].buffer();
    auto sBuf = (*this)["scores"].buffer();
    descScores.write({ sBuf, qkvBuf, qkvBuf });
    descSoftmax.write({ sBuf });
    descValues.write({ outBuf, qkvBuf, sBuf });

    const auto& shape = (*this)["in0"].shape();
    const uint32_t D = shape[0] / 3, N = shape[1] * shape[2];
    struct Gemm {
        uint32_t M, N, K, offA, ldB, offB, ldO, offO;
        float alpha;
    };
    for (uint32_t q0 = 0; q0 < N; q0 += block_)
    {
        const uint32_t cols = std::min(block_, N - q0);
        // S[key, q] = K[:, key] . Q[:, q0 + q] / sqrt(D); K at row D of QKV.
        const Gemm scores = { N, cols, D, D * N, N, q0, block_, 0, float(1.0 / std::sqrt(double(D))) };
        // O[c, q0 + q] = sum_key V[c, key] S[key, q]; V at row 2D.
        const Gemm values = { D, cols, N, 2 * D * N, block_, 0, N, q0, 1.0f };
        struct { uint32_t rows, cols; } softmax = { N, block_ };   // whole rows, the tail's spare columns too

        cmdBuff
            .bindPipeline(pipeScores)
            .bindDescSets({descScores})
            .setPushConstants(0, sizeof(scores), &scores)
            .dispatch(CEIL_DIV(cols, kTileN * kWgSize), CEIL_DIV(N, kTileM))
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / sBuf / eva::SYNC_SCOPE::COMPUTE_READ)
            .bindPipeline(pipeSoftmax)
            .bindDescSets({descSoftmax})
            .setPushConstants(0, sizeof(softmax), &softmax)
            .dispatch(CEIL_DIV(cols, kWgSize))
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / sBuf / eva::SYNC_SCOPE::COMPUTE_READ)
            .bindPipeline(pipeValues)
            .bindDescSets({descValues})
            .setPushConstants(0, sizeof(values), &values)
            .dispatch(CEIL_DIV(cols, kTileN * kWgSize), CEIL_DIV(D, kTileM))
            // Also orders this block's reads of S before the next block's writes.
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
    }
}

} // namespace zimage

#endif // ZIMAGE_VAE_ATTENTION_H
