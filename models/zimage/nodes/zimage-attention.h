#ifndef ZIMAGE_ATTENTION_H
#define ZIMAGE_ATTENTION_H

#include "vai/nodes/node-common.h"
#include "vai/nodes/attention.h"   // ggmlCoopmatAllowedAttention
#include <cmath>

namespace zimage {

/*=============================================================================
  FullAttentionNode - bidirectional multi-head attention over one sequence
  Input : Q [H, N, Dh] f16, K [H, N, Dh] f16 (QkNormRopeNode's layout),
          V [N, H*Dh] (token-major, the projection output as is)
  Output: out0 [N, H*Dh] = softmax(Q K^T / sqrt(Dh)) V, token-major

  A DiT runs a few thousand rows of prefill-only attention per layer, so
  this node's kernels put many rows on one shared K/V copy (prefetched a
  tile ahead), with no cache, append or split-K to carry:
    coopmat  16 rows per subgroup, numSubgroups subgroups per workgroup,
             16x16x16 f16 cooperative matrices (zimage-attention.comp).
             Measured on an A40 (DiT layer, N = 4128, 30 heads): 4.8 ms with
             8 subgroups, 5.8 ms with 4, vs. 9.0 ms for vai::node::Attention
             run cacheless under an all-bidirectional mask (its coopmat
             kernel plus the permute its head-major V needs); 32 rows per
             subgroup spilled and ran 3x slower.
    scalar   32 rows per 128 threads on plain FMAs, for devices without
             (trusted) cooperative matrices (zimage-attention-scalar.comp).
             Its tile is scalarKeysPerTile: with 32 keys the per-thread
             scores and prefetch spill registers on an Adreno 830 and run
             ~140x slower (past the driver's submit timeout at DiT lengths).
    int8     where int8Supported(): Q and K quantized to int8 per row (the
             softmax scale in Q's) and V cast to fp16 by
             zimage-attention-i8-quant, then a register-tiled kernel with
             no shared memory: Q·K on packed int8 dot products, P·V summed
             in fp16 (zimage-attention-i8.comp). On a cool Galaxy S25 Ultra
             (N = 1056, 30 heads) 12.6 ms (0.8 + 11.8) vs. 83 ms scalar and
             34 ms for the shared-memory int8 kernel it replaced; the 512x512
             forward's rel L2 against fp32 is 3.99e-2 (3.70e-2 scalar, 3.91e-2
             the old int8 kernel). Needs N % kInt8KeysPerTile == 0; other
             lengths run scalar.
  coopmatSupported() picks first; allowCoopmat = false and int8QK = false
  turn the kernels off.
=============================================================================*/
class FullAttentionNode : public vai::Node
{
    eva::ComputePipeline pipe, pipeQuant;
    eva::DescriptorSet descSet, descQuant;

public:
    static constexpr uint32_t kInt8RowsPerWg = 32, kInt8KeysPerTile = 32;   // zimage-attention-i8.comp
    static constexpr uint32_t kQuantRowsPerWg = 8;                           // zimage-attention-i8-quant.comp

    struct PipeParams {
        bool coopmat = false;
        bool int8 = false;
        uint32_t keysPerTile = 0;
        vai::Dtype vDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        uint32_t numHeads;
        uint32_t headDim;                       // multiple of 16
        uint32_t numSubgroups = 8;              // coopmat: query rows per workgroup = 16 * numSubgroups
        uint32_t keysPerTile = 32;              // coopmat; multiple of 16, N must be a multiple of it
        uint32_t scalarKeysPerTile = 16;        // scalar; N must be a multiple of it
        bool allowCoopmat = true;
        bool int8QK = true;                     // scalar devices: zimage-attention-i8 where supported
        vai::Dtype oDtype = vai::Dtype::None;   // None: oDefaultDtype
    } config_;

    FullAttentionNode(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

    // Whether the coopmat kernel can run with this config on `device`.
    static bool coopmatSupported(const Config& config, eva::Device device);
    // Whether the int8 kernel can run with this config on `device`.
    static bool int8Supported(const Config& config, eva::Device device);
    bool usingCoopmat() const { return pipeParams.coopmat; }
    bool usingInt8() const { return pipeParams.int8; }

private:
    PipeParams pipeParams;
    uint32_t rowsPerWg = 0;
};


inline bool FullAttentionNode::coopmatSupported(const Config& c, eva::Device device)
{
    if (!device.supportsCooperativeMatrix() || !vai::node::ggmlCoopmatAllowedAttention(device))
        return false;

    bool shapeOk = false;   // 16x16x16, f16 A/B, f32 accumulator, subgroup scope
    for (const auto& p : device.cooperativeMatrixProperties())
        if (p.M == 16 && p.N == 16 && p.K == 16 &&
            p.aType == eva::COMPONENT_TYPE::FLOAT16 && p.bType == eva::COMPONENT_TYPE::FLOAT16 &&
            p.cType == eva::COMPONENT_TYPE::FLOAT32 && p.resultType == eva::COMPONENT_TYPE::FLOAT32 &&
            p.scope == eva::SCOPE::SUBGROUP)
            shapeOk = true;

    const uint32_t sg = device.subgroupSize();
    if (!shapeOk || (sg != 16 && sg != 32 && sg != 64) || c.keysPerTile % (sg / 16) != 0)
        return false;

    // Shared arrays of zimage-attention.comp.
    const size_t rows = 16 * c.numSubgroups;
    const size_t shared = 2 * size_t(c.keysPerTile) * (c.headDim / 4 + 2) * 8   // s_k, s_v
                        + rows * (c.keysPerTile + 4) * 4                         // s_s
                        + rows * (c.keysPerTile + 8) * 2                         // s_p
                        + rows * 4;                                              // s_row
    return shared <= device.maxComputeSharedMemorySize();
}

inline bool FullAttentionNode::int8Supported(const Config& c, eva::Device device)
{
    // zimage-attention-i8.comp: compile-time head dim and tile; the quant
    // pass's row max is one butterfly over 32 invocations. The kernels use
    // the signed dot form (1.2x the mixed one on Adreno 830), which devices
    // accelerating the mixed form also accelerate, and fp16 arithmetic, as
    // linear-q4-q8 does under the same check.
    return device.isDot4x8PackedMixedAccelerated()
        && c.headDim == 128 && device.subgroupSize() >= 32;
}

inline FullAttentionNode::FullAttentionNode(Config config)
: config_(config)
{
    addSlot("out0", vai::NodeSlot::output);
    addSlot("Q", vai::NodeSlot::input);
    addSlot("K", vai::NodeSlot::input);
    addSlot("V", vai::NodeSlot::input);
    addSlot("qk8", vai::NodeSlot::temporary);      // int8: Q's rows, then K's
    addSlot("scales", vai::NodeSlot::temporary);   // int8: each row's scale, same order
    addSlot("v16", vai::NodeSlot::temporary);      // int8: V as fp16

    VAI_ASSERT(config_.headDim % 16 == 0 && config_.keysPerTile % 16 == 0);
    config_.oDtype = (config_.oDtype == vai::Dtype::None) ? oDefaultDtype() : config_.oDtype;
}

inline void FullAttentionNode::prepare()
{
    const uint32_t H = config_.numHeads, Dh = config_.headDim;
    const vai::Tensor& q = (*this)["Q"];
    const vai::Tensor& k = (*this)["K"];
    const vai::Tensor& v = (*this)["V"];

    VAI_ASSERT(q.dtype() == vai::Dtype::Fp16 && k.dtype() == vai::Dtype::Fp16);
    VAI_ASSERT(q.isShapeOf(H, -1, Dh));
    const uint32_t N = q.shape()[1];
    VAI_ASSERT(k.isShapeOf(H, N, Dh));
    VAI_ASSERT(v.isShapeOf(N, H * Dh));
    const bool coopmat = config_.allowCoopmat && coopmatSupported(config_, device());
    const bool int8 = !coopmat && config_.int8QK && N % kInt8KeysPerTile == 0
                   && int8Supported(config_, device());
    const PipeParams p = {
        .coopmat = coopmat,
        .int8 = int8,
        .keysPerTile = coopmat ? config_.keysPerTile : int8 ? kInt8KeysPerTile : config_.scalarKeysPerTile,
        .vDtype = v.dtype(),
    };
    VAI_ASSERT(N % 16 == 0 && N % p.keysPerTile == 0);
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        if (p.int8)
        {
            rowsPerWg = kInt8RowsPerWg;
            pipe = pipelinePool().request(ZIMAGE_SHADER_OUTPUT_DIR, "zimage-attention-i8", {
                eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype)       // typeO
            });
            pipeQuant = pipelinePool().request(ZIMAGE_SHADER_OUTPUT_DIR, "zimage-attention-i8-quant", {
                eva::constant_id<4>(p.vDtype)                          // typeV
            });
            descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
            descQuant = vai::gVaiDestSetPool(pipeQuant.descSetLayout(0));
        }
        else
        {
            const uint32_t sg = p.coopmat ? device().subgroupSize() : 0;
            const uint32_t wgSize = p.coopmat ? config_.numSubgroups * sg : 128;
            rowsPerWg = p.coopmat ? 16 * config_.numSubgroups : wgSize / 4;

            pipe = pipelinePool().request(ZIMAGE_SHADER_OUTPUT_DIR,
                p.coopmat ? "zimage-attention" : "zimage-attention-scalar", {
                    eva::constant_id<0>(wgSize),
                    eva::constant_id<1>(sg),                               // sgSize (coopmat; scalar ignores)
                    eva::constant_id<2>(Dh),
                    eva::constant_id<3>(p.keysPerTile),                    // Bc
                    eva::constant_id<4>(p.vDtype),                         // typeV
                    eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype)       // typeO
                },
                { .requiredSubgroupSize = sg });                           // 0 = not forced
            descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
        }
    }

    if (p.int8)
    {
        VAI_ASSERT(v.dtype() == vai::Dtype::Fp32 || v.dtype() == vai::Dtype::Fp16);
        (*this)["qk8"] = vai::Tensor({2 * H * N * Dh / 4}, vai::Dtype::Fp32);   // bytes, as floats
        (*this)["scales"] = vai::Tensor({2 * H * N}, vai::Dtype::Fp32);
        (*this)["v16"] = vai::Tensor({N, H * Dh}, vai::Dtype::Fp16);
    }
    (*this)["out0"] = vai::Tensor({N, H * Dh}, config_.oDtype);
}

inline void FullAttentionNode::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    const uint32_t N = (*this)["Q"].shape()[1], H = config_.numHeads;
    const float scaleLog2 = float(1.4426950408889634 / std::sqrt(double(config_.headDim)));

    if (pipeParams.int8)
    {
        auto qk8 = (*this)["qk8"].buffer();
        auto scales = (*this)["scales"].buffer();
        auto vIn = (*this)["V"].buffer();
        auto v16 = (*this)["v16"].buffer();
        descQuant.write({ qk8, scales, (*this)["Q"].buffer(), (*this)["K"].buffer(), vIn, v16 });
        descSet.write({ outBuf, qk8, v16, scales });

        struct { uint32_t rows; float scaleLog2; } quant = { H * N, scaleLog2 };
        struct { uint32_t N, H; } attn = { N, H };
        cmdBuff
            .bindPipeline(pipeQuant)
            .bindDescSets({descQuant})
            .setPushConstants(0, sizeof(quant), &quant)
            .dispatch(CEIL_DIV(H * N, kQuantRowsPerWg), 3)
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / qk8 / eva::SYNC_SCOPE::COMPUTE_READ)
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / scales / eva::SYNC_SCOPE::COMPUTE_READ)
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / v16 / eva::SYNC_SCOPE::COMPUTE_READ)
            .bindPipeline(pipe)
            .bindDescSets({descSet})
            .setPushConstants(0, sizeof(attn), &attn)
            .dispatch(CEIL_DIV(N, rowsPerWg), H)
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
        return;
    }

    descSet.write({ outBuf, (*this)["Q"].buffer(), (*this)["K"].buffer(), (*this)["V"].buffer() });

    struct {
        uint32_t N, H;
        float scaleLog2;
    } constants = { N, H, scaleLog2 };

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch((constants.N + rowsPerWg - 1) / rowsPerWg, constants.H)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

} // namespace zimage

#endif // ZIMAGE_ATTENTION_H
