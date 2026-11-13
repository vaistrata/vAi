#ifndef VAI_NODES_TOKEN_EMBEDDING_H
#define VAI_NODES_TOKEN_EMBEDDING_H

// Token (+ optional positional) embedding lookup.
#include "node-common.h"

namespace vai::node {


/*=============================================================================
  TokenEmbedding - token embedding, optionally fused with positional embedding
  Input : in0  token IDs   [..., T] (uint32)
  Output: out0 embedding   [..., T, D] (fp32)

  Leading axes are more sequences swept in one dispatch, the way a Linear takes its
  leading axes as more rows, so the output keeps the input's rank.

  posTableSize == 0 : O[n] = scale * W[ids[n]]                       (e.g. Llama3 — RoPE)
  posTableSize  > 0 : O[n] = scale * W[ids[n]] + posW[cachedCount + n]  (e.g. GPT-2)

  Only Gemma uses scale (sqrt(D)). It stays out of the weights because a tied LM head
  reads the same table. posTableSize > 0 adds a "posWeight" [posTableSize, D] slot
  and needs setKvCacheCounter() to supply cachedCount; positions restart at each
  sequence, which the kernel only has to work out when there is more than one.
=============================================================================*/
class TokenEmbedding : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;
    eva::Buffer kvCacheCounter;       // position offset; only used with a positional table

public:
    struct PipeParams {
        bool batched = false;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        uint32_t vocabSize;
        uint32_t embedDim;
        uint32_t posTableSize = 0;          // learned positional table rows; 0 = token only
        float scale = 1.0f;                 // multiplies the looked-up embedding
    } config_;

    TokenEmbedding(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

    // Position offset for incremental (KV-cache) decoding; only used with a positional table.
    void setKvCacheCounter(eva::Buffer buffer) { kvCacheCounter = buffer; }

private:
    PipeParams pipeParams;
};


inline TokenEmbedding::TokenEmbedding(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0",  NodeSlot::input);
    addSlot("weight", NodeSlot::persistent);
    if (config_.posTableSize)
        addSlot("posWeight", NodeSlot::persistent);
}

inline void TokenEmbedding::prepare()
{
    const Dtype oDtype = (oDtype_ == Dtype::None) ? oDefaultDtype() : oDtype_;
    const Tensor& inTen   = (*this)["in0"];
    const Tensor& wTen = (*this)["weight"];

    VAI_ASSERT(inTen.validShape());
    VAI_ASSERT(wTen.isShapeOf(config_.vocabSize, config_.embedDim));

    if (config_.posTableSize)
    {
        const Tensor& posWTen = (*this)["posWeight"];
        VAI_ASSERT(posWTen.dtype() == wTen.dtype());
        VAI_ASSERT(posWTen.isShapeOf(config_.posTableSize, config_.embedDim));
    }

    const auto& inShape = inTen.shape();
    const uint32_t seqCount = outerSize(inShape);
    std::vector<uint32_t> outShape = inShape;
    outShape.push_back(config_.embedDim);
    (*this)["out0"] = Tensor(std::move(outShape), oDtype);

    // Only more than one sequence makes a position wrap. The flag is baked in, so
    // a lone sequence runs the kernel it always did.
    const PipeParams p = { .batched = seqCount > 1 };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        // clang-format off
        pipe = pipelinePool().request("token-embedding", {
            eva::constant_id<0>(config_.posTableSize ? 1u : 0u),   // hasPosEmbed
            eva::constant_id<1>(p.batched ? 1u : 0u),              // batched
            eva::constant_id<SPEC_ID_TYPE_O>(oDtype),      // typeO
            eva::constant_id<SPEC_ID_TYPE_W>(wTen.dtype())         // typeW (W & posW)
        });
        // clang-format on
        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }
}

inline void TokenEmbedding::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf  = (*this)["out0"].buffer();
    auto inBuf   = (*this)["in0"].buffer();
    auto wBuf = (*this)["weight"].buffer();
    auto posWBuf = config_.posTableSize ? (*this)["posWeight"].buffer() : eva::Buffer();

    // Single set: 0=O(out), 1=tokenIds, 2=W, 3=posW, 4=kvCacheCount.
    // Without positional embedding, posW and the counter are bound empty.
    descSet.write({
        outBuf,
        inBuf,
        wBuf,
        config_.posTableSize ? posWBuf : eva::Buffer(),
        config_.posTableSize ? kvCacheCounter : eva::Buffer(),
    });

    const auto& inShape = (*this)["in0"].shape();
    const uint32_t T = inShape.back(), rows = outerSize(inShape) * T;

    struct {
        uint32_t rows, T, D;
        float scale;
    } constants = { rows, T, config_.embedDim, config_.scale };

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch2(config_.embedDim, rows)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace vai::node

#endif // VAI_NODES_TOKEN_EMBEDDING_H
