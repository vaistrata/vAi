#ifndef EXPERIMENTAL_RTDETRV2_NODES_TOPK_H
#define EXPERIMENTAL_RTDETRV2_NODES_TOPK_H

#include "vai/nodes/common.h"
#include <limits>

namespace model::experimental::rtdetrv2::node {

/*=============================================================================
  TopK - Row-wise top-k selection
  Input:  in0     [N] or [M, N] (contiguous fp32 or fp16)
  Output: values  [K] or [M, K] (same dtype as in0)
          indices [K] or [M, K] (raw uint32 indices, in 4-byte slots)

  Each row selects its K largest values, sorted by descending value then ascending
  index. Indices are positions within the input row; read with getData<uint32_t>().
  NaNs sort after all numbers, including -infinity, with ties broken by index.
  Output values preserve the input bits, including signed zero and NaN payloads.
=============================================================================*/
class TopK : public vai::Node
{
    eva::ComputePipeline blockPipe_, mergePipe_;
    eva::DescriptorSet blockDs_;
    std::vector<eva::DescriptorSet> mergeDs_;
    uint32_t rows_ = 0, columns_ = 0, groups_ = 0;

public:
    struct Config
    {
        uint32_t k = 1;   // fixed at construction; 1 <= k <= min(N, 512)
    };
    const Config config_;

    explicit TopK(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmd) override;
};

/////////////////////////////////////////////////////////////////////////////////////////
// TopK
/////////////////////////////////////////////////////////////////////////////////////////
inline TopK::TopK(Config config)
: config_(config)
{
    VAI_ASSERT(config_.k >= 1 && config_.k <= 512);

    addSlot("in0", vai::NodeSlot::input);
    addSlot("values", vai::NodeSlot::output);
    addSlot("indices", vai::NodeSlot::output);
    addSlot("scratch0", vai::NodeSlot::temporary);
    addSlot("scratch1", vai::NodeSlot::temporary);
}

inline void TopK::prepare()
{
    const auto& input = (*this)["in0"];
    VAI_ASSERT(input.dtype() == vai::Dtype::Fp32 || input.dtype() == vai::Dtype::Fp16);
    blockPipe_ = pipelinePool().request(
        EXPERIMENTAL_RTDETRV2_SHADER_OUTPUT_DIR,
        "topk-block",
        { eva::constant_id<0>(config_.k), eva::constant_id<SPEC_ID_TYPE_I>(input.dtype()) },
        {});
    mergePipe_ = pipelinePool().request(
        EXPERIMENTAL_RTDETRV2_SHADER_OUTPUT_DIR,
        "topk-merge",
        { eva::constant_id<0>(config_.k), eva::constant_id<SPEC_ID_TYPE_I>(input.dtype()) },
        {});
    if (!blockDs_)
        blockDs_ = vai::gVaiDestSetPool(blockPipe_.descSetLayout(0));
    const auto& shape = input.shape();
    VAI_ASSERT(shape.size() == 1 || shape.size() == 2);
    rows_ = shape.size() == 1 ? 1 : shape[0];
    columns_ = shape.back();
    VAI_ASSERT(rows_ > 0);
    VAI_ASSERT(columns_ >= config_.k);
    groups_ = columns_ / 512 + (columns_ % 512 != 0);

    // Block: (groups,rows,1); merge: (ceil(2*K/64),ceil(groups/2),rows).
    VAI_ASSERT(groups_ <= device().maxComputeWorkGroupCount()[0]);
    VAI_ASSERT(rows_ <= device().maxComputeWorkGroupCount()[1]);
    VAI_ASSERT(rows_ <= device().maxComputeWorkGroupCount()[2]);
    VAI_ASSERT((groups_ + 1) / 2 <= device().maxComputeWorkGroupCount()[1]);
    VAI_ASSERT(CEIL_DIV(2 * config_.k, 64u) <= device().maxComputeWorkGroupCount()[0]);
    // Shader addresses are uint32. Check products before narrowing/allocating.
    constexpr uint64_t maxIndex = std::numeric_limits<uint32_t>::max();
    VAI_ASSERT(uint64_t(rows_) * columns_ <= maxIndex);
    VAI_ASSERT(uint64_t(rows_) * groups_ * config_.k * 2 <= maxIndex);

    auto outShape = shape;
    outShape.back() = config_.k;
    (*this)["values"] = vai::Tensor(outShape, input.dtype());
    (*this)["indices"] = vai::Tensor(outShape, vai::Dtype::Fp32);
    // Only intermediate passes need candidate buffers; the final pass writes
    // values and indices directly. Clear unused slots after a shape change.
    (*this)["scratch0"] =
        groups_ > 1 ? vai::Tensor({ rows_, groups_ * config_.k, 2 }, vai::Dtype::Fp32) : vai::Tensor{};
    (*this)["scratch1"] =
        groups_ > 2 ? vai::Tensor({ rows_, ((groups_ + 1) / 2) * config_.k, 2 }, vai::Dtype::Fp32) : vai::Tensor{};

    size_t passes = 0;
    for (uint32_t g = groups_; g > 1; g = (g + 1) / 2)
        ++passes;
    while (mergeDs_.size() < passes)
        mergeDs_.push_back(vai::gVaiDestSetPool(mergePipe_.descSetLayout(0)));
}

inline void TopK::run(eva::CommandBuffer cmd)
{
    auto values = (*this)["values"].buffer(), indices = (*this)["indices"].buffer();
    auto a = groups_ > 1 ? (*this)["scratch0"].buffer() : values;
    auto b = groups_ > 2 ? (*this)["scratch1"].buffer() : values;
    const uint32_t blockConstants[] = { columns_, groups_ };
    // Binding 0 receives pairs on intermediate passes, values on the final
    // pass. Binding 2 is unused on intermediate passes; reuse their output.
    blockDs_.write({ a, (*this)["in0"].buffer(), groups_ == 1 ? indices : a });
    cmd.bindPipeline(blockPipe_)
        .bindDescSets({ blockDs_ })
        .setPushConstants(0, sizeof(blockConstants), blockConstants)
        .dispatch(groups_, rows_);
    if (groups_ > 1)
        cmd.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / a / eva::SYNC_SCOPE::COMPUTE_READ);

    size_t pass = 0;
    for (uint32_t g = groups_; g > 1; g = (g + 1) / 2, ++pass)
    {
        // Each recorded pass keeps its own bindings. The output row stride
        // shrinks to ceil(g/2)*K, so every pass packs all rows afresh.
        const bool finalPass = g <= 2;
        mergeDs_[pass].write({ finalPass ? values : b, a, finalPass ? indices : b });
        cmd.bindPipeline(mergePipe_)
            .bindDescSets({ mergeDs_[pass] })
            .setPushConstants(0, sizeof(g), &g)
            .dispatch2(2 * config_.k, (g + 1) / 2, rows_);
        if (!finalPass)
        {
            cmd.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / b / eva::SYNC_SCOPE::COMPUTE_READ)
                .barrier(eva::SYNC_SCOPE::COMPUTE_READ / a / eva::SYNC_SCOPE::COMPUTE_WRITE);
            std::swap(a, b);
        }
    }

    // The graph can reuse the input and scratch allocations after this node.
    // Finish their reads before a later node overwrites the same memory.
    cmd.barrier(eva::SYNC_SCOPE::COMPUTE_READ / (*this)["in0"].buffer() / eva::SYNC_SCOPE::COMPUTE_WRITE);
    for (auto name : { "scratch0", "scratch1" })
        if ((*this)[name].validShape())
            cmd.barrier(eva::SYNC_SCOPE::COMPUTE_READ / (*this)[name].buffer() / eva::SYNC_SCOPE::COMPUTE_WRITE);

    cmd.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / values / eva::SYNC_SCOPE::COMPUTE_READ)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / indices / eva::SYNC_SCOPE::COMPUTE_READ)
        // An unconsumed values output can also be recycled by the graph.
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / values / eva::SYNC_SCOPE::COMPUTE_WRITE)
        // Both outputs may be read back via vai::Tensor::getData().
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / values / eva::SYNC_SCOPE::TRANSFER_SRC)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / indices / eva::SYNC_SCOPE::TRANSFER_SRC);
}

}   // namespace model::experimental::rtdetrv2::node

#endif   // EXPERIMENTAL_RTDETRV2_NODES_TOPK_H
