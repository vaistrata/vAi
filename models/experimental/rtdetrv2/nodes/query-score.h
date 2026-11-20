#ifndef EXPERIMENTAL_RTDETRV2_NODES_QUERY_SCORE_H
#define EXPERIMENTAL_RTDETRV2_NODES_QUERY_SCORE_H

#include "vai/nodes/common.h"

namespace model::experimental::rtdetrv2::node {

/////////////////////////////////////////////////////////////////////////////////////////
// QueryScore - Maximum class logit per candidate
// Input:  in0 [N, C] (fp32)
// Output: out0 [N] (fp32)
/////////////////////////////////////////////////////////////////////////////////////////
class QueryScore : public vai::Node
{
    eva::ComputePipeline pipe_;
    eva::DescriptorSet descSet_;

public:
    QueryScore();
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;
};

/////////////////////////////////////////////////////////////////////////////////////////
// QueryScore
/////////////////////////////////////////////////////////////////////////////////////////
inline QueryScore::QueryScore()
{
    addSlot("in0", vai::NodeSlot::input);
    addSlot("out0", vai::NodeSlot::output);
    pipe_ = pipelinePool().request(EXPERIMENTAL_RTDETRV2_SHADER_OUTPUT_DIR, "query-score", {}, {});
    descSet_ = vai::gVaiDestSetPool(pipe_.descSetLayout(0));
}

inline void QueryScore::prepare()
{
    const auto& logits = (*this)["in0"];
    VAI_ASSERT(logits.isShapeOf(-1, -1) && logits.dtype() == vai::Dtype::Fp32);
    (*this)["out0"] = vai::Tensor({ logits.shape()[0] }, vai::Dtype::Fp32);
}

inline void QueryScore::run(eva::CommandBuffer cmdBuff)
{
    const auto& logits = (*this)["in0"];
    auto scoresBuf = (*this)["out0"].buffer();
    uint32_t constants[] = { logits.shape()[0], logits.shape()[1] };
    descSet_.write({ scoresBuf, logits.buffer() });
    cmdBuff.bindPipeline(pipe_)
        .bindDescSets({ descSet_ })
        .setPushConstants(0, sizeof(constants), constants)
        .dispatch2(constants[0])
        .barrier(eva::SYNC_SCOPE::COMPUTE_READ / logits.buffer() / eva::SYNC_SCOPE::COMPUTE_WRITE)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / scoresBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace model::experimental::rtdetrv2::node

#endif   // EXPERIMENTAL_RTDETRV2_NODES_QUERY_SCORE_H
