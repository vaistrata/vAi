#ifndef EXPERIMENTAL_RTDETRV2_NODES_INDEX_SELECT_H
#define EXPERIMENTAL_RTDETRV2_NODES_INDEX_SELECT_H

#include "vai/nodes/common.h"
#include <limits>

namespace model::experimental::rtdetrv2::node {

/////////////////////////////////////////////////////////////////////////////////////////
// IndexSelect - Select rows along the leading axis
// Input:  in0     [N, d1, ..., dn] (rank >= 1, contiguous fp32 or fp16)
//         indices [K]             (raw uint32 indices, in 4-byte slots)
// Output: out0    [K, d1, ..., dn] (same dtype as in0)
//
// Copies whole rows in index order, preserving duplicates and the input dtype.
// Indices use fp32 storage as in TopK; read with getData<uint32_t>().
// N and K must be positive. Out-of-range indices produce zero-filled rows.
/////////////////////////////////////////////////////////////////////////////////////////
class IndexSelect : public vai::Node
{
    eva::ComputePipeline pipe_;
    eva::DescriptorSet descSet_;
    uint32_t rows_ = 0, rowElems_ = 0, outElems_ = 0;
    uint32_t groupsX_ = 0, groupsY_ = 0;

public:
    struct Config
    {
        bool constantInput = false;
    };
    const Config config_;

    explicit IndexSelect(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmd) override;
};

/////////////////////////////////////////////////////////////////////////////////////////
// IndexSelect
/////////////////////////////////////////////////////////////////////////////////////////
inline IndexSelect::IndexSelect(Config config)
: config_(config)
{

    addSlot("out0", vai::NodeSlot::output);
    addSlot("in0", config_.constantInput ? vai::NodeSlot::persistent : vai::NodeSlot::input);
    addSlot("indices", vai::NodeSlot::input);
}

inline void IndexSelect::prepare()
{
    const auto& input = (*this)["in0"];
    VAI_ASSERT(input.dtype() == vai::Dtype::Fp32 || input.dtype() == vai::Dtype::Fp16);
    pipe_ = pipelinePool().request(
        EXPERIMENTAL_RTDETRV2_SHADER_OUTPUT_DIR,
        "index-select",
        { eva::constant_id<SPEC_ID_TYPE_I>(input.dtype()) },
        {});
    if (!descSet_)
        descSet_ = vai::gVaiDestSetPool(pipe_.descSetLayout(0));
    const auto& indices = (*this)["indices"];
    VAI_ASSERT(input.validShape());
    VAI_ASSERT(indices.validShape());
    VAI_ASSERT(indices.shape().size() == 1);
    VAI_ASSERT(indices.dtype() == vai::Dtype::Fp32);

    const auto& shape = input.shape();
    rows_ = shape[0];
    constexpr uint64_t maxIndex = std::numeric_limits<uint32_t>::max();
    // Bound each product before multiplying, including tensors of arbitrary rank.
    uint64_t elements = 1;
    for (uint32_t dim : shape)
    {
        VAI_ASSERT(elements <= maxIndex / dim);
        elements *= dim;
    }
    rowElems_ = uint32_t(elements / rows_);
    const uint64_t count = uint64_t(indices.shape()[0]) * rowElems_;
    VAI_ASSERT(count <= maxIndex);
    outElems_ = uint32_t(count);

    const auto limits = device().maxComputeWorkGroupCount();
    const uint64_t groups = (count + 63) / 64;
    groupsX_ = uint32_t(std::min(groups, uint64_t(limits[0])));
    groupsY_ = uint32_t((groups + groupsX_ - 1) / groupsX_);
    VAI_ASSERT(groupsY_ <= limits[1]);

    auto outShape = shape;
    outShape[0] = indices.shape()[0];
    (*this)["out0"] = vai::Tensor(outShape, input.dtype());
}

inline void IndexSelect::run(eva::CommandBuffer cmd)
{
    auto output = (*this)["out0"].buffer();
    auto input = (*this)["in0"].buffer();
    auto indices = (*this)["indices"].buffer();
    const uint32_t constants[] = { rows_, rowElems_, outElems_ };
    descSet_.write({ output, input, indices });
    cmd.bindPipeline(pipe_)
        .bindDescSets({ descSet_ })
        .setPushConstants(0, sizeof(constants), constants)
        .dispatch(groupsX_, groupsY_)
        // Finish reads before the graph reuses the input allocations.
        .barrier(eva::SYNC_SCOPE::COMPUTE_READ / input / eva::SYNC_SCOPE::COMPUTE_WRITE)
        .barrier(eva::SYNC_SCOPE::COMPUTE_READ / indices / eva::SYNC_SCOPE::COMPUTE_WRITE)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / output / eva::SYNC_SCOPE::COMPUTE_READ)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / output / eva::SYNC_SCOPE::TRANSFER_SRC);
}

}   // namespace model::experimental::rtdetrv2::node

#endif   // EXPERIMENTAL_RTDETRV2_NODES_INDEX_SELECT_H
