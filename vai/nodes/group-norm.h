#ifndef VAI_NODES_GROUP_NORM_H
#define VAI_NODES_GROUP_NORM_H

#include "vai/nodes/node-common.h"

namespace vai::node {

/*=============================================================================
  GroupNorm - group normalization over channel groups, then an epilogue
  Input : activations [C, ...] (channel-major, e.g. a conv's [C, H, W])
  Output: normalized  [C, ...]

  O[c, j] = act((I[c, j] - mean[g]) * rsqrt(var[g] + eps) * γ[c] + β[c])

  with g = c / (C / numGroups), the statistics over all of group g's
  channels and positions, as torch.nn.GroupNorm (batch 1). A group of a
  decoder's full-resolution map is ~1M elements, far more than one
  workgroup should reduce, so the node runs two kernels over the same grid
  of chunkLen-element chunks: group-norm-stats writes each chunk's shifted
  sums, group-norm-apply folds its group's sums into mean and variance and
  normalizes its chunk. Accumulation is fp32 whatever the I/O dtypes.

  The epilogue (e.g. SiLU) is the activation a ResNet block applies right
  after each norm.
=============================================================================*/
class GroupNorm : public vai::Node
{
    eva::ComputePipeline pipeStats, pipeApply;
    eva::DescriptorSet descStats, descApply;

public:
    struct Config {
        uint32_t numGroups = 32;
        float eps = 1e-6f;
        vai::Epilogue epilogue = vai::Epilogue::None;
        vai::Dtype wDtype = vai::Dtype::Fp32;   // gamma / beta
        vai::Dtype oDtype = vai::Dtype::None;   // None: oDefaultDtype
    } config_;

    GroupNorm(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    static constexpr uint32_t kChunkLen = 4096;   // elements per workgroup; a multiple of 256
    vai::Dtype iDtype_ = vai::Dtype::None;
    uint32_t groupLen_ = 0, numChunks_ = 0, spatial_ = 0;
};


inline GroupNorm::GroupNorm(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
    addSlot("gamma", NodeSlot::persistent);
    addSlot("beta", NodeSlot::persistent);
    addSlot("partials", NodeSlot::temporary);

    VAI_ASSERT(config_.numGroups > 0);
    VAI_ASSERT(config_.epilogue < Epilogue::Max);
    config_.wDtype = (config_.wDtype == Dtype::None) ? wDefaultDtype() : config_.wDtype;
    config_.oDtype = (config_.oDtype == Dtype::None) ? oDefaultDtype() : config_.oDtype;
}

inline void GroupNorm::prepare()
{
    const Tensor& inTen = (*this)["in0"];
    VAI_ASSERT(inTen.shape().size() >= 2);
    const uint32_t C = inTen.shape()[0];
    const uint32_t S = uint32_t(inTen.numElements() / C);
    VAI_ASSERT(S > 0 && C % config_.numGroups == 0);
    VAI_ASSERT((*this)["gamma"].dtype() == config_.wDtype && (*this)["gamma"].isShapeOf(C));
    VAI_ASSERT((*this)["beta"].dtype() == config_.wDtype && (*this)["beta"].isShapeOf(C));

    if (!pipeStats || inTen.dtype() != iDtype_)
    {
        iDtype_ = inTen.dtype();
        pipeStats = pipelinePool().request("group-norm-stats", {
            eva::constant_id<SPEC_ID_TYPE_I>(iDtype_),
        });
        pipeApply = pipelinePool().request("group-norm-apply", {
            eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),
            eva::constant_id<SPEC_ID_TYPE_I>(iDtype_),
            eva::constant_id<SPEC_ID_TYPE_W>(config_.wDtype),
            eva::constant_id<SPEC_ID_EPILOGUE>(config_.epilogue),
        });
        if (!descStats)
        {
            descStats = gVaiDestSetPool(pipeStats.descSetLayout(0));
            descApply = gVaiDestSetPool(pipeApply.descSetLayout(0));
        }
    }

    spatial_ = S;
    groupLen_ = C / config_.numGroups * S;
    numChunks_ = CEIL_DIV(groupLen_, kChunkLen);
    (*this)["partials"] = Tensor({config_.numGroups, numChunks_, 2}, Dtype::Fp32);
    (*this)["out0"] = Tensor(inTen.shape(), config_.oDtype);
}

inline void GroupNorm::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf  = (*this)["in0"].buffer();
    auto pBuf   = (*this)["partials"].buffer();

    descStats.write({ pBuf, inBuf });
    descApply.write({ outBuf, inBuf, pBuf, (*this)["gamma"].buffer(), (*this)["beta"].buffer() });

    struct { uint32_t groupLen, chunkLen, numChunks; } stats = { groupLen_, kChunkLen, numChunks_ };
    struct {
        uint32_t groupLen, chunkLen, numChunks, spatial, chPerGroup;
        float eps;
    } apply = { groupLen_, kChunkLen, numChunks_, spatial_, groupLen_ / spatial_, config_.eps };

    cmdBuff
        .bindPipeline(pipeStats)
        .setPushConstants(0, sizeof(stats), &stats)
        .bindDescSets({descStats})
        .dispatch(numChunks_, config_.numGroups)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / pBuf / eva::SYNC_SCOPE::COMPUTE_READ)
        .bindPipeline(pipeApply)
        .setPushConstants(0, sizeof(apply), &apply)
        .bindDescSets({descApply})
        .dispatch(numChunks_, config_.numGroups)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace vai::node

#endif // VAI_NODES_GROUP_NORM_H
