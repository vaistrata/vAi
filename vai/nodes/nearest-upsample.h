#ifndef VAI_NODES_NEAREST_UPSAMPLE_H
#define VAI_NODES_NEAREST_UPSAMPLE_H

#include "node-common.h"

namespace vai::node {


/*=============================================================================
  NearestUpsample - Replicates CHW pixels by an integer scale factor.
  Input:  in0  [C, H, W]
  Output: out0 [C, H*scale, W*scale]

  scaleFactor must be greater than one. Each input pixel is replicated without
  interpolation.
=============================================================================*/
class NearestUpsample : public vai::Node
{
    uint32_t scale;

    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config
    {
    } config_;

    NearestUpsample(uint32_t scaleFactor = 2);
    NearestUpsample(uint32_t scaleFactor, Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};


inline NearestUpsample::NearestUpsample(uint32_t scaleFactor) : NearestUpsample(scaleFactor, Config()) {}

inline NearestUpsample::NearestUpsample(uint32_t scaleFactor, Config config)
: scale(scaleFactor)
, config_(config)
{
    VAI_ASSERT(scale > 1u);

    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
}

inline void NearestUpsample::prepare()
{
    const Dtype oDtype = (oDtype_ == Dtype::None) ? oDefaultDtype() : oDtype_;
    const Tensor& inTen = (*this)["in0"];

    const Dtype iDtype = inTen.dtype();

    const PipeParams p = { .iDtype = iDtype };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        pipe = pipelinePool().request("upsample", {
                                                      eva::constant_id<0>(scale),
                                                      eva::constant_id<SPEC_ID_TYPE_O>(oDtype),   // typeO
                                                      eva::constant_id<SPEC_ID_TYPE_I>(iDtype)    // typeI
                                                  });
        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }
    VAI_ASSERT(inTen.shape().size() == 3);

    const auto& s = inTen.shape();
    (*this)["out0"] = Tensor({s[0], s[1] * scale, s[2] * scale}, oDtype);
}

inline void NearestUpsample::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf  = (*this)["in0"].buffer();

    descSet.write({ outBuf, inBuf });

    const auto& s = (*this)["in0"].shape();
    uint32_t C = s[0];
    uint32_t Hi = s[1];
    uint32_t Wi = s[2];

    struct {
        uint32_t C, Hi, Wi;
    } constants = { C, Hi, Wi };

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch2(Wi, Hi, C)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace vai::node

#endif // VAI_NODES_NEAREST_UPSAMPLE_H
