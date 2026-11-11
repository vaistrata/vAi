#ifndef ZIMAGE_ADA_NORM_H
#define ZIMAGE_ADA_NORM_H

#include "vai/nodes/node-common.h"

namespace zimage {

/*=============================================================================
  AdaNormNode - RMS / layer norm with adaLN modulation and a gated residual
  Input : in0 [rows, F], residual [rows, F] (optional), mod [1, k*F] (optional)
  Params: gamma [F] (optional)
  Output: out0 [rows, F] = residual + tanh(gate) * norm(in0) * γ * (1 + scale)

  scale = mod[scaleOffset : scaleOffset + F], gate = mod[gateOffset : + F].
  Every factor is optional and drops out when its option is off, so the one
  node covers the four sites of a Z-Image block (the modulated pre-norms, the
  gated post-norm residuals, their unmodulated refiner forms) and the final
  layer's affine-free layer norm.

  center selects layer norm (x - mean) over RMS. The statistics accumulate in
  fp32; mod and residual must share in0's dtype.
=============================================================================*/
class AdaNormNode : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        float eps = 1e-5f;
        bool center = false;        // true: layer norm, false: RMS norm
        bool hasGamma = true;       // multiply by γ ("gamma" slot)
        bool hasScale = false;      // multiply by (1 + mod[scaleOffset + j])
        bool hasGate = false;       // multiply by tanh(mod[gateOffset + j])
        bool hasResidual = false;   // add "residual"
        uint32_t scaleOffset = 0;   // elements into mod
        uint32_t gateOffset = 0;    // elements into mod
        vai::Dtype wDtype = vai::Dtype::None;   // None: wDefaultDtype
        vai::Dtype oDtype = vai::Dtype::None;   // None: oDefaultDtype
    } config_;

    AdaNormNode(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    static constexpr uint32_t wgSize = 256;
    PipeParams pipeParams;
};


inline AdaNormNode::AdaNormNode(Config config)
: config_(config)
{
    addSlot("out0", vai::NodeSlot::output);
    addSlot("in0", vai::NodeSlot::input);
    if (config_.hasResidual) addSlot("residual", vai::NodeSlot::input);
    if (config_.hasScale || config_.hasGate) addSlot("mod", vai::NodeSlot::input);
    if (config_.hasGamma) addSlot("gamma", vai::NodeSlot::persistent);

    config_.wDtype = (config_.wDtype == vai::Dtype::None) ? wDefaultDtype() : config_.wDtype;
    config_.oDtype = (config_.oDtype == vai::Dtype::None) ? oDefaultDtype() : config_.oDtype;
}

inline void AdaNormNode::prepare()
{
    const vai::Tensor& inTen = (*this)["in0"];
    const vai::Dtype iDtype = inTen.dtype();
    VAI_ASSERT(inTen.shape().size() == 2);
    const uint32_t F = inTen.shape()[1];

    if (config_.hasResidual)
    {
        const vai::Tensor& rTen = (*this)["residual"];
        VAI_ASSERT(rTen.dtype() == iDtype);
        VAI_ASSERT(rTen.shape() == inTen.shape());
    }
    if (config_.hasScale || config_.hasGate)
    {
        const vai::Tensor& mTen = (*this)["mod"];
        VAI_ASSERT(mTen.dtype() == iDtype);
        if (config_.hasScale) VAI_ASSERT(config_.scaleOffset + F <= mTen.numElements());
        if (config_.hasGate)  VAI_ASSERT(config_.gateOffset + F <= mTen.numElements());
    }
    if (config_.hasGamma)
    {
        const vai::Tensor& gTen = (*this)["gamma"];
        VAI_ASSERT(gTen.dtype() == config_.wDtype);
        VAI_ASSERT(gTen.isShapeOf(F));
    }

    const PipeParams p = { .iDtype = iDtype };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        pipe = pipelinePool().request(ZIMAGE_SHADER_OUTPUT_DIR, "zimage-ada-norm", {
            eva::constant_id<0>(wgSize),
            eva::constant_id<1>(uint32_t(config_.center)),
            eva::constant_id<2>(uint32_t(config_.hasGamma)),
            eva::constant_id<3>(uint32_t(config_.hasScale)),
            eva::constant_id<4>(uint32_t(config_.hasGate)),
            eva::constant_id<5>(uint32_t(config_.hasResidual)),
            eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),          // typeO
            eva::constant_id<SPEC_ID_TYPE_I>(iDtype),                  // typeI (in0, residual, mod)
            eva::constant_id<SPEC_ID_TYPE_W>(config_.wDtype)           // typeW (γ)
        });
        if (!descSet)
            descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
    }

    (*this)["out0"] = vai::Tensor(inTen.shape(), config_.oDtype);
}

inline void AdaNormNode::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf  = (*this)["in0"].buffer();
    // Absent operands bind in0 as a stand-in; their spec flags fold the reads away.
    auto rBuf = config_.hasResidual ? (*this)["residual"].buffer() : inBuf;
    auto gBuf = config_.hasGamma ? (*this)["gamma"].buffer() : inBuf;
    auto mBuf = (config_.hasScale || config_.hasGate) ? (*this)["mod"].buffer() : inBuf;

    descSet.write({ outBuf, inBuf, rBuf, gBuf, mBuf });

    const auto& shape = (*this)["in0"].shape();
    struct {
        uint32_t rows, F;
        float eps;
        uint32_t scaleOffset, gateOffset;
    } constants = { shape[0], shape[1], config_.eps, config_.scaleOffset, config_.gateOffset };

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch(constants.rows)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

} // namespace zimage

#endif // ZIMAGE_ADA_NORM_H
