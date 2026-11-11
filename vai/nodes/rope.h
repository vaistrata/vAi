#ifndef VAI_NODES_ROPE_H
#define VAI_NODES_ROPE_H

#include "node-common.h"

namespace vai::node {

/*=============================================================================
  Rope - rotary position embedding (half-split rotation)
  Input:  in0 [.., H, Tq, Dh]   (Q or K, Dh even)
  Params: freqs [Dh/2]          per-dimension angular frequency, built by the constructor
  Output: out0 [.., H, Tq, Dh]

  The rotation depends on the token position alone, so anything ahead of the head
  axis is one more sweep and folds into the head count.

  Positions continue from setKvCacheCounter()'s buffer, so a KV-cache decode
  step rotates by the absolute position rather than by its offset in this forward.

  cfg.type picks how the frequencies are scaled: Default leaves them alone,
  Linear divides every one by factor, and Llama3 divides only the long
  wavelengths, interpolating the band between the two frequency bounds.
=============================================================================*/
class Rope : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

    eva::Buffer kvCacheCounter;

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        RopeConfig cfg;         // frequencies; describe the whole model
        uint32_t headDim;       // Dh, must be even
    } config_;

    Rope(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

    void setKvCacheCounter(eva::Buffer buffer) { kvCacheCounter = buffer; }

private:
    PipeParams pipeParams;
};


inline Rope::Rope(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
    addSlot("freqs", NodeSlot::persistent);

    VAI_ASSERT(config_.headDim >= 2 && config_.headDim % 2 == 0);


    // freqs[d] depends only on the config, so it is built once here. Kept fp32
    // whatever the activations are: it is tiny and the angles are precision critical.
    const uint32_t halfDh = config_.headDim / 2;
    std::vector<float> freqData(halfDh);
    {
        const float PI = 3.14159265358979323846f;
        const float log2Base = std::log2f(config_.cfg.base);
        for (uint32_t d = 0; d < halfDh; ++d)
        {
            float freq = std::exp2f(-(float(d) / float(halfDh)) * log2Base);

            if (config_.cfg.type == RopeType::Linear)
            {
                freq /= config_.cfg.factor;
            }
            else if (config_.cfg.type == RopeType::Llama3)
            {
                float wavelength = 2.0f * PI / freq;

                // 0 at the low-frequency end, 1 at the high-frequency end.
                float t = std::fmax(0.0f, std::fmin(1.0f,
                    (float(config_.cfg.originalMaxPos) / wavelength - config_.cfg.lowFreqFactor)
                    / (config_.cfg.highFreqFactor - config_.cfg.lowFreqFactor)));

                freq = (freq / config_.cfg.factor) * (1.0f - t) + freq * t;
            }

            freqData[d] = freq;
        }
    }
    (*this)["freqs"] = Tensor({halfDh}, Dtype::Fp32).setFloats(std::move(freqData));
}

inline void Rope::prepare()
{
    const Dtype oDtype = (oDtype_ == Dtype::None) ? oDefaultDtype() : oDtype_;
    const Tensor& inTen = (*this)["in0"];

    const Dtype iDtype = inTen.dtype();

    const PipeParams p = { .iDtype = iDtype };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        pipe = pipelinePool().request("rope", {
                                                  eva::constant_id<SPEC_ID_TYPE_O>(oDtype),   // typeO
                                                  eva::constant_id<SPEC_ID_TYPE_I>(iDtype)    // typeI
                                              });
        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }

    VAI_ASSERT(inTen.shape().size() >= 3);
    VAI_ASSERT(inTen.shape().back() == config_.headDim);

    (*this)["out0"] = Tensor(inTen.shape(), oDtype);
}

inline void Rope::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf   = (*this)["out0"].buffer();
    auto inBuf    = (*this)["in0"].buffer();
    auto freqsBuf = (*this)["freqs"].buffer();

    // Single set: 0 = O(out), 1 = I(in), 2 = freqs, 3 = kvCacheCount (uniform).
    descSet.write({ outBuf, inBuf, freqsBuf, kvCacheCounter });

    const auto& shape = (*this)["in0"].shape();
    uint32_t H = outerSize(shape, 2);

    struct {
        uint32_t H, Tq, Dh;
    } constants = { H, shape[shape.size() - 2], shape.back() };

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch2(constants.Dh / 2, constants.Tq, constants.H)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace vai::node

#endif // VAI_NODES_ROPE_H
