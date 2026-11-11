#ifndef VAI_NODES_TIME_EMBEDDING_H
#define VAI_NODES_TIME_EMBEDDING_H

#include "node-common.h"
#include <vector>

namespace vai::node {

/*=============================================================================
  TimeEmbedding - sinusoidal embedding of a timestep the node counts down itself
  Input:  in0  [.., rows, *]    only its shape is read: one embedding per row
  Output: out0 [.., rows, embedDim]

  out0[r, i]        = sin(2*pi * t / p_i)
  out0[r, half + i] = cos(2*pi * t / p_i),   p_i = minPeriod * (maxPeriod / minPeriod)^(i / (half - 1))

  Run k after resetTime() embeds t = 1 - (k % numSteps) / numSteps, so a
  flow-matching loop of numSteps steps walks t from 1 down with no host write.
  resetTime() starts it over (e.g. pi0 after a prefill); the count wraps after
  2^32 workgroups, so a caller that never resets does so within that.
=============================================================================*/
class TimeEmbedding : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;
    eva::Buffer ticketBuffer;           // workgroups run since resetTime()

public:
    struct Config {
        uint32_t embedDim;              // even; sin in the first half, cos in the second
        float minPeriod;                // the fastest sinusoid, in units of t
        float maxPeriod;                // ... and the slowest
        uint32_t numSteps;              // runs before t is back at 1
    } config_;

    TimeEmbedding(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

    void resetTime() { *(uint32_t*)ticketBuffer.map() = 0; }
};


inline TimeEmbedding::TimeEmbedding(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);

    VAI_ASSERT(config_.embedDim >= 4 && config_.embedDim % 2 == 0);
    VAI_ASSERT(config_.minPeriod > 0.0f && config_.maxPeriod > config_.minPeriod);
    VAI_ASSERT(config_.numSteps >= 1);

    // Device-local, so the atomics stay in VRAM; host-visible, so resetTime() can write it.
    ticketBuffer = device().createBuffer({
        .size = sizeof(uint32_t),
        .usage = eva::BUFFER_USAGE::STORAGE_BUFFER,
        .reqMemProps = eva::MEMORY_PROPERTY::DEVICE_LOCAL
                     | eva::MEMORY_PROPERTY::HOST_VISIBLE | eva::MEMORY_PROPERTY::HOST_COHERENT
    });
    resetTime();
}

inline void TimeEmbedding::prepare()
{
    const Dtype oDtype = (oDtype_ == Dtype::None) ? oDefaultDtype() : oDtype_;
    const Tensor& inTen = (*this)["in0"];
    VAI_ASSERT(inTen.validShape());
    VAI_ASSERT(inTen.shape().size() >= 2);


    if (!pipe)
    {
        pipe = pipelinePool().request("time-embedding", {
                                                            eva::constant_id<SPEC_ID_TYPE_O>(oDtype)   // typeO
                                                        });
        descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }

    std::vector<uint32_t> outShape = inTen.shape();
    outShape.back() = config_.embedDim;
    (*this)["out0"] = Tensor(std::move(outShape), oDtype);
}

inline void TimeEmbedding::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();

    // Single set: 0 = O(out), 1 = ticket.
    descSet.write({ outBuf, ticketBuffer });

    struct {
        uint32_t rows, D;
        float minPeriod, log2Ratio;
        uint32_t numSteps;
    } constants = { outerSize((*this)["out0"].shape()), config_.embedDim,
                    config_.minPeriod, std::log2(config_.maxPeriod / config_.minPeriod),
                    config_.numSteps };

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch2(constants.D / 2, constants.rows)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}


// The same embedding of one t on the host, for a model that computes every
// step's time condition at load (e.g. pi0, pi0.5).
inline std::vector<float> timeEmbedding(double t, uint32_t embedDim, double minPeriod, double maxPeriod)
{
    const uint32_t half = embedDim / 2;
    std::vector<float> out(embedDim);
    for (uint32_t i = 0; i < half; ++i)
    {
        const double period = minPeriod * std::pow(maxPeriod / minPeriod, double(i) / (half - 1));
        const double angle = 2.0 * 3.14159265358979323846 * t / period;
        out[i] = float(std::sin(angle));
        out[half + i] = float(std::cos(angle));
    }
    return out;
}

}   // namespace vai::node

#endif // VAI_NODES_TIME_EMBEDDING_H
