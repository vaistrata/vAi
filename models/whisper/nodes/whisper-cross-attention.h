#ifndef WHISPER_NODES_CROSS_ATTENTION_H
#define WHISPER_NODES_CROSS_ATTENTION_H

#include "vai/nodes/node-common.h"

namespace whisper::node {


/*=============================================================================
  CrossAttention - non-causal attention of new queries over a fixed key/value set
  Input : Q [..., H, Tq, headDim], K / V [..., H, Tk, headDim]
  Output: out0 [..., Tq, H, headDim]   (token-major, as vai::node::Attention)

  vai::node::Attention ties K / V to the queries' own tokens and appends them to
  its cache. Here K / V carry their own Tk and nothing is written back, so the
  caller computes them once (e.g. Whisper's encoder states) and feeds the same
  tensors every step.
=============================================================================*/
class CrossAttention : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    static constexpr uint32_t wgSize = 128;   // = WG_SIZE in the kernel

    struct PipeParams {
        uint32_t Tk = 0;
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        uint32_t headDim;
        uint32_t numHeads;
        vai::Dtype oDtype = vai::Dtype::None;   // None: oDefaultDtype
    } config_;

    CrossAttention(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
    uint32_t B = 0;    // sequences in the leading axes
    uint32_t Tq = 0;
};


inline CrossAttention::CrossAttention(Config config)
: config_(config)
{
    addSlot("out0", vai::NodeSlot::output);
    addSlot("Q", vai::NodeSlot::input);
    addSlot("K", vai::NodeSlot::input);
    addSlot("V", vai::NodeSlot::input);

    VAI_ASSERT(config_.headDim > 0 && config_.headDim <= wgSize);
    config_.oDtype = (config_.oDtype == vai::Dtype::None) ? oDefaultDtype() : config_.oDtype;
}

inline void CrossAttention::prepare()
{
    const vai::Tensor& Q = (*this)["Q"];
    const vai::Tensor& K = (*this)["K"];
    const vai::Tensor& V = (*this)["V"];

    const auto& qs = Q.shape();
    const auto& ks = K.shape();
    VAI_ASSERT(qs.size() >= 3 && ks.size() == qs.size());
    VAI_ASSERT(qs[qs.size() - 3] == config_.numHeads && qs.back() == config_.headDim);
    VAI_ASSERT(ks[ks.size() - 3] == config_.numHeads && ks.back() == config_.headDim);
    VAI_ASSERT(V.shape() == ks);
    VAI_ASSERT(K.dtype() == Q.dtype() && V.dtype() == Q.dtype());

    B  = vai::outerSize(qs, 3);
    Tq = qs[qs.size() - 2];
    VAI_ASSERT(vai::outerSize(ks, 3) == B);

    // Tk sizes the kernel's shared score buffer, so it is a spec constant
    const PipeParams p = {.Tk = ks[ks.size() - 2], .iDtype = Q.dtype()};
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        pipe = pipelinePool().request(WHISPER_SHADER_OUTPUT_DIR, "whisper-cross-attention", {
            eva::constant_id<0>(p.Tk),                         // maxTk
            eva::constant_id<1>(config_.headDim),              // Dh
            eva::constant_id<2>(p.iDtype),                     // typeI
            eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype)   // typeO
        });
        if (!descSet)
            descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
    }

    std::vector<uint32_t> outShape = qs;
    outShape[qs.size() - 3] = Tq;
    outShape[qs.size() - 2] = config_.numHeads;
    (*this)["out0"] = vai::Tensor(outShape, config_.oDtype);
}

inline void CrossAttention::run(eva::CommandBuffer cmdBuff)
{
    auto O = (*this)["out0"].buffer();

    struct {
        uint32_t Tq, Tk, H;
        float scale;
    } constants = {Tq, pipeParams.Tk, config_.numHeads, 1.0f / std::sqrt(float(config_.headDim))};

    descSet.write({O, (*this)["Q"].buffer(), (*this)["K"].buffer(), (*this)["V"].buffer()});
    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch(Tq, B * config_.numHeads, 1)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / O / eva::SYNC_SCOPE::COMPUTE_READ);
}


}   // namespace whisper::node

#endif // WHISPER_NODES_CROSS_ATTENTION_H
