#ifndef VAI_NODES_ROPE_H
#define VAI_NODES_ROPE_H

#include "common.h"

namespace vai::node {

/*=============================================================================
  Rope - rotary position embedding (half-split rotation)
  Input:  in0 [..., H, Tq, Dh]   (Q or K, Dh even)
  Params: freqs [Dh/2]   per-pair angular frequency, built by the constructor
  Output: out0 [..., H, Tq, Dh]

  The rotation depends on the token position alone, so anything ahead of the head
  axis is one more sweep and folds into the head count.

  The values in cfg pick how the frequencies are scaled: factor 1 leaves them
  alone, any other divides every one, and a nonzero originalMaxPos makes it
  Llama3's, dividing only the long wavelengths and interpolating the band between
  the two frequency bounds.

  cfg.posType fixes where positions come from. setPosParam() binds one uint, read
  at dispatch time, whose meaning the type gives:
    Sequential  token tq at posParam + tq, posParam the first token's position
    Grid2D      tokens are a raster grid posParam wide, row and column from tq
    MRoPE       each token's (t, h, w) from the buffer setPositions() binds;
                rope.comp picks the pair's axis from the sections
=============================================================================*/
class Rope : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

    eva::Buffer posParam;
    eva::Buffer positions;

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

    void setPosParam(eva::Buffer buffer) { posParam = buffer; }
    void setPositions(eva::Buffer buffer) { positions = buffer; }

private:
    PipeParams pipeParams;
};


inline Rope::Rope(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
    addSlot("freqs", NodeSlot::persistent);

    const RopeConfig& cfg = config_.cfg;
    const auto& section = cfg.mropeSection;
    const uint32_t halfDh = config_.headDim / 2;
    VAI_ASSERT(config_.headDim >= 2 && config_.headDim % 2 == 0);
    VAI_ASSERT(cfg.posType != RopePosType::Grid2D || halfDh % 2 == 0);
    VAI_ASSERT(cfg.posType != RopePosType::MRoPE || section[0] + section[1] + section[2] == halfDh);

    // freqs depends only on the config, so it is built once here. Kept fp32 whatever
    // the activations are: it is tiny and the angles are precision critical.
    // Grid2D runs the frequencies of headDim / 2 once per axis; rope.comp splits the
    // pairs between row and column itself.
    const uint32_t span = (cfg.posType == RopePosType::Grid2D) ? halfDh / 2 : halfDh;
    std::vector<float> freqData(halfDh);
    {
        const float PI = 3.14159265358979323846f;
        const float log2Base = std::log2f(cfg.base);
        for (uint32_t d = 0; d < halfDh; ++d)
        {
            float freq = std::exp2f(-(float(d % span) / float(span)) * log2Base);

            if (cfg.originalMaxPos > 0)
            {
                float wavelength = 2.0f * PI / freq;

                // 0 at the low-frequency end, 1 at the high-frequency end.
                float t = std::fmax(
                    0.0f,
                    std::fmin(
                        1.0f,
                        (float(cfg.originalMaxPos) / wavelength - cfg.lowFreqFactor) /
                            (cfg.highFreqFactor - cfg.lowFreqFactor)));

                freq = (freq / cfg.factor) * (1.0f - t) + freq * t;
            }
            else
            {
                freq /= cfg.factor;
            }

            freqData[d] = freq;
        }
    }
    (*this)["freqs"] = Tensor({ halfDh }, Dtype::Fp32).setFloats(std::move(freqData));
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
        const auto& section = config_.cfg.mropeSection;
        // clang-format off
        pipe = pipelinePool().request("rope", {
            eva::constant_id<SPEC_ID_TYPE_O>(oDtype),   // typeO
            eva::constant_id<SPEC_ID_TYPE_I>(iDtype),           // typeI
            eva::constant_id<0>((uint32_t) config_.cfg.posType),   // posType
            eva::constant_id<1>(section[1]),                    // sectionH
            eva::constant_id<2>(section[2]),                    // sectionW
            eva::constant_id<3>(config_.cfg.tileSize)           // tileSize
        });
        // clang-format on
        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }

    VAI_ASSERT(inTen.shape().size() >= 3);
    VAI_ASSERT(inTen.shape().back() == config_.headDim);

    const uint32_t Tq = inTen.shape()[inTen.shape().size() - 2];
    if (config_.cfg.posType == RopePosType::MRoPE)
        VAI_ASSERT(positions && positions.size() >= uint64_t(Tq) * 3 * sizeof(uint32_t));
    if (config_.cfg.posType == RopePosType::Grid2D)
    {
        VAI_ASSERT(posParam && "Rope: setPosParam() was never called");
        const uint32_t gridW = *(uint32_t*)posParam.map();
        const uint32_t T = config_.cfg.tileSize;
        VAI_ASSERT(gridW > 0 && Tq % gridW == 0 && gridW % T == 0 && (Tq / gridW) % T == 0);
    }

    (*this)["out0"] = Tensor(inTen.shape(), oDtype);
}

inline void Rope::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf   = (*this)["out0"].buffer();
    auto inBuf    = (*this)["in0"].buffer();
    auto freqsBuf = (*this)["freqs"].buffer();

    // Single set: 0 = O(out), 1 = I(in), 2 = freqs, 3 = posParam (uniform), 4 = positions.
    // What the position type does not read stays null.
    const bool mrope = config_.cfg.posType == RopePosType::MRoPE;
    descSet.write({
        outBuf,
        inBuf,
        freqsBuf,
        mrope ? eva::Buffer() : posParam,
        mrope ? positions : eva::Buffer(),
    });

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
