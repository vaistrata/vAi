#ifndef VAI_NODES_ELE_AFFINE_H
#define VAI_NODES_ELE_AFFINE_H

#include "node-common.h"

namespace vai::node {


/*=============================================================================
  EleAffine - element-wise affine, O = scale * in0 + shift
  Input: in0    [..., D]
         scale  broadcast to in0's shape (right-aligned, size-1 axes broadcast);
                scale 1 when absent
         shift  likewise; shift 0 when absent
  Output: out0  in0's shape

  scale / shift are inputs under scaleAsInput / shiftAsInput, otherwise
  parameters at the loader's dtype. Each applies when present: filled by the
  loader, or wired; at least one has to be. Each may broadcast over any axes as long as the axes it keeps
  are adjacent, e.g. [1, D] over [T, D], or [C, 1, 1] over [C, H, W].

  An add is shift alone, a multiply scale alone, and a gated residual x + g * y
  both, EleAffine(in0 = y, scale = g, shift = x) (e.g. pi0.5's adaRMS layers).
=============================================================================*/
class EleAffine : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    // An operand at element e of O reads (e / inner) % count; count 0 skips the
    // modulo (the kept axes run to the outermost one).
    struct Broadcast {
        uint32_t inner = 1;
        uint32_t count = 0;

        bool operator==(const Broadcast&) const = default;
    };

    struct PipeParams {
        Broadcast  scale;
        Broadcast  shift;
        bool       aligned    = false;
        vai::Dtype iDtype     = vai::Dtype::None;
        vai::Dtype scaleDtype = vai::Dtype::None;
        vai::Dtype shiftDtype = vai::Dtype::None;
        bool hasScale = false;
        bool hasShift = false;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        bool scaleAsInput = false;   // input instead of persistent
        bool shiftAsInput = false;
    } config_;

    EleAffine(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

    // The loader filled it, or under scaleAsInput / shiftAsInput it is wired.
    bool hasScale() const { return (*this)["scale"].validShape(); }
    bool hasShift() const { return (*this)["shift"].validShape(); }

private:
    PipeParams pipeParams;
};


// Collapses an operand's broadcast against O into (inner, count); asserts the
// kept axes are adjacent.
inline EleAffine::Broadcast eleAffineBroadcast(const std::vector<uint32_t>& o, const std::vector<uint32_t>& s)
{
    VAI_ASSERT(s.size() <= o.size());
    const size_t pad = o.size() - s.size();

    int first = -1, last = -1;   // the run of kept axes; size-1 axes of O take either side
    bool ended = false;
    for (size_t d = 0; d < o.size(); ++d)
    {
        const uint32_t sd = (d < pad) ? 1u : s[d - pad];
        VAI_ASSERT(sd == o[d] || sd == 1u);
        if (o[d] == 1u) continue;
        if (sd == o[d])
        {
            VAI_ASSERT(!ended);   // one contiguous run
            if (first < 0) first = int(d);
            last = int(d);
        }
        else if (first >= 0)
            ended = true;
    }

    EleAffine::Broadcast b;
    if (first < 0) { b.count = 1; return b; }   // a scalar

    for (size_t d = size_t(last) + 1; d < o.size(); ++d) b.inner *= o[d];

    bool outermost = true;
    for (int d = 0; d < first; ++d) outermost &= (o[d] == 1u);
    if (!outermost)
    {
        b.count = 1;
        for (int d = first; d <= last; ++d) b.count *= o[d];
    }
    return b;
}


inline EleAffine::EleAffine(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
    addSlot("scale", config_.scaleAsInput ? NodeSlot::input : NodeSlot::persistent);
    addSlot("shift", config_.shiftAsInput ? NodeSlot::input : NodeSlot::persistent);
}

inline void EleAffine::prepare()
{
    const Dtype oDtype = (oDtype_ == Dtype::None) ? oDefaultDtype() : oDtype_;
    const Tensor& inTen    = (*this)["in0"];
    const Tensor* scaleTen = hasScale() ? &(*this)["scale"] : nullptr;
    const Tensor* shiftTen = hasShift() ? &(*this)["shift"] : nullptr;
    VAI_ASSERT(inTen.validShape());
    VAI_ASSERT(scaleTen || shiftTen);   // neither is a TypeCast

    const uint32_t N = inTen.numElements();

    auto broadcastOf = [&](const Tensor* t)
    {
        return t ? eleAffineBroadcast(inTen.shape(), t->shape()) : Broadcast{};
    };

    // vec4 threads: an operand either holds one value across each vec4 (inner a
    // multiple of 4, read once and splatted) or runs along it (inner 1, whole
    // vec4s per period, a 16-byte-aligned view).
    auto vec4Ok = [](const Tensor* t, const Broadcast& b)
    {
        if (!t || b.inner % 4u == 0) return true;
        return b.inner == 1u && b.count % 4u == 0 && t->byteOffset() % (4u * bytesPerElement(t->dtype())) == 0;
    };

    PipeParams p = {
        .scale = broadcastOf(scaleTen),
        .shift = broadcastOf(shiftTen),
        .iDtype = inTen.dtype(),
        .scaleDtype = scaleTen ? scaleTen->dtype() : Dtype::None,
        .shiftDtype = shiftTen ? shiftTen->dtype() : Dtype::None,
        .hasScale = scaleTen != nullptr,
        .hasShift = shiftTen != nullptr,
    };
    p.aligned = N % 4u == 0 && inTen.byteOffset() % (4u * bytesPerElement(inTen.dtype())) == 0
             && vec4Ok(scaleTen, p.scale) && vec4Ok(shiftTen, p.shift);
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        pipe = pipelinePool().request("ele-affine",
                                      {
                                          eva::constant_id<0>(p.scale.inner),                      // scaleInner
                                          eva::constant_id<1>(p.scale.count),                      // scaleCount
                                          eva::constant_id<2>(p.shift.inner),                      // shiftInner
                                          eva::constant_id<3>(p.shift.count),                      // shiftCount
                                          eva::constant_id<4>(p.hasScale ? 1u : 0u),               // hasScale
                                          eva::constant_id<5>(p.hasShift ? 1u : 0u),               // hasShift
                                          eva::constant_id<6>(p.scaleDtype),                       // typeScale
                                          eva::constant_id<7>(p.shiftDtype),                       // typeShift
                                          eva::constant_id<SPEC_ID_TYPE_O>(oDtype),                // typeO
                                          eva::constant_id<SPEC_ID_TYPE_I>(p.iDtype),              // typeI
                                          eva::constant_id<SPEC_ID_ALIGNED>(p.aligned ? 1u : 0u)   // aligned (vec4)
                                      });
        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }

    (*this)["out0"] = Tensor(inTen.shape(), oDtype);
}

inline void EleAffine::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf   = (*this)["out0"].buffer();
    auto inBuf    = (*this)["in0"].buffer();
    auto scaleBuf = pipeParams.hasScale ? (*this)["scale"].buffer() : eva::Buffer();
    auto shiftBuf = pipeParams.hasShift ? (*this)["shift"].buffer() : eva::Buffer();

    descSet.write({ outBuf, inBuf, scaleBuf, shiftBuf });

    // One thread per element (vec4 when aligned), workgroups folded over grid
    // x/y — x alone caps at 65535 on some devices (Intel).
    constexpr uint32_t wgSize = 256;   // = local_size_x of ele-affine
    const uint32_t N      = (*this)["in0"].numElements();
    const uint32_t items  = pipeParams.aligned ? N / 4u : N;
    const uint32_t groups = CEIL_DIV(items, wgSize);
    const uint32_t limX   = device().maxComputeWorkGroupCount()[0];
    const uint32_t gy     = std::max(1u, CEIL_DIV(groups, limX));
    const uint32_t gx     = CEIL_DIV(groups, gy);

    struct {
        uint32_t items;   // threads with work: N, or N / 4 when aligned
        uint32_t gridX;   // workgroups along x, so the shader can undo the fold
    } constants = { items, gx };

    cmdBuff
        .bindPipeline(pipe)
        .setPushConstants(0, sizeof(constants), &constants)
        .bindDescSets({descSet})
        .dispatch(gx, gy)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace vai::node

#endif // VAI_NODES_ELE_AFFINE_H
