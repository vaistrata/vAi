#ifndef EXPERIMENTAL_DIFFUSION_POLICY_NODES_GROUP_NORM_MISH_H
#define EXPERIMENTAL_DIFFUSION_POLICY_NODES_GROUP_NORM_MISH_H

#include "vai/nodes/common.h"

namespace model::experimental::diffusion_policy::node {


// The FiLM a GroupNormMish applies after its Mish: lerobot's
// use_film_scale_modulation picks ScaleShift, otherwise Shift.
enum class Film : uint32_t {
    None,
    Shift,        // y + film[c],                  film [..., C]
    ScaleShift,   // film[c] · y + film[C + c],    film [..., 2C]
};


/*=============================================================================
  GroupNormMish - GroupNorm, Mish, then FiLM and a residual, over [C, T]
  Input : in0      [..., C, T]
          film     [..., C] or [..., 2C], one row per sample (Film::Shift / ScaleShift)
          residual [..., C, T] (acceptResidual)
  Output: out0     [..., C, T]

  y = mish(scale[c] · (in0 − μ[g(c)]) · rsqrt(σ²[g(c)] + eps) + shift[c])
  y = film[c] · y + film[C + c]   (ScaleShift)  /  y + film[c]   (Shift)
  out0 = y + residual

  The tail of lerobot's DiffusionConv1dBlock (GroupNorm(groups, C), Mish) with
  what its residual block does right after: FiLM after the first block, the
  residual after the second. One pass, so neither needs a node or a tensor of
  its own. scale / shift [C] are the GroupNorm's weight / bias (parameters);
  the statistics are per (sample, group) over C/groups channels and T, two
  passes in fp32. Axes ahead of C are samples, each with its own film row.

  One workgroup per (group, sample): a group is a contiguous run of (C/groups)·T.
=============================================================================*/
class GroupNormMish : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;
        vai::Dtype wDtype = vai::Dtype::None;   // scale / shift
        vai::Dtype fDtype = vai::Dtype::None;   // film
        vai::Dtype rDtype = vai::Dtype::None;   // residual

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        uint32_t groups;
        float eps = 1e-5f;
        Film film = Film::None;        // input "film"
        bool acceptResidual = false;   // input "residual"
    } config_;

    GroupNormMish(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
    uint32_t C_ = 0, T_ = 0, samples_ = 0;
};


inline GroupNormMish::GroupNormMish(Config config)
: config_(config)
{
    VAI_ASSERT(config_.groups != 0u);
    addSlot("out0", vai::NodeSlot::output);
    addSlot("in0", vai::NodeSlot::input);
    addSlot("scale", vai::NodeSlot::persistent);
    addSlot("shift", vai::NodeSlot::persistent);
    if (config_.film != Film::None)
        addSlot("film", vai::NodeSlot::input);
    if (config_.acceptResidual)
        addSlot("residual", vai::NodeSlot::input);
}

inline void GroupNormMish::prepare()
{
    const vai::Dtype oDtype = (oDtype_ == vai::Dtype::None) ? oDefaultDtype() : oDtype_;
    const vai::Tensor& inTen = (*this)["in0"];
    const vai::Tensor& scaleTen = (*this)["scale"];
    const vai::Tensor& shiftTen = (*this)["shift"];
    VAI_ASSERT(inTen.validShape());

    const auto& shape = inTen.shape();
    VAI_ASSERT(shape.size() >= 2u);
    C_ = shape[shape.size() - 2u];
    T_ = shape[shape.size() - 1u];
    samples_ = uint32_t(inTen.numElements() / (size_t(C_) * T_));
    VAI_ASSERT(C_ % config_.groups == 0u);
    VAI_ASSERT(scaleTen.isShapeOf(C_) && shiftTen.isShapeOf(C_) && scaleTen.dtype() == shiftTen.dtype());

    PipeParams p = { .iDtype = inTen.dtype(), .wDtype = scaleTen.dtype() };
    if (config_.film != Film::None)
    {
        const vai::Tensor& filmTen = (*this)["film"];
        const uint32_t row = config_.film == Film::ScaleShift ? 2u * C_ : C_;
        VAI_ASSERT(filmTen.shape().back() == row && filmTen.numElements() == size_t(samples_) * row);
        p.fDtype = filmTen.dtype();
    }
    if (config_.acceptResidual)
    {
        const vai::Tensor& rTen = (*this)["residual"];
        VAI_ASSERT(rTen.numElements() == inTen.numElements());
        p.rDtype = rTen.dtype();
    }

    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        // clang-format off
        pipe = pipelinePool().request(EXPERIMENTAL_DIFFUSION_POLICY_SHADER_OUTPUT_DIR, "group-norm-mish", {
            eva::constant_id<0>((uint32_t) config_.film),                      // film
            eva::constant_id<1>(config_.acceptResidual ? 1u : 0u),            // hasResidual
            eva::constant_id<2>(p.fDtype == vai::Dtype::None ? vai::Dtype::Fp32 : p.fDtype),   // typeF
            eva::constant_id<3>(p.rDtype == vai::Dtype::None ? vai::Dtype::Fp32 : p.rDtype),   // typeR
            eva::constant_id<SPEC_ID_TYPE_O>(oDtype),                         // typeO
            eva::constant_id<SPEC_ID_TYPE_I>(p.iDtype),                       // typeI
            eva::constant_id<SPEC_ID_TYPE_W>(p.wDtype)                        // typeW (scale & shift)
        });
        // clang-format on
        if (!descSet)
            descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
    }

    (*this)["out0"] = vai::Tensor(shape, oDtype);
}

inline void GroupNormMish::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf = (*this)["in0"].buffer();
    // Unused slots bind an existing buffer; specialization removes every read.
    auto filmBuf = config_.film != Film::None ? (*this)["film"].buffer() : inBuf;
    auto rBuf = config_.acceptResidual ? (*this)["residual"].buffer() : inBuf;

    descSet.write({ outBuf, inBuf, (*this)["scale"].buffer(), (*this)["shift"].buffer(), filmBuf, rBuf });

    struct {
        uint32_t C, T, groups;
        float eps;
    } constants = { C_, T_, config_.groups, config_.eps };

    cmdBuff
        .bindPipeline(pipe)
        .setPushConstants(0, sizeof(constants), &constants)
        .bindDescSets({descSet})
        .dispatch(config_.groups, samples_)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace model::experimental::diffusion_policy::node

#endif // EXPERIMENTAL_DIFFUSION_POLICY_NODES_GROUP_NORM_MISH_H
