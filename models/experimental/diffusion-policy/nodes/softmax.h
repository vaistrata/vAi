#ifndef EXPERIMENTAL_DIFFUSION_POLICY_NODES_SOFTMAX_H
#define EXPERIMENTAL_DIFFUSION_POLICY_NODES_SOFTMAX_H

#include "vai/nodes/common.h"

namespace model::experimental::diffusion_policy::node {


/*=============================================================================
  Softmax - softmax along one axis, single kernel (torch.nn.functional.softmax)
  Input : in0  [any shape]
  Output: out0 [same shape]

  O[o, :, i] = exp(I[o, :, i] - max) / Σ exp(I[o, :, i] - max)

  dim picks the axis as torch's does: from the front, or from the back when
  negative (-1: the last axis). The input is seen as [outer, D, inner], D the
  softmax axis, and one workgroup handles one (o, i) row; the max and the
  exp-sum are reduced through shared memory in fp32 regardless of I/O dtypes.

  The workgroup size depends on D (pow2ceil(D) in [subgroup, 256]), so the
  pipeline is requested in prepare() once the shape is known, and a change
  re-requests it from the pool.
=============================================================================*/
class Softmax : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams {
        uint32_t   wgSize = 0;   // threads per row
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        int32_t dim = -1;   // the softmax axis; negative counts from the back
    } config_;

    Softmax(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
    uint32_t rows_ = 0;    // outer * inner
    uint32_t D_ = 0;       // softmax axis length
    uint32_t inner_ = 0;   // elements past the axis: stride between a row's elements
};


// wgSize = pow2ceil(D) in [subgroup, 256]: a short axis (e.g. a 3x3 feature map)
// keeps one subgroup, a long one stays within the shared-tree budget.
inline uint32_t softmaxWgSize(uint32_t D, eva::Device device)
{
    const uint32_t sg = device.maxSubgroupSize();
    const uint32_t cap = std::max(sg, 256u);
    uint32_t wg = sg;
    while (wg < D && wg < cap) wg *= 2u;
    return wg;
}


inline Softmax::Softmax(Config config)
: config_(config)
{
    addSlot("out0", vai::NodeSlot::output);
    addSlot("in0", vai::NodeSlot::input);
}

inline void Softmax::prepare()
{
    const vai::Dtype oDtype = (oDtype_ == vai::Dtype::None) ? oDefaultDtype() : oDtype_;
    const vai::Tensor& inTen = (*this)["in0"];
    VAI_ASSERT(inTen.validShape());

    const auto& shape = inTen.shape();
    const int32_t rank = int32_t(shape.size());
    const int32_t dim = config_.dim < 0 ? config_.dim + rank : config_.dim;
    VAI_ASSERT(rank >= 1 && dim >= 0 && dim < rank);

    D_ = shape[dim];
    inner_ = 1;
    for (int32_t a = dim + 1; a < rank; ++a)
        inner_ *= shape[a];
    rows_ = uint32_t(inTen.numElements() / D_);
    // run() folds rows over grid x/y, as LayerNorm does.
    VAI_ASSERT(uint64_t(rows_) <= uint64_t(device().maxComputeWorkGroupCount()[0])
                                  * device().maxComputeWorkGroupCount()[1]);

    const PipeParams p = { .wgSize = softmaxWgSize(D_, device()), .iDtype = inTen.dtype() };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        // clang-format off
        pipe = pipelinePool().request(EXPERIMENTAL_DIFFUSION_POLICY_SHADER_OUTPUT_DIR, "softmax", {
            eva::constant_id<0>(p.wgSize),                 // wgSize
            eva::constant_id<SPEC_ID_TYPE_O>(oDtype),      // typeO
            eva::constant_id<SPEC_ID_TYPE_I>(p.iDtype)     // typeI
        });
        // clang-format on
        if (!descSet)
            descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
    }

    (*this)["out0"] = vai::Tensor(shape, oDtype);
}

inline void Softmax::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf = (*this)["in0"].buffer();

    descSet.write({ outBuf, inBuf });

    // One workgroup per row, rows folded over grid x/y -- x alone caps at 65535
    // on some devices (Intel).
    const uint32_t limX = device().maxComputeWorkGroupCount()[0];
    const uint32_t gy   = std::max(1u, CEIL_DIV(rows_, limX));
    const uint32_t gx   = CEIL_DIV(rows_, gy);

    struct {
        uint32_t rows, D, inner, gridX;
    } constants = { rows_, D_, inner_, gx };

    cmdBuff
        .bindPipeline(pipe)
        .setPushConstants(0, sizeof(constants), &constants)
        .bindDescSets({descSet})
        .dispatch(gx, gy)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace model::experimental::diffusion_policy::node

#endif // EXPERIMENTAL_DIFFUSION_POLICY_NODES_SOFTMAX_H
