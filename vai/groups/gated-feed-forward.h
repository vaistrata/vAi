#ifndef VAI_GROUPS_GATED_FEED_FORWARD_H
#define VAI_GROUPS_GATED_FEED_FORWARD_H

#include "vai/nodes/linear.h"
#include "vai/nodes/gated-mul.h"

namespace vai::group {


/////////////////////////////////////////////////////////////////////////////////////////
// GatedFeedForward - gated MLP, one projection feeding both GLU halves
// Input: in0 [..., inDim], optional residual [..., outDim]
// Output: out0 [..., outDim] = out(activation(gate(in0)) * value(in0))
//
// Gate and value come out of ONE projection over the concatenated weights, so
// the hidden width is doubled once rather than projected twice. activation
// names the GLU variant — silu: SwiGLU, gelu: GEGLU.
//
// acceptResidual opens "residual" for the block's skip and folds the add into the
// output projection. A block that normalizes its branch output has to add
// outside instead, and leaves it false: no slot is opened.
//
// fuseGlu runs the gating inside gateValue (Linear's glu: in the GEMM's store
// on Xe3, a second pass elsewhere) and leaves gatedMul out of the graph.
/////////////////////////////////////////////////////////////////////////////////////////
struct GatedFeedForward : public vai::NodeGroup
{
    vai::node::Linear gateValue;    // [..., inDim] -> [..., 2 * hiddenDim]
    vai::node::GatedMul gatedMul;   // [..., 2 * hiddenDim] -> [..., hiddenDim]
    vai::node::Linear out;          // [..., hiddenDim] -> [..., outDim]

    struct Config {
        uint32_t inDim;
        uint32_t hiddenDim;
        uint32_t outDim;
        vai::ActType actType = vai::ActType::Silu;   // the GLU variant
        bool acceptResidual = false;   // open "residual" and fold its addend into the output projection
        bool fuseGlu = false;          // the gating inside gateValue (Linear::Config::glu), no gatedMul node
    };

    GatedFeedForward(Config config)
    : gateValue({
        .inDim = config.inDim,
        .outDim = 2 * config.hiddenDim,
        .actType = config.fuseGlu ? config.actType : vai::ActType::None,   // fused: the gate's
        .glu = config.fuseGlu})
    , gatedMul({config.actType})
    , out({
        .inDim = config.hiddenDim,
        .outDim = config.outDim,
        .acceptResidual = config.acceptResidual})
    {
        // clang-format off
        if (config.fuseGlu)
            gateValue - out;
        else
            gateValue - gatedMul - out;
        // clang-format on

        defineSlot("in0", gateValue.slot("in0"));
        if (config.acceptResidual)
            defineSlot("residual", out.slot("residual"));
        defineSlot("out0", out.slot("out0"));
    }
};


}   // namespace vai::group

#endif // VAI_GROUPS_GATED_FEED_FORWARD_H
