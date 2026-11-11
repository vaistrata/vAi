#ifndef VAI_GROUPS_FEED_FORWARD_H
#define VAI_GROUPS_FEED_FORWARD_H

#include "vai/nodes/linear.h"

namespace vai::group {


/////////////////////////////////////////////////////////////////////////////////////////
// FeedForward - two-layer MLP, the activation fused into the first projection
// Input: in0 [..., inDim], optional residual [..., outDim]
// Output: out0 [..., outDim] = fc2(activation(fc1(in0)))
//
// The activation rides fc1's epilogue, so the hidden tensor is written once.
//
// acceptResidual opens "residual" for the block's skip and folds the add into fc2.
// A block that normalizes its branch output has to add outside instead, and
// leaves it false: no slot is opened.
/////////////////////////////////////////////////////////////////////////////////////////
struct FeedForward : public vai::NodeGroup
{
    vai::node::Linear fc1;   // [..., inDim] -> [..., hiddenDim]
    vai::node::Linear fc2;   // [..., hiddenDim] -> [..., outDim]

    struct Config {
        uint32_t inDim;
        uint32_t hiddenDim;
        uint32_t outDim;
        vai::Epilogue activation = vai::Epilogue::None;
        bool acceptResidual = false;   // open "residual" and fold its addend into fc2
    };

    FeedForward(Config config)
        : fc1({.inDim = config.inDim, .outDim = config.hiddenDim, .epilogue = config.activation}),
          fc2({.inDim = config.hiddenDim, .outDim = config.outDim, .acceptResidual = config.acceptResidual})
    {
        fc1 - fc2;

        defineSlot("in0", fc1.slot("in0"));
        if (config.acceptResidual)
            defineSlot("residual", fc2.slot("residual"));
        defineSlot("out0", fc2.slot("out0"));
    }
};


}   // namespace vai::group

#endif // VAI_GROUPS_FEED_FORWARD_H
