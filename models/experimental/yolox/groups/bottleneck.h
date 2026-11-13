#ifndef EXPERIMENTAL_YOLOX_GROUPS_BOTTLENECK_H
#define EXPERIMENTAL_YOLOX_GROUPS_BOTTLENECK_H

#include "common.h"
#include "vai/nodes/identity.h"


namespace model::experimental::yolox::group {


/////////////////////////////////////////////////////////////////////////////////////////
// Bottleneck - YOLOX bottleneck with an optional residual connection
// Input : in0  [Ci, H, W]
// Output: out0 [Co, H, W]
/////////////////////////////////////////////////////////////////////////////////////////
struct Bottleneck : public vai::NodeGroup
{
    vai::node::Identity input;
    vai::node::Convolution conv1;
    vai::node::Convolution conv2;   // adds the shortcut as its "residual"

    Bottleneck(uint32_t inChannels, uint32_t outChannels, bool shortcut = true, float expansion = 0.5f)
    : conv1(baseConv(inChannels, static_cast<uint32_t>(outChannels * expansion), 1u, 1u))
    , conv2(baseConv(static_cast<uint32_t>(outChannels * expansion), outChannels, 3u, 1u,
                     shortcut && inChannels == outChannels))
    {
        // clang-format off
        input - conv1 - conv2;
        if (conv2.config_.acceptResidual)
            input - "residual"/conv2;
        // clang-format on
        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", conv2.slot("out0"));
    }
};


}   // namespace model::experimental::yolox::group

#endif   // EXPERIMENTAL_YOLOX_GROUPS_BOTTLENECK_H
