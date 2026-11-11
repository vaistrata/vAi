#ifndef EXPERIMENTAL_YOLOX_GROUPS_SPP_BOTTLENECK_H
#define EXPERIMENTAL_YOLOX_GROUPS_SPP_BOTTLENECK_H

#include "common.h"
#include "../nodes/sppf.h"


namespace model::experimental::yolox::group {


/////////////////////////////////////////////////////////////////////////////////////////
// SPPBottleneck - Fused SPPF equivalent of parallel 5x5, 9x9 and 13x13 pooling
// Input : in0  [Ci, H, W]
// Output: out0 [Co, H, W]
/////////////////////////////////////////////////////////////////////////////////////////
struct SPPBottleneck : public vai::NodeGroup
{
    vai::node::Convolution conv1;
    node::SPPF sppf;
    vai::node::Convolution conv2;

    SPPBottleneck(uint32_t inChannels, uint32_t outChannels)
        : conv1(baseConv(inChannels, inChannels / 2u, 1u, 1u)), conv2(baseConv(2u * inChannels, outChannels, 1u, 1u))
    {
        conv1 - sppf - conv2;

        defineSlot("in0", conv1.slot("in0"));
        defineSlot("out0", conv2.slot("out0"));
    }
};


}   // namespace model::experimental::yolox::group

#endif   // EXPERIMENTAL_YOLOX_GROUPS_SPP_BOTTLENECK_H
