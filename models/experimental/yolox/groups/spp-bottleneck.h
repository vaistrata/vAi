#ifndef EXPERIMENTAL_YOLOX_GROUPS_SPP_BOTTLENECK_H
#define EXPERIMENTAL_YOLOX_GROUPS_SPP_BOTTLENECK_H

#include "vai/nodes/convolution.h"
#include "vai/nodes/pool2d.h"
#include "vai/nodes/concat.h"


namespace model::experimental::yolox::group {


/////////////////////////////////////////////////////////////////////////////////////////
// SPPBottleneck - Parallel 5x5, 9x9 and 13x13 max pooling between 1x1 convolutions
// Input : in0  [B, Ci, H, W]
// Output: out0 [B, Co, H, W]
/////////////////////////////////////////////////////////////////////////////////////////
struct SPPBottleneck : public vai::NodeGroup
{
    vai::node::Convolution conv1, conv2;
    vai::node::MaxPool2d pool5, pool9, pool13;
    vai::node::Concat concat;

    SPPBottleneck(uint32_t inChannels, uint32_t outChannels)
    : conv1(
          { .inChannels = inChannels,
            .outChannels = inChannels / 2u,
            .kernelSize = 1u,
            .stride = 1u,
            .actType = vai::ActType::Silu })
    , pool5({ .kernelSize = 5u, .stride = 1u, .padding = 2u })
    , pool9({ .kernelSize = 9u, .stride = 1u, .padding = 4u })
    , pool13({ .kernelSize = 13u, .stride = 1u, .padding = 6u })
    , concat({ .axis = 1, .inputCount = 4u })
    , conv2(
          { .inChannels = 2u * inChannels,
            .outChannels = outChannels,
            .kernelSize = 1u,
            .stride = 1u,
            .actType = vai::ActType::Silu })
    {
        // clang-format off
        conv1 - "in0" / concat;
        conv1 - pool5 - "in1" / concat;
        conv1 - pool9 - "in2" / concat;
        conv1 - pool13 - "in3" / concat;
        // clang-format on

        // clang-format off
        concat - conv2;
        // clang-format on

        defineSlot("in0", conv1.slot("in0"));
        defineSlot("out0", conv2.slot("out0"));
    }
};


}   // namespace model::experimental::yolox::group

#endif   // EXPERIMENTAL_YOLOX_GROUPS_SPP_BOTTLENECK_H
