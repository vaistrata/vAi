#ifndef EXPERIMENTAL_YOLOX_GROUPS_CSP_LAYER_H
#define EXPERIMENTAL_YOLOX_GROUPS_CSP_LAYER_H

#include "bottleneck.h"
#include "vai/nodes/concat.h"
#include <vector>


namespace model::experimental::yolox::group {


/////////////////////////////////////////////////////////////////////////////////////////
// CSPLayer - Two-branch CSP layer with a Bottleneck chain on the first branch
// Input : in0  [Ci, H, W]
// Output: out0 [Co, H, W]
/////////////////////////////////////////////////////////////////////////////////////////
struct CSPLayer : public vai::NodeGroup
{
    vai::node::Identity input;
    vai::node::Convolution conv1;
    vai::node::Convolution conv2;
    std::vector<Bottleneck> bottlenecks;
    vai::node::Concat concat;
    vai::node::Convolution conv3;

    CSPLayer(uint32_t inChannels, uint32_t outChannels, uint32_t repeats, bool shortcut = true, float expansion = 0.5f)
        : conv1(baseConv(inChannels, static_cast<uint32_t>(outChannels * expansion), 1u, 1u)),
          conv2(baseConv(inChannels, static_cast<uint32_t>(outChannels * expansion), 1u, 1u)),
          conv3(baseConv(2u * static_cast<uint32_t>(outChannels * expansion), outChannels, 1u, 1u))
    {
        VAI_ASSERT(repeats >= 1u);
        const uint32_t hiddenChannels = static_cast<uint32_t>(outChannels * expansion);
        bottlenecks.reserve(repeats);
        for (uint32_t i = 0; i < repeats; ++i)
            bottlenecks.emplace_back(hiddenChannels, hiddenChannels, shortcut, 1.0f);

        input - conv1 - bottlenecks[0];
        for (uint32_t i = 1; i < repeats; ++i)
            bottlenecks[i - 1u] - bottlenecks[i];
        bottlenecks.back() - "in0" / concat;
        input - conv2 - "in1" / concat;
        concat - conv3;

        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", conv3.slot("out0"));
    }
};


}   // namespace model::experimental::yolox::group

#endif   // EXPERIMENTAL_YOLOX_GROUPS_CSP_LAYER_H
