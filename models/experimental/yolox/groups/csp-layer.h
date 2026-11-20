#ifndef EXPERIMENTAL_YOLOX_GROUPS_CSP_LAYER_H
#define EXPERIMENTAL_YOLOX_GROUPS_CSP_LAYER_H

#include "vai/nodes/convolution.h"
#include "vai/nodes/identity.h"
#include "vai/nodes/slice.h"
#include "vai/nodes/concat.h"
#include <vector>


namespace model::experimental::yolox::group {


/////////////////////////////////////////////////////////////////////////////////////////
// Bottleneck - YOLOX bottleneck with an optional residual connection
// Input : in0  [B, C, H, W]
// Output: out0 [B, C, H, W]
/////////////////////////////////////////////////////////////////////////////////////////
struct Bottleneck : public vai::NodeGroup
{
    vai::node::Identity input;
    vai::node::Convolution conv1;
    vai::node::Convolution conv2;

    Bottleneck(uint32_t channels, bool shortcut = true)
    : conv1(
          { .inChannels = channels,
            .outChannels = channels,
            .kernelSize = 1u,
            .stride = 1u,
            .epilogue = vai::Epilogue::Silu })
    , conv2(
          { .inChannels = channels,
            .outChannels = channels,
            .kernelSize = 3u,
            .stride = 1u,
            .acceptResidual = shortcut,
            .epilogue = vai::Epilogue::Silu })
    {
        // clang-format off
        input - conv1 - conv2;
        // clang-format on

        if (conv2.config_.acceptResidual)
            // clang-format off
            input - "residual" / conv2;
        // clang-format on
        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", conv2.slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// CSPLayer - Two-branch CSP layer with a Bottleneck chain on the first branch
// Input : in0  [B, Ci, H, W]
// Output: out0 [B, Co, H, W]
/////////////////////////////////////////////////////////////////////////////////////////
struct CSPLayer : public vai::NodeGroup
{
    vai::node::Identity input;
    vai::node::Convolution conv12;
    vai::node::Slice conv1;
    vai::node::Slice conv2;
    std::vector<Bottleneck> bottlenecks;
    vai::node::Concat concat{ { .axis = 1 } };
    vai::node::Convolution conv3;

    CSPLayer(uint32_t inChannels, uint32_t outChannels, uint32_t repeats, bool shortcut = true, float expansion = 0.5f)
    : conv12(
          { .inChannels = inChannels,
            .outChannels = 2u * static_cast<uint32_t>(outChannels * expansion),
            .kernelSize = 1u,
            .stride = 1u,
            .epilogue = vai::Epilogue::Silu })
    , conv1({ .axis = 1, .start = 0, .count = static_cast<uint32_t>(outChannels * expansion) })
    , conv2({ .axis = 1, .start = conv1.config_.count, .count = conv1.config_.count })
    , conv3(
          { .inChannels = 2u * static_cast<uint32_t>(outChannels * expansion),
            .outChannels = outChannels,
            .kernelSize = 1u,
            .stride = 1u,
            .epilogue = vai::Epilogue::Silu })
    {
        VAI_ASSERT(repeats >= 1u);
        const uint32_t hiddenChannels = static_cast<uint32_t>(outChannels * expansion);
        bottlenecks.reserve(repeats);
        for (uint32_t i = 0; i < repeats; ++i)
            bottlenecks.emplace_back(hiddenChannels, shortcut);


        // Both branches share the input; conv12 packs their output channels.
        // clang-format off
        input - conv12 - conv1 - bottlenecks[0];
        // clang-format on
        for (uint32_t i = 1; i < repeats; ++i)
            // clang-format off
            bottlenecks[i - 1u] - bottlenecks[i];
        bottlenecks.back() - "in0" / concat;
        conv12 - conv2 - "in1" / concat;
        concat - conv3;
        // clang-format on

        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", conv3.slot("out0"));
    }
};


}   // namespace model::experimental::yolox::group

#endif   // EXPERIMENTAL_YOLOX_GROUPS_CSP_LAYER_H
