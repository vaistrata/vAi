#ifndef EXPERIMENTAL_RTDETRV2_GROUPS_CSP_LAYER_H
#define EXPERIMENTAL_RTDETRV2_GROUPS_CSP_LAYER_H

#include "vai/nodes/convolution.h"
#include "vai/nodes/identity.h"

namespace model::experimental::rtdetrv2::group {

/////////////////////////////////////////////////////////////////////////////////////////
// CSPLayerConfig - output channels and branch expansion
/////////////////////////////////////////////////////////////////////////////////////////
struct CSPLayerConfig
{
    uint32_t outChannels;
    float expansion;

    uint32_t hiddenChannels() const { return uint32_t(outChannels * expansion); }
};

/////////////////////////////////////////////////////////////////////////////////////////
// CSPLayer - Two projected branches with three fused RepVGG convolutions on the main branch
// Input:  in0  [2 * D, H, W]
// Output: out0 [D, H, W]
// Branches use D * expansion channels; an optional projection restores D.
/////////////////////////////////////////////////////////////////////////////////////////
struct CSPLayer : public vai::NodeGroup
{
    vai::node::Identity input;
    vai::node::Convolution conv1, conv2;
    std::optional<vai::node::Convolution> optionConv;
    // Each RepVGG becomes one 3x3 Conv + bias + SiLU; the loader folds both Conv/BN branches into it.
    vai::node::Convolution repVgg0, repVgg1, repVgg2;

    CSPLayer(CSPLayerConfig config)
    : conv1(
          { .inChannels = 2 * config.outChannels,
            .outChannels = config.hiddenChannels(),
            .kernelSize = 1,
            .stride = 1,
            .epilogue = vai::Epilogue::Silu })
    , conv2(
          { .inChannels = 2 * config.outChannels,
            .outChannels = config.hiddenChannels(),
            .kernelSize = 1,
            .stride = 1,
            .epilogue = vai::Epilogue::Silu })
    , repVgg0(
          { .inChannels = config.hiddenChannels(),
            .outChannels = config.hiddenChannels(),
            .kernelSize = 3,
            .stride = 1,
            .epilogue = vai::Epilogue::Silu })
    , repVgg1(
          { .inChannels = config.hiddenChannels(),
            .outChannels = config.hiddenChannels(),
            .kernelSize = 3,
            .stride = 1,
            .epilogue = vai::Epilogue::Silu })
    , repVgg2(
          { .inChannels = config.hiddenChannels(),
            .outChannels = config.hiddenChannels(),
            .kernelSize = 3,
            .stride = 1,
            .acceptResidual = true,
            .epilogue = vai::Epilogue::Silu })
    {
        input - conv1 - repVgg0 - repVgg1 - repVgg2;
        input - conv2 - "residual" / repVgg2;

        if (config.expansion != 1.0f)
        {
            optionConv.emplace(vai::node::Convolution::Config{
                .inChannels = config.hiddenChannels(),
                .outChannels = config.outChannels,
                .kernelSize = 1,
                .stride = 1,
                .epilogue = vai::Epilogue::Silu });
        }

        repVgg2 - optionConv;

        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", optionConv ? optionConv->slot("out0") : repVgg2.slot("out0"));
    }
};

}   // namespace model::experimental::rtdetrv2::group

#endif   // EXPERIMENTAL_RTDETRV2_GROUPS_CSP_LAYER_H
