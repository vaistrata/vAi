#ifndef EXPERIMENTAL_RTDETRV2_GROUPS_RESNET_VD_H
#define EXPERIMENTAL_RTDETRV2_GROUPS_RESNET_VD_H

#include "vai/nodes/convolution.h"
#include "vai/nodes/pool2d.h"
#include "resnet-vd-block.h"
#include <iterator>
#include <numeric>

namespace model::experimental::rtdetrv2::group {

/////////////////////////////////////////////////////////////////////////////////////////
// ResNetVdConfig - shared channel defaults and checkpoint-specific residual stages
/////////////////////////////////////////////////////////////////////////////////////////
struct ResNetVdConfig
{
    uint32_t inChannels = 3;
    uint32_t stemChannels[3] = { 32, 32, 64 };
    uint32_t stageChannels[4] = { 64, 128, 256, 512 };
    uint32_t stageDepths[4] = { 3, 4, 6, 3 };
    bool bottleneck = false;

    uint32_t outChannels(uint32_t stage) const
    {
        const uint32_t expansion = bottleneck ? 4 : 1;
        return stageChannels[stage] * expansion;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////
// ResNetVd - ResNet-vd stem and four residual stages
// Input:  in0 [3, 640, 640]
// Output: p3 [128 * E, 80, 80], p4 [256 * E, 40, 40], p5 [512 * E, 20, 20]
// E is 1 for R18/R34 and 4 for R50/R101.
/////////////////////////////////////////////////////////////////////////////////////////
struct ResNetVd : public vai::NodeGroup
{
    vai::node::Convolution stem0, stem1, stem2;
    vai::node::MaxPool2d maxPool;
    std::vector<ResNetVdBlock> blocks;

    ResNetVd(ResNetVdConfig config)
    : stem0(
          { .inChannels = config.inChannels,
            .outChannels = config.stemChannels[0],
            .kernelSize = 3,
            .stride = 2,
            .epilogue = vai::Epilogue::Relu })
    , stem1(
          { .inChannels = config.stemChannels[0],
            .outChannels = config.stemChannels[1],
            .kernelSize = 3,
            .stride = 1,
            .epilogue = vai::Epilogue::Relu })
    , stem2(
          { .inChannels = config.stemChannels[1],
            .outChannels = config.stemChannels[2],
            .kernelSize = 3,
            .stride = 1,
            .epilogue = vai::Epilogue::Relu })
    , maxPool({ .kernelSize = 3, .stride = 2, .padding = 1 })
    {
        stem0 - stem1 - stem2 - maxPool;

        blocks.reserve(std::accumulate(std::begin(config.stageDepths), std::end(config.stageDepths), 0u));
        uint32_t inChannels = config.stemChannels[std::size(config.stemChannels) - 1];
        for (uint32_t stage = 0; stage < std::size(config.stageDepths); ++stage)
        {
            const uint32_t baseChannels = config.stageChannels[stage];
            for (uint32_t numBlock = 0; numBlock < config.stageDepths[stage]; ++numBlock)
            {
                blocks.emplace_back(ResNetVdBlock::Config{
                    .inChannels = inChannels,
                    .baseChannels = baseChannels,
                    .stride = (stage && numBlock == 0) ? 2u : 1u,
                    .bottleneck = config.bottleneck,
                    .shortcut = (numBlock == 0) });
                inChannels = baseChannels * (config.bottleneck ? 4 : 1);
            }
            if (stage)
                defineSlot(
                    "p" + std::to_string(stage + 2),
                    blocks.back().slot(
                        "out0"));   // p is feature pyramid ex. p3 = 2^3, input 640 x 640 -> 640 / 8 x 640 / 8 -> 80 x 80
        }

        maxPool - blocks.front();
        for (size_t i = 1; i < blocks.size(); ++i)
            blocks[i - 1] - blocks[i];

        defineSlot("in0", stem0.slot("in0"));
    }
};

}   // namespace model::experimental::rtdetrv2::group

#endif   // EXPERIMENTAL_RTDETRV2_GROUPS_RESNET_VD_H
