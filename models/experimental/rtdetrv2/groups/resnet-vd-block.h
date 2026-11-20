#ifndef EXPERIMENTAL_RTDETRV2_GROUPS_RESNET_VD_BLOCK_H
#define EXPERIMENTAL_RTDETRV2_GROUPS_RESNET_VD_BLOCK_H

#include "vai/nodes/convolution.h"
#include "vai/nodes/identity.h"
#include "vai/nodes/pool2d.h"

namespace model::experimental::rtdetrv2::group {

/////////////////////////////////////////////////////////////////////////////////////////
// ResidualBlock - ResNet-vd basic or bottleneck block with a ReLU residual sum
// Input:  in0  [Ci, H, W]
// Output: out0 [Co, H / stride, W / stride] (even spatial sizes when stride is 2)
// Co is baseChannels for a basic block and 4 * baseChannels for a bottleneck block.
// The first block of each stage projects the shortcut, using average pooling
// before the projection when downsampling.
/////////////////////////////////////////////////////////////////////////////////////////
struct ResidualBlock : public vai::NodeGroup
{
    vai::node::Identity input;
    std::vector<vai::node::Convolution> main;
    std::optional<vai::node::AvgPool2d> avgPool;
    std::optional<vai::node::Convolution> shortcut;

    struct Config
    {
        uint32_t inChannels;
        uint32_t baseChannels;
        uint32_t stride;
        bool bottleneck;
        bool shortcut;
    };

    ResidualBlock(Config config)
    {
        const uint32_t outChannels = config.baseChannels * (config.bottleneck ? 4 : 1);
        if (config.bottleneck)
        {
            // Bottleneck: 1x1 -> 3x3 -> 1x1; downsample in the 3x3 convolution.
            main.reserve(3);
            main.emplace_back(vai::node::Convolution::Config{
                .inChannels = config.inChannels,
                .outChannels = config.baseChannels,
                .kernelSize = 1,
                .stride = 1,
                .epilogue = vai::Epilogue::Relu });
            main.emplace_back(vai::node::Convolution::Config{
                .inChannels = config.baseChannels,
                .outChannels = config.baseChannels,
                .kernelSize = 3,
                .stride = config.stride,
                .epilogue = vai::Epilogue::Relu });
            main.emplace_back(vai::node::Convolution::Config{
                .inChannels = config.baseChannels,
                .outChannels = outChannels,
                .kernelSize = 1,
                .stride = 1,
                .acceptResidual = true,
                .residualOrder = vai::ResidualOrder::BeforeEpilogue,
                .epilogue = vai::Epilogue::Relu });
        }
        else
        {
            // Basic: 3x3 -> 3x3; downsample in the first convolution.
            main.reserve(2);
            main.emplace_back(vai::node::Convolution::Config{
                .inChannels = config.inChannels,
                .outChannels = config.baseChannels,
                .kernelSize = 3,
                .stride = config.stride,
                .epilogue = vai::Epilogue::Relu });
            main.emplace_back(vai::node::Convolution::Config{
                .inChannels = config.baseChannels,
                .outChannels = outChannels,
                .kernelSize = 3,
                .stride = 1,
                .acceptResidual = true,
                .residualOrder = vai::ResidualOrder::BeforeEpilogue,
                .epilogue = vai::Epilogue::Relu });
        }

        input - main.front();
        for (size_t numBlock = 1; numBlock < main.size(); ++numBlock)
            main[numBlock - 1] - main[numBlock];

        if (config.shortcut)
        {
            shortcut.emplace(vai::node::Convolution::Config{
                .inChannels = config.inChannels, .outChannels = outChannels, .kernelSize = 1, .stride = 1 });
            if (config.stride == 2)
                avgPool.emplace(vai::node::AvgPool2d::Config{
                    .kernelSize = 2,
                    .stride = 2,
                    .padding = 0,
                    .ceilMode = true,
                    .countIncludePad = false,
                    .channelsLast = false });
        }

        input - avgPool - shortcut - "residual" / main.back();

        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", main.back().slot("out0"));
    }
};

}   // namespace model::experimental::rtdetrv2::group

#endif   // EXPERIMENTAL_RTDETRV2_GROUPS_RESNET_VD_BLOCK_H
