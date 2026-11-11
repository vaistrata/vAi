#ifndef VAI_GROUPS_RESNET_H
#define VAI_GROUPS_RESNET_H

#include "vai/nodes/identity.h"
#include "vai/nodes/convolution.h"
#include "vai/nodes/group-norm.h"
#include "vai/nodes/pool2d.h"
#include <array>
#include <optional>
#include <vector>


namespace vai {


// The norm after each conv, as torchvision's norm_layer.
enum class ResNetNorm {
    BatchNorm,   // folded into the conv's scale and bias
    GroupNorm,   // a node of its own
};

struct ResNetConfig
{
    ResNetNorm norm = ResNetNorm::BatchNorm;
    uint32_t groupNormWidth = 16;   // channels per group, GroupNorm only
    float normEps = 1e-5f;
};


}   // namespace vai

namespace vai::group {


/////////////////////////////////////////////////////////////////////////////////////////
// ConvNorm - a conv, its norm, and the activation after them
// Input : in0 [..., Ci, H, W], residual [..., Co, H / stride, W / stride] when accepted
// Output: out0 [..., Co, H / stride, W / stride]
//
// out0 = epilogue(norm(conv(in0)) + residual)
//
// With BatchNorm the conv takes the residual and the epilogue; with GroupNorm
// the norm node does.
/////////////////////////////////////////////////////////////////////////////////////////
struct ConvNorm : public vai::NodeGroup
{
    vai::node::Convolution conv;
    std::optional<vai::node::GroupNorm> norm;

    ConvNorm(const ResNetConfig& config, uint32_t inChannels, uint32_t outChannels, uint32_t kernelSize,
             uint32_t stride, vai::Epilogue epilogue, bool acceptResidual = false)
        : conv(config.norm == ResNetNorm::BatchNorm
                   ? vai::node::Convolution::Config{.inChannels = inChannels,
                                                    .outChannels = outChannels,
                                                    .kernelSize = kernelSize,
                                                    .stride = stride,
                                                    .acceptResidual = acceptResidual,
                                                    .residualOrder = vai::ResidualOrder::BeforeEpilogue,
                                                    .epilogue = epilogue}
                   : vai::node::Convolution::Config{.inChannels = inChannels,
                                                    .outChannels = outChannels,
                                                    .kernelSize = kernelSize,
                                                    .stride = stride})
    {
        if (config.norm == ResNetNorm::GroupNorm)
        {
            norm.emplace(vai::node::GroupNorm::Config{.groups = outChannels / config.groupNormWidth,
                                                      .eps = config.normEps,
                                                      .acceptResidual = acceptResidual,
                                                      .residualOrder = vai::ResidualOrder::BeforeEpilogue,
                                                      .epilogue = epilogue});
        }

        conv - norm;

        vai::Node& last = norm ? static_cast<vai::Node&>(*norm) : conv;
        defineSlot("in0", conv.slot("in0"));
        if (acceptResidual)
            defineSlot("residual", last.slot("residual"));
        defineSlot("out0", last.slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// BasicBlock - torchvision ResNet basic block
// Input : in0  [..., Ci, H, W]
// Output: out0 [..., planes, H / stride, W / stride]
//
// out0 = relu(norm2(conv2(relu(norm1(conv1(in0))))) + shortcut(in0))
/////////////////////////////////////////////////////////////////////////////////////////
struct BasicBlock : public vai::NodeGroup
{
    static constexpr uint32_t expansion = 1;

    vai::node::Identity input;
    ConvNorm conv1;
    ConvNorm conv2;
    std::optional<ConvNorm> shortcut;   // empty: in0 itself

    BasicBlock(const ResNetConfig& config, uint32_t inChannels, uint32_t planes, uint32_t stride)
        : conv1(config, inChannels, planes, 3, stride, vai::Epilogue::Relu),
          conv2(config, planes, planes, 3, 1, vai::Epilogue::Relu, true)
    {
        if (stride != 1u || inChannels != planes)
            shortcut.emplace(config, inChannels, planes, 1, stride, vai::Epilogue::None);

        input - conv1 - conv2;
        input - shortcut - "residual" / conv2;

        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", conv2.slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// Bottleneck - torchvision ResNet bottleneck block
// Input : in0  [..., Ci, H, W]
// Output: out0 [..., planes * 4, H / stride, W / stride]
//
// out0 = relu(norm3(conv3(h)) + shortcut(in0)),
//    h = relu(norm2(conv2(relu(norm1(conv1(in0))))))
//
// The stride sits on the 3x3, conv2, as in torchvision.
/////////////////////////////////////////////////////////////////////////////////////////
struct Bottleneck : public vai::NodeGroup
{
    static constexpr uint32_t expansion = 4;

    vai::node::Identity input;
    ConvNorm conv1;
    ConvNorm conv2;
    ConvNorm conv3;
    std::optional<ConvNorm> shortcut;   // empty: in0 itself

    Bottleneck(const ResNetConfig& config, uint32_t inChannels, uint32_t planes, uint32_t stride)
        : conv1(config, inChannels, planes, 1, 1, vai::Epilogue::Relu),
          conv2(config, planes, planes, 3, stride, vai::Epilogue::Relu),
          conv3(config, planes, planes * expansion, 1, 1, vai::Epilogue::Relu, true)
    {
        if (stride != 1u || inChannels != planes * expansion)
            shortcut.emplace(config, inChannels, planes * expansion, 1, stride, vai::Epilogue::None);

        input - conv1 - conv2 - conv3;
        input - shortcut - "residual" / conv3;

        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", conv3.slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// ResNet - torchvision ResNet without its classifier head: the stem and layer1 .. layer4
// Input : in0  [..., 3, H, W], the axes ahead of 3 a batch of images
// Output: out0 [..., 512 * expansion, H / 32, W / 32]
/////////////////////////////////////////////////////////////////////////////////////////
template <typename Block, uint32_t... blocksPerLayer>
struct ResNet : public vai::NodeGroup
{
    static_assert(sizeof...(blocksPerLayer) == 4, "layer1 .. layer4");
    static constexpr uint32_t outChannels = 512 * Block::expansion;
    static constexpr std::array<uint32_t, 4> depths{blocksPerLayer...};

    ConvNorm conv1;
    vai::node::MaxPool2d maxPool;
    std::vector<Block> blocks;   // layer1 .. layer4, in order

    explicit ResNet(const ResNetConfig& config = {})
        : conv1(config, 3, 64, 7, 2, vai::Epilogue::Relu), maxPool({.kernelSize = 3, .stride = 2, .padding = 1})
    {
        // Built in place: a block's slots point back at it, so it cannot move.
        blocks.reserve(depths[0] + depths[1] + depths[2] + depths[3]);
        uint32_t inChannels = 64;
        for (uint32_t l = 0; l < 4; ++l)
        {
            const uint32_t planes = 64 << l;
            for (uint32_t b = 0; b < depths[l]; ++b)
            {
                blocks.emplace_back(config, inChannels, planes, (l != 0 && b == 0) ? 2 : 1);
                inChannels = planes * Block::expansion;
            }
        }

        conv1 - maxPool - blocks.front();
        for (size_t i = 1; i < blocks.size(); ++i)
            blocks[i - 1] - blocks[i];

        defineSlot("in0", conv1.slot("in0"));
        defineSlot("out0", blocks.back().slot("out0"));
    }
};


// torchvision's depths: the blocks in layer1 .. layer4.
using ResNet18 = ResNet<BasicBlock, 2, 2, 2, 2>;
using ResNet34 = ResNet<BasicBlock, 3, 4, 6, 3>;
using ResNet50 = ResNet<Bottleneck, 3, 4, 6, 3>;
using ResNet101 = ResNet<Bottleneck, 3, 4, 23, 3>;
using ResNet152 = ResNet<Bottleneck, 3, 8, 36, 3>;


}   // namespace vai::group

#endif   // VAI_GROUPS_RESNET_H
