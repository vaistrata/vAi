#ifndef PYVAI_MODELS_MNIST_NET_H
#define PYVAI_MODELS_MNIST_NET_H

// Verbatim copy of the model and its weight loader from examples/mnist/main.cpp, which
// keeps them in one file with main(), so nothing there can be included. The loader only
// takes the Checkpoint instead of composing its path. Re-copy when the source changes.

#include "vai/nodes/convolution.h"
#include "vai/nodes/pool2d.h"
#include "vai/nodes/flatten.h"
#include "vai/nodes/linear.h"
#include "vai/engine/weights.h"
#include <cstdint>
#include <string>
#include <vector>

namespace mnist {

/////////////////////////////////////////////////////////////////////////////////////////
// ConvBlock - Conv2D (ReLU) followed by 2x2 max pooling
// Input : in0  [C, H, W]
// Output: out0 [F, H/2, W/2]
/////////////////////////////////////////////////////////////////////////////////////////
struct ConvBlock : public vai::NodeGroup
{
    vai::node::Convolution conv;
    vai::node::MaxPool2d maxpool;

    ConvBlock(uint32_t inChannels, uint32_t outChannels, uint32_t kernelSize)
        : conv({.inChannels = inChannels,
                .outChannels = outChannels,
                .kernelSize = kernelSize,
                .stride = 1,
                .epilogue = vai::Epilogue::Relu}),
          maxpool({.kernelSize = 2, .stride = 2})
    {
        conv - maxpool;
        defineSlot("in0", conv.slot("in0"));
        defineSlot("out0", maxpool.slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// ConvSequence - chain of ConvBlocks (channels[i] -> channels[i + 1])
// Input : in0  [channels.front(), H, W]
// Output: out0 [channels.back(), H / 2^n, W / 2^n]
/////////////////////////////////////////////////////////////////////////////////////////
struct ConvSequence : public vai::NodeGroup
{
    std::vector<ConvBlock> blocks;

    ConvSequence(std::vector<uint32_t> channels, uint32_t kernelSize)
    {
        const uint32_t nBlocks = channels.size() - 1;
        VAI_ASSERT(nBlocks >= 1);

        blocks.reserve(nBlocks);
        for (uint32_t i = 0; i < nBlocks; ++i)
        {
            blocks.emplace_back(channels[i], channels[i + 1], kernelSize);
            if (i > 0)
                blocks[i - 1] - blocks[i];
        }

        defineSlot("in0", blocks[0].slot("in0"));
        defineSlot("out0", blocks[nBlocks - 1].slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// Net - MNIST classifier: 2x ConvBlock -> flatten -> Linear(10)
// Input : [1, 28, 28]
// Output: [10] class logits
/////////////////////////////////////////////////////////////////////////////////////////
struct Net : public vai::NeuralNet
{
    ConvSequence convX2;
    vai::node::Flatten flatten;
    vai::node::Linear linear;

    Net(const vai::NeuralNetConfig& netCfg)
        : NeuralNet(netCfg), convX2({1, 32, 64}, 3), flatten(), linear({.inDim = 7 * 7 * 64, .outDim = 10})
    {
        input("in0") - convX2 - flatten - linear - output("out0");
    }
};

}   // namespace mnist

namespace mnist::jroot {

// This checkpoint has no upstream distributor, so the tag names the contributor
// who trained it.
inline void loadWeights(Net& net, const vai::Checkpoint& ckpt)
{
    // Weights load at the net's wDefaultDtype. Convolution and Linear share
    // the same weight/bias interface.
    const vai::Dtype wDt = net.config().wDefaultDtype;
    auto load = [&](auto& node, const std::string& wKey, const std::string& bKey) {
        node["weight"] = vai::Tensor(ckpt[wKey], wDt);
        node["bias"] = vai::Tensor(ckpt[bKey], wDt);
    };

    // Conv blocks
    load(net.convX2.blocks[0].conv, "conv1.weight", "conv1.bias");   // (C1, C0, K, K)
    load(net.convX2.blocks[1].conv, "conv2.weight", "conv2.bias");   // (C2, C1, K, K)

    // Fully-connected classification head
    load(net.linear, "fc.weight", "fc.bias");   // (10, C2 * H/4 * W/4)
}

}   // namespace mnist::jroot

#endif   // PYVAI_MODELS_MNIST_NET_H
