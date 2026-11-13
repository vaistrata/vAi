# Quick Start

Build vAi and run the MNIST classifier example.

## Installation

### Requirements

| | |
|---|---|
| GPU | Vulkan conformant (NVIDIA / AMD / Intel) |
| Vulkan SDK | 1.3+, including glslc|
| CMake | 3.16+  |
| Compiler | C++23 |

### Clone

eva(Vulkan wrapper) is included as a submodule and needs to be cloned as well.

```bash
git clone --recursive https://github.com/vaistrata/vAi.git
cd vAi
```

If you have already cloned the vAi repository:

```bash
git submodule update --init --recursive
```

### Build

```bash
cmake -B build
cmake --build build
```

Build options:

| Option | Default | Description |
|---|---|---|
| `VAI_BUILD_MODEL_DEMO` | ON | Each model's demo (`demo-gpt2`, `demo-yolox`, etc.) |
| `VAI_BUILD_EXAMPLES` | OFF | Examples outside the models (`demo-mnist`, `chat-llama3`, `chat-gemma3`) |
| `VAI_BUILD_TESTS` | OFF | Kernel-level tests, A/B |
| `VAI_BUILD_BENCH` | OFF | Benchmarks against llama.cpp, ncnn, and other frameworks |

## Simple Example

MNIST classfier—from model definition and weight loading to execution [`examples/mnist/main.cpp`](../../examples/mnist/main.cpp)—is contained in a single file. Below, we walk through this file from top to bottom. The checkpoint is in [`data/ckpts/mnist/jroot/`](../../data/ckpts/mnist/jroot/) and the input images in [`data/images/`](../../data/images/).

### Connecting Nodes

We first define the repeated unit.

```cpp
struct ConvBlock : public vai::NodeGroup {
    vai::node::Convolution conv;
    vai::node::MaxPool2d maxpool;

    ConvBlock(uint32_t inChannels, uint32_t outChannels, uint32_t kernelSize)
    : conv({
        .inChannels = inChannels,
        .outChannels = outChannels,
        .kernelSize = kernelSize,
        .stride = 1,
        .epilogue = vai::Epilogue::Relu
    })
    , maxpool({
        .kernelSize = 2,
        .stride = 2
    })
    {
        conv - maxpool;
        defineSlot("in0", conv.slot("in0"));
        defineSlot("out0", maxpool.slot("out0"));
    }
};
```




Node configuration is passed through the 'options' structure. By specifying ReLU as the epilogue, the operation is completed inside the conv kernel without adding a separate activation node.


ConvBlock's in0 and out0 are not newly created slots. they are aliases for conv's in0 and maxpool's out0, respectively. When this group is connected from the outside, it actually connects to the slots of the internal nodes.

### Repeating Blocks

When stacking the same block multiple times, store them in a vector and chain them with a loop, instead of listing members individually. NodeGroups can also be connected with -, just like nodes.

```cpp
struct ConvSequence : public vai::NodeGroup {
    std::vector<ConvBlock> blocks;

    ConvSequence(std::vector<uint32_t> channels, uint32_t kernelSize)
    {
        const uint32_t nBlocks = channels.size() - 1;

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
```

`channels` specifies the number of channels, so `{1, 32, 64}` creates two blocks: 1→32 and 32→64.

`reserve` and `emplace_back` prevent value reallocation (see [Precautions when using containers in NodeGroup](03-usage/node-group.md#precautions-when-using-containers)).

### Model

Inherit from `vai::NeuralNet`, creates the graph's input and output, using `input()` and `output()`.


```cpp
struct Net : public vai::NeuralNet {
    ConvSequence convX2;
    vai::node::Flatten flatten;
    vai::node::Linear linear;

    Net(const vai::NeuralNetConfig& netCfg)
    : NeuralNet(netCfg)
    , convX2({1, 32, 64}, 3)
    , flatten()
    , linear({7 * 7 * 64, 10})
    {
        input("in0") - convX2 - flatten - linear - output("out0");
    }
};
```

Connecting all members with `-` creates the MNIST model. 28x28 input passes through two blocks and is reduced to 7x7, then flattened to produce 10 logits.

### Weight Loading

Weight loading is implemented by specializing `loadWeights<deployment_name>(model)`. The deployment name is the name of the contributor who trained this MNIST model.

```cpp
template<>
void loadWeights<"jroooot", mnist::Net>(mnist::Net& net)
{
    std::string path = VAI_DATA_DIR "/ckpts/mnist/jroot/model.safetensors";
    auto parser = vai::SafetensorsParser(path.c_str());

    // Weights load at the net's wDefaultDtype.
    const vai::Dtype wDt = net.config().wDefaultDtype;
    auto load = [&](auto& node, const std::string& wKey, const std::string& bKey)
    {
        node["weight"] = vai::Tensor(parser[wKey], wDt);
        node["bias"]   = vai::Tensor(parser[bKey], wDt);
    };

    load(net.convX2.blocks[0].conv, "conv1.weight", "conv1.bias");
    load(net.convX2.blocks[1].conv, "conv2.weight", "conv2.bias");
    load(net.linear, "fc.weight", "fc.bias");
}
```

`VAI_DATA_DIR` is the repository's data/ path embedded by the build system. This function is the only place that knows the checkpoint key names and disk paths, and the model definition knows neither. For the provided models, these specializations are placed in a separate file named `models/<impl>/<model>/<deployment_name>-weights.h` (see [Weight loading in Provided Models](04-models/README.md#weight-loading) for details on this structure).

### Execution

```cpp
int main()
{
    // Load the 28x28 gray image and normalize it to [0, 1]
    std::vector<float> inputData = vai::utils::ImageProcessor(VAI_DATA_DIR "/images/0.png", 1)
        .normalize(0.0f, 1.0f)
        .chw();

    // Create the model. defaultDtype is the default precision for parameters.
    mnist::Net net({
        .defaultDtype = Dtype::Fp32,
    });

    // Load weights from the check point ("jroooot"is the contributor who trained it)
    loadWeights<"jroooot">(net);

    Tensor input = Tensor({1, 28, 28}, Dtype::Fp32).setFloats(inputData);
    Tensor output;

    // Run the model and get the logits
    net({{"in0", input}}, {{"out0", &output}});

    // The network output is fp32.
    const float* data = output.getData();

    // Print logits for each class
    printf("\n=== Output logits ===\n");
    for (int i = 0; i < 10; ++i)
        printf("data[%d] = %f\n", i, data[i]);

    return 0;
}
```

model(...) passes tensors by name. The names defined earlier with input("in0") and output("out0") are used as the keys (see [Execution Pipeline](03-usage/execution-pipeline.md) for what happens during a single call).

The output looks as below. Since the input is `0.png`, the logit for class 0 is overwhelmingly high.


```
=== Output logits ===
data[0] = 23.026661
data[1] = -24.376106
data[2] = -1.872803
data[3] = -15.056368
data[4] = -14.481293
data[5] = -13.030799
data[6] = -0.130337
data[7] = -14.873937
data[8] = 1.507278
data[9] = -2.433280
```
