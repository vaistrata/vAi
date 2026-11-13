# Overview

vAi is a **Vulkan inference engine built on a graph DSL**. It assembles model graphs using a C++ DSL (a dedicated syntax for connecting operator nodes) and executes them using Vulkan Compute Shaders.

## 1. Features

### 1.1 Portability

A single codebase can run and be tested directly on multiple GPUs. Because vAi uses only Vulkan, the industry-standard GPU API, there is no need to rewrite code for vendor-specific SDKs from NVIDIA, AMD, Intel, Qualcomm, and others.

### 1.2 Lightweight

When deploying an LLM with vAi, the runtime binary is just over 2 MB. The dependency chain is designed for on-device environments. Selective builds include only the model code and operator kernels actually used, without additional configuration.

### 1.3 High performance

GPU command buffers are reused when conditions such as input shapes and buffer bindings remain the same. This requires very little CPU work during inference. Conventional inference engines typically traverse the computation graph and repeatedly submit kernel execution commands to the GPU on every inference, but vAi creates the command buffer once and reuses it. GPU kernel tuning also selects different kernels for different devices, so portability does not have to come at the cost of performance.

### 1.4 Intuitiveness

The graph DSL exposes the model structure directly in code. With vAi's dedicated graph syntax and object-oriented design, a model can be built by connecting `Node` and `NodeGroup` objects like LEGO bricks.

```cpp
template<TransformerType layout, QkvLayout qkv>
struct TransformerLayer : public vai::NodeGroup {
    vai::node::Identity input;
    typename BranchNorm<layout>::Type norm1;
    MultiHeadAttention<qkv> mhAttn;
    std::optional<vai::node::Slice> lastRow;
    std::optional<vai::node::Gate> gate;
    typename BranchNorm<layout>::Type norm2;
    typename BranchFfn<layout>::Type ffn;

    TransformerLayer(...)
    {
        input - norm1 -            mhAttn - lastRow - gate/"then" - norm2 -            ffn;
        input         - "residual"/mhAttn - lastRow - gate/"then"         - "residual"/ffn;
    }
};
```

The member list is the block's structure, and the connections are its data flow. The first line in the constructor draws the main path, and the second draws the residual path, like a design diagram. Because the code is the model's structural diagram, it is easy to read and modify.

### 1.5 Extensibility

New operators or layers with custom optimizations can be added directly. Inherit from `Node` and write the required kernel **without modifying a single line of core code**. Newly created kernels can be shared with anyone who needs them without waiting for a merge into the upstream project. There is no need to maintain a fork or conform to a plugin specification.

## 2. Project structure

```
vai/                        Core engine (vAi library)
  engine/
    neural-net.h
    neural-net.cpp           Graph sorting, buffer preparation, command recording
    graph-flow.h             Connection DSL and NodeGroup
    graph-controller.h       Execution paths based on inputs and conditional branches
    tensor.h
    safetensors-parser.h
    weights.h
    buffer-pool.h
    pipeline-pool.h
    error.h
  nodes/                    Built-in nodes
  groups/                   Built-in groups
  shaders/                  Built-in kernels (.comp)
  utils/

data/                       Model checkpoints, input data, etc.

models/                     Supported models
  <impl>/<model-name>/
    net.h                     Model definition
    <distribution-name>-weights.h         Checkpoint loader
    demo.cpp                  demo-<model-name>

examples/                   Programs beyond one model's demo (e.g., mnist/, chat-llama3/)
benchmarks/
tests/
external/eva/               Vulkan wrapper (submodule)
docs/
```

Start by running MNIST with the [Quickstart](02-quickstart.md), then read [Node](03-usage/node.md) and [NodeGroup](03-usage/node-group.md). The following [NeuralNet](03-usage/neural-net.md) and [Tensor](03-usage/tensor.md) pages cover graph execution and data representation.


| What to change | Documentation and code locations |
|---|---|
| Model composition and connections | [MNIST example](../../examples/mnist/main.cpp), [supported models](../../models/), [Writing custom models](contributing/models.md) |
| Blocks shared by multiple models | [Common groups](../../vai/groups/), [NodeGroup](03-usage/node-group.md), [Writing custom groups](contributing/groups.md) |
| Shape handling and GPU computation for an operation | [Nodes](../../vai/nodes/) and [shaders](../../vai/shaders/), [Writing custom nodes and kernels](contributing/nodes.md) |
| Graph execution and buffer management | [neural-net.cpp](../../vai/engine/neural-net.cpp), [Execution pipeline](03-usage/execution-pipeline.md) |

## 3. Common code conventions

**3.1 CamelCase**: Types start with an uppercase letter, while functions and variables start with a lowercase letter.

```cpp
class Linear;            // Type
void addSlot(...);       // Function
uint32_t numQHeads;       // Variable
```

The rules for each construct are as follows.

| Construct | Rule | Example |
|---|---|---|
| Classes, structs, `enum`, type aliases, `concept` | `PascalCase` | `NeuralNet`, `PipeParams` |
| Functions, member functions | `camelCase` | `loadWeights()`, `setProfileTag()` |
| Variables, parameters | `camelCase` | `outShape`, `targetDtype` |
| `public` member variables | `camelCase` | `hasBias`, `iDtype` |
| `private`, `protected` member variables | `camelCase` followed by `_` | `device_`, `cmdBuffer_` |
| `enum` values | `PascalCase` | `Fp16`, `Bilinear` |
| Constants | `camelCase`, without a `k` prefix | `tokensPerSide` |
| Namespaces | Lowercase | `vai`, `node`, `detail` |
| Macros | `UPPER_CASE` | `VAI_ASSERT` |

The following names are exceptions.

- Tensor dimensions use the letters from their formulas. (Examples: `N`, `K`, `Co`, `Dh`, `Tmax`.)
- Configuration functions for model sizes start with `_` because names cannot start with a digit. (Examples: `_124M()`, `_3B_224()`.)
- The `public` member `config_` of `Node` has a trailing `_` to distinguish it from the constructor parameter `config`.

**3.2 Opening braces**: The following rules apply to each construct.

| Construct | Opening brace |
|---|---|
| Regular functions, member functions | Next line (Allman) |
| `if`, `for`, `while`, `switch` blocks | Next line (Allman) |
| Struct and class declarations | Next line (Allman) |
| Lambda functions, `namespace`, `enum` | Same line |

```cpp
namespace vai::node {

enum class Mode { Fast, Exact };

class Linear : public vai::Node
{
public:
    void prepare()
    {
        if (config_.hasBias)
            addBias();
        auto scaled = [&](uint32_t x) { return x * factor_; };
    }
};

}   // namespace vai::node
```

**3.3 namespace**: Use vai as the general namespace, and vai::node and vai::group for common nodes and node groups.
For custom nodes and node groups, use the model name in place of vai.

```cpp
namespace vai
namespace vai::node
namespace vai::group

// For RT-DETRv2
namespace rtdetrv2
namespace rtdetrv2::node
namespace rtdetrv2::group
```

**3.4 Automated checks**: Rules 3.1 and 3.2 are configured in `.clang-format` and `.clang-tidy` at the repository root and checked automatically on commits and PRs. [Contribution workflow and automated checks](contributing/README.md) covers the remaining formatting rules, such as line length and indentation, and how to pass the checks.
