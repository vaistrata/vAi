# Overview

vAi is a **Vulkan inference engine built on a graph DSL**. It assembles model graphs using a C++ DSL (a dedicated syntax for connecting operator nodes) and executes them using Vulkan Compute Shaders.

## Features

### 1. Portability

Run and test on a wide range of GPUs and SoCs with a single codebase. Because vAi uses only the industry-standard Vulkan API, there is no need to rewrite code from scratch for vendor-specific SDKs from NVIDIA, AMD, Intel, Qualcomm, and others. Switching devices, which traditionally required reimplementation, is reduced to a single build with vAi.

### 2. Lightweight

When deploying LLMs with vAi, the runtime binary is only around 2 MB. Its dependency chain has been carefully optimized from the ground up for edge devices. As a result, selective builds, which include only the logic and operator kernels actually used, are enabled by default with no additional configuration.

### 3. High Performance

The CPU does very little work during each inference. Conventional inference engines typically traverse the computation graph and submit kernel execution commands to the GPU on every inference. vAi instead creates the command sequence once and reuses it repeatedly. In addition, GPU kernels are tuned for each device, so you don't have to sacrifice performance for portability.

### 4. Flexibility

Vision models and LLMs can be handled together within a single runtime. Typically, vision models are handled by general-purpose inference engines such as ONNX Runtime, ncnn, or ExecuTorch, while models requiring optimization, such as LLMs, use dedicated frameworks such as llama.cpp, vLLM, or TensorRT-LLM. vAi leverages polymorphism to let each operator define its own dynamically updated state, such as the KV cache, and how that state is transitioned. This allows vAi to handle vision models, LLMs, VLMs, and VLAs without adding another runtime layer.

### 5. Intuitiveness

With its graph DSL, the model structure is directly represented in code. vAi uses a dedicated graph syntax and object-oriented design, allowing you to build a model by connecting nodes like LEGO bricks. LLaMA 3 transformer block can be written as follows:

```cpp
struct TransformerBlock : public vai::NodeGroup {
    vai::node::Identity input;
    vai::node::RmsNorm norm1;
    MultiHeadAttention mhAttn;
    vai::node::Add add1;
    vai::node::RmsNorm norm2;
    GatedFeedForward ffn;
    vai::node::Add add2;

    TransformerBlock(...)
    {
        input - norm1 - mhAttn - "in0"/add1 - norm2 - ffn - "in0"/add2;
        input                  - "in1"/add1               - "in1"/add2;
    }
};
```

The member list is the block's structure, and the connections represent the data flow. The first line in the constructor represents the main path, while the second represents the residual path, making the design visible at a glance. Because the code serves as the model's structural diagram, it is easy to read and modify.

### 6. Extensibility

You can create and add new operators or layers with custom optimizations. vAi lets you inherit from a node class and implement only the kernel you need **without modifying a single line of core code**. You can share newly created kernels with anyone who needs them without waiting for them to be merged into the upstream project. There is no need to maintain a fork or conform to a plugin specification.

## Project Structure

```
vai/                        Core engine (vAi library)
  engine/
    neural-net.h
    tensor.h
    safetensors-parser.h
    weights.h
    buffer-pool.h
    pipeline-pool.h
    error.h
  nodes/                    Built-in nodes
  shaders/                  Built-in kernels (.comp)
  utils/

data/                       Model checkpoints · input data, etc.

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

## Common Code Conventions

**1. CamelCase**: Types start with an uppercase letter, while functions and variables start with a lowercase letter.

```cpp
class Linear;        // Type
void addSlot(...);       // Function
uint32_t numHeads;       // Variable
```

**2. Opening Braces**: The following rules apply depending on the construct.

| Construct | Opening Braces |
|---|---|
| `if` · `for` blocks | Allman |
| Regular functions · member functions | Allman |
| Lambda functions | Flexible |
| Struct · class declarations | Flexible |

```cpp
// Allman style
void foo()
{
    ...
}
```
