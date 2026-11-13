# Writing custom models

> **Custom models can be built however you choose**. To include a contributed model in the official distribution, follow the conventions and rules below and submit a PR.

A model is a graph of connected `Node` objects. A checkpoint loader fills each `Node`'s weight slots. This chapter covers the process from implementing a model's graph to verifying its execution. It follows the GPT-2 implementation as an example of model structure, weight loading, and execution checks.

## 1. Creating directories and files

Separate the model structure, weight loading, and execution example. Changes to checkpoint keys or storage formats should be handled by the checkpoint loader.

| File | Role |
|---|---|
| **`models/<impl>/<model>/net.h`** | Model: the complete graph, model configuration, inputs and outputs, and execution state |
| **`models/<impl>/<model>/<distribution>-weights.h`** | Checkpoint loader: reads checkpoints and assigns tensors to weight slots |
| **`models/<impl>/<model>/demo.cpp`** | Execution example: prepares inputs, runs the model, and checks outputs |
| `models/<impl>/<model>/CMakeLists.txt` | Build target of the execution example |
| `models/<impl>/<model>/groups/*.h` | Add when combining nodes and groups into model-specific blocks |
| `models/<impl>/<model>/nodes/*.h` | Add when existing nodes cannot implement the required behavior |
| `models/<impl>/<model>/shaders/*.comp` | Add when new GPU computation is needed |
| `data/ckpts/<model>/<distribution>/` | Default checkpoint directory |
| `data/ckpts/<model>/<distribution>/<size>/` | Size-specific checkpoint directory (for example, `data/ckpts/gpt2/openai/124M/`) |
| `data/tokenizers/<model>/` | Tokenizer files |
| `data/images/` | Input images |

`<impl>` is `production` for a finished implementation and `experimental` for one still in progress.

GPT-2 uses `gpt2` as the model name and `openai` as the checkpoint distribution name. To support another distributor's checkpoint for the same model structure, add a new loader.

Before implementing a custom model, identify the required nodes and check the [Built-in nodes](../06-nodes.md) and [Common groups](../../../vai/groups/). Implement new ones if the built-in nodes and common groups do not provide what is needed. [Using vAi](../03-usage/README.md) covers creating new nodes and groups. This document focuses on building a custom model from those nodes and groups, along with the additional rules for submission to the official repository.

## 2. Building the graph

### 2.1 Separating model structure from runtime settings

First, check the original model's operation order and inputs and outputs. The number of layers, dimensions, activation functions, and normalization method must match the checkpoint. Define the different `Config` structures according to the following distinctions.

| Configuration | Determines | GPT-2 example |
|---|---|---|
| `ModelConfig` | Model structure determined by the checkpoint | Number of layers, embedding dimension, vocabulary size, trained context length |
| `RuntimeConfig` | Runtime settings selected by the user | KV cache dtype, maximum runtime context length |
| `vai::NeuralNetConfig` | Device and engine resources | Device, default weight and output dtypes, buffer pool |

Define the `Config` structures in [`net.h`](../../../models/production/gpt2/net.h). The model constructor then receives these configurations and passes them to each component.
```cpp

Net(const vai::NeuralNetConfig& netCfg, const ModelConfig& modelCfg, const RuntimeConfig& runtimeCfg = {})
: NeuralNet(netCfg)                       // Engine configuration
, modelCfg(modelCfg)
, runtimeCfg(runtimeCfg)
, maxRuntimeContext(std::min(runtimeCfg.maxRuntimeContext, modelCfg.languageCfg.maxTrainedContext))
, embedding({                             // Model structure configuration
      .vocabSize = modelCfg.languageCfg.vocabSize,
      .embedDim = modelCfg.languageCfg.layerCfg.embedDim,
      .posTableSize = modelCfg.languageCfg.maxTrainedContext,
  })
, decoder(modelCfg.languageCfg.with(      // Apply runtime settings to the structure configuration
      { .kvCacheSize = maxRuntimeContext, .kvCacheDtype = runtimeCfg.kvCacheDtype, .sliceLastRow = true }))
, lmHead({ .inDim = modelCfg.languageCfg.layerCfg.embedDim, .outDim = modelCfg.languageCfg.vocabSize })
{
    // ...
}
```

If the model structure is fixed, `ModelConfig` can be omitted. If there is also only one externally configurable runtime setting, a separate `RuntimeConfig` can be omitted.

### 2.2 Assembling nodes and groups

Choose the model's input and output names, then connect nodes and groups in the constructor. The following code is an excerpt from the [GPT-2 model](../../../models/production/gpt2/net.h)'s members and constructor.
```cpp
namespace model::production::gpt2 {

using Transformer = vai::group::Transformer<vai::TransformerType::Classic, vai::QkvLayout::Fused>;

struct Net : public vai::NeuralNet
{
    // ... Model configuration members

    vai::node::TokenEmbedding embedding;
    Transformer decoder;
    vai::node::Linear lmHead;
    vai::node::TopKSampling topkSampling;

    Net(...)
    // ... Class and member initializer list
    {
        input("inTokenIds") - embedding - decoder - lmHead - topkSampling - output("nextTokenId");

        // ... Execution state initialization omitted
    }
};

}   // namespace model::production::gpt2
```

The model holds nodes and groups as members. In the constructor, connect them from the input boundary to the output boundary. In the GPT-2 example above, the order is embedding, the `Transformer` decoder group, output projection, and token sampling.

Connect the model's boundaries with `input()` and `output()`. In addition to the main path, match the slot names for residual paths and multiple inputs.

The addresses of connected nodes and groups must remain stable. Declaring members by value is recommended. For vectors, reserve space with `reserve()` before constructing elements with `emplace_back()`. For other ownership options, see [Precautions when using containers](../03-usage/node-group.md#4-precautions-when-using-containers).

### 2.3 Managing state across repeated execution

After processing a prompt, GPT-2 **calls the model repeatedly** to generate one token at a time. It stores previously computed K/V values in a KV cache and reuses them to avoid recomputing earlier tokens at every step. The model therefore maintains the cache and the number of tokens processed so far. The token count determines the embedding position of the next input and the position at which values are written to the cache.

When implementing a model that needs repeated execution, define the following.

- Information to retain: cache and position information needed by subsequent calls.
- Update timing: update required values before processing new input.
- Reset timing: prevent the previous cache from being used when starting an independent prompt.

In GPT-2, `step()` waits for the previous GPU work to complete before updating position and sampling values. `resetCache()` resets the cache position to zero.

## 3. Loading weights

After building the graph, assign the trained weights to each `Node`. The checkpoint loader adapts the file's tensors to the slots, shapes, dtypes, and value semantics required by the model. Assign weights to the corresponding `Node`'s `persistent` slots. State prepared by a `Node`, such as a cache, follows the initialization rules defined in section 2.

### 3.1 Mapping checkpoint keys to slots

A tensor's name in a checkpoint is called its key. The checkpoint loader maps these keys to the model's weight slots. Keep this mapping in the loader so that the same model structure can be used when keys or the file layout change. The following excerpt from the [GPT-2 loader](../../../models/production/gpt2/openai-weights.h) assigns weights to slots. The loading function receives `model` and `ckpt` as arguments.

```cpp
const vai::Dtype wDt = net.config().wDefaultDtype;   // the net's default weight dtype
net.embedding["weight"] = vai::Tensor(ckpt["wte.weight"], wDt);
net.embedding["posWeight"] = vai::Tensor(ckpt["wpe.weight"], wDt);
```

`wte.weight` and `wpe.weight` are checkpoint keys. `weight` and `posWeight` are slot names defined by the `Node`. `Tensor(ckpt[key], dtype)` reads the tensor at that key in the specified dtype. A `Node` does not configure its weights' dtype; it reads the dtype of the `Tensor` in the slot. Load weights at the net's `wDefaultDtype` and normalization coefficients at FP32. When the same node type occurs repeatedly, use a shared loading function. Have it accept the target node and key prefix, then fill the required slots. Check the node configuration before reading optional parameters. For example, assign bias only to nodes with `hasBias` enabled.

### 3.2 Converting weight shapes and values

First, check the shape and dtype required by the target slot. Even if slot names match, axis order or the meaning of stored values may differ. An optional parameter such as `bias` or `shift` applies only when a `Tensor` is assigned to its slot, so assign every one the checkpoint has.
Check slot requirements in [Built-in nodes](../06-nodes.md) and the corresponding node implementation. The table below shows checkpoint loaders that handle different shapes and value semantics. Each example is in the code for a model included with vAi.

| Difference | Checkpoint loader action | Implementation |
|---|---|---|
| Axes and shapes differ | Use `permute()` to reorder axes. Use `reshape()` to change the shape or `vai::slice()` to read the required rows. | [PaliGemma loader](../../../models/production/paligemma/google-weights.h) |
| Weights are stored separately | Combine them with `vai::concat()` in the order the target node reads them. | [Llama 3 loader](../../../models/production/llama3/meta-weights.h) |
| Values have different meanings | Compute values according to the formula used by the node. | [Gemma 3 loader](../../../models/production/gemma3/google-weights.h) |
| Two slots share weights | Assign the same `Tensor` to both slots. | [GPT-2 loader](../../../models/production/gpt2/openai-weights.h) |

Complete conversions before the first execution. After conversion, check that the target slots have the correct shapes and dtypes.

If the original files are not `.safetensors`, also provide a conversion script named `<distribution>-weights-convertor.py`.

<a id="distribution-weightsh"></a>

### 3.3 Writing the weight-loading function

Provide a function so that calling code can load weights with `loadWeights<"distribution">(net)`.
This function opens the checkpoint and fills the model's weight slots.

The following code is an excerpt from the [GPT-2 loader](../../../models/production/gpt2/openai-weights.h).

```cpp
template <>
inline void loadWeights<"openai", model::production::gpt2::Net>(model::production::gpt2::Net& net)
{
    std::string dir = std::string(VAI_DATA_DIR "/ckpts/gpt2/openai/") + net.modelCfg.variant;
    model::production::gpt2::openai::loadWeights(net, vai::Checkpoint(dir));
}
```

In this example, `loadWeights<"openai">()` opens the checkpoint directory.
`model::production::gpt2::openai::loadWeights()` assigns tensors from that checkpoint to slots.
Separate functions for selecting files and filling slots are recommended.
This allows the slot-filling code to remain unchanged when file paths change.

Define the `loadWeights<"distribution">` function outside any `namespace` block.
Add `inline` when defining it in a header.

The default checkpoint directory is `data/ckpts/<model>/<distribution>/`. If checkpoints come in multiple sizes, store them in separate subdirectories.

```
data/ckpts/gpt2/openai/124M/
data/ckpts/gpt2/openai/355M/
data/ckpts/gpt2/openai/774M/
data/ckpts/gpt2/openai/1.5B/
```

`vai::Checkpoint` opens all `.safetensors` files in a directory and finds tensors by key. The same code therefore works when weights are split across multiple files. Because every `.safetensors` file in the directory is read, keep only that checkpoint's files in it. See [weights.h](../../../vai/engine/weights.h) for details.

## 4. Running the model

A demo (`demo.cpp`) is a useful example of the sequence needed to run a model. Write it in this order: model creation, weight loading, input preparation, execution, and output checks. Handle file reading, preprocessing, and postprocessing in the demo, while the model accepts and returns tensors. If a configuration file defines the model structure, read it first. Then construct the model from that configuration and load its weights.

### 4.1 Input and output requirements

Make the following information available to users.

| Item | Information to specify |
|---|---|
| Names | Input and output names defined in the graph |
| Shapes | Meaning of each axis and supported sizes |
| dtype | Input and output data types |
| Value semantics | How to normalize inputs and interpret outputs |

After preprocessing the original input, create a `Tensor` that meets these requirements. Floating-point inputs can be written with `setFloats()`, which converts them to the tensor's dtype. Since `setData<T>()` copies data without dtype conversion, also check that the storage format matches.

The following example illustrates the call syntax of [`NeuralNet::operator()`](../../../vai/engine/neural-net.h). It assumes that `net` and `input` are ready and all required state updates have been made. The input and output names are `in0` and `out0`, and the output dtype is FP32.

```cpp
vai::Tensor output;
net({{"in0", input}}, {{"out0", &output}});
const float* result = output.getData<float>();
```

Pass each input as a name and tensor pair. Specify each output by the address of the tensor that will receive it. The output tensor can be declared empty.

### 4.2 Managing state in execution methods

If repeated calls require preparation, combine the preparation code and graph call in one method. Keep cache and counter management inside the model. The demo then only needs to call the execution and reset methods.

[GPT-2's `prefill()` and `decode()`](../../../models/production/gpt2/net.h) are examples of this execution interface.

The engine reuses recorded GPU commands, so a node's command-recording code does not run on every call.
Handle values that the CPU must update on every call in the model's execution methods. Before changing GPU-readable state on the CPU, wait for the previous GPU work to complete.

Values retained between executions and reset conditions follow the rules defined in section 2.

## 5. Building and validating

### 5.1 Registering an executable target

Register the executable target in `models/<impl>/<model>/CMakeLists.txt`.
Name it `demo-<model>`.

The following is the [GPT-2 build configuration](../../../models/production/gpt2/CMakeLists.txt).

```cmake
cmake_minimum_required(VERSION 3.16)

project(demo-gpt2)

# demo-gpt2 — one-prompt completion with timings
add_executable(${PROJECT_NAME}
    demo.cpp
)
```

This file needs no further registration. The [model build configuration](../../../models/CMakeLists.txt) finds every `models/<impl>/<model>/CMakeLists.txt`, adds it to the build, and links `vAi`.
Include model-specific headers in the files that use them.

Put a program that shows a task other than the demo's, such as a chat, in `examples/<task>-<model>/`. (Example: `examples/chat-llama3/`.)

If there are model-specific kernels, add `add_dependencies(${PROJECT_NAME} <impl>-<model>-shaders)` to the executable target. The model build configuration collects `models/<impl>/<model>/shaders/*.comp` and creates this shader target. The compiled kernel path is passed through the `<IMPL>_<MODEL>_SHADER_OUTPUT_DIR` macro.

The [YOLOX build configuration](../../../models/experimental/yolox/CMakeLists.txt) is an example.

[Writing custom nodes and kernels](nodes.md) explains how nodes use this path.

### 5.2 Building the demo and comparing reference outputs

See [Quickstart](../02-quickstart.md) for the build environment. The example below builds a demo target. Model demos are part of the build by default (`VAI_BUILD_MODEL_DEMO`). Run the commands from the repository root. For a new model, replace the target name with that model's name.

```bash
cmake -S . -B build
cmake --build build --target demo-gpt2 --config Release
./build/bin/demo-gpt2
```

Check the following before submitting.

- Compare outputs with the reference implementation using the same weights and inputs. Match preprocessing and postprocessing conditions as well.
- Run with supported shapes and dtypes. Also check repeated execution and state resets.
- For models with stochastic outputs, match random-number and sampling conditions. Where possible, also compare numerical outputs before selection.
- Record the inputs, dtype, tolerances, and results used in the comparison.
- Document download, conversion, build, and execution instructions so other contributors can reproduce the results.

### 5.3 Code and comment conventions

| Item | Rule |
|---|---|
| Code style | Follow the [Common code conventions](../index.md#3-common-code-conventions). |
| Models and groups | Put models and model-specific groups in the namespace `model::<impl>::<model>`. Use `struct`, with models inheriting publicly from `vai::NeuralNet` and groups from `vai::NodeGroup`. |
| Model name | Always name the model structure `Net`. Distinguish models by namespace, as in `model::production::gpt2::Net` and `model::production::llama3::Net`. Calling code shortens it with an alias such as `namespace gpt2 = model::production::gpt2;`. |
| Header stamp | Put a block comment in the form `/* === ... === */` at the top of the file. The first field is `Written by`. |
| Model stamp | In `Model`, state the model name and supported sizes. |
| Loader stamp | In `Checkpoint`, state the source, storage path, and any required conversion. |
| Demo stamp | In `Task`, state the execution purpose. In `Assets`, list paths and sources for required external files. |
| Checkpoints | Provide download instructions. For files included in the repository, check their distribution terms and add exceptions to `.gitignore`. |

**Model and group declarations**

Use `PascalCase` for type names, capitalizing the first letter of every word (for example, `ModelConfig` and `RuntimeConfig`). Use `camelCase` for functions and variables, starting the first word in lowercase and capitalizing each subsequent word (for example, `resetState()`, `numLayers`, and `modelCfg`). Put opening braces for functions and structures on the next line.

The following example shows the format of `net.h`. The node configuration and constructor are omitted. Replace `model_name`, the model name in the comment, and the author email with the contributor's information. Keep the structure name `Net`.

```cpp
/*=============================================================================
  Written by   contributor@example.com
  Model        <model name> (<supported sizes>)
=============================================================================*/

namespace model::experimental::model_name {

struct Block : public vai::NodeGroup
{
    // ...
};

struct Net : public vai::NeuralNet
{
    uint32_t numLayers = 0;

    void resetState()
    {
        // ...
    }

    // ...
};

}   // namespace model::experimental::model_name
```

**Checkpoint loader comments**

At the top of the checkpoint loader, state the download source and storage path. If conversion is required, also state the original path and conversion method. The following example is for a checkpoint that requires conversion.

```cpp
/*=============================================================================
  Written by   contributor@example.com
  Checkpoint   Each tier reads data/ckpts/yolox/megvii/<tierName>/model.safetensors.
               Megvii ships PyTorch checkpoints in the official YOLOX release:
                 download   github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0
                              -> data/ckpts/yolox/megvii/<tierName>/model.pth
                 convert    megvii-weights-convertor.py
                              -> data/ckpts/yolox/megvii/<tierName>/model.safetensors
=============================================================================*/
```

**Demo comments**

In `Task`, state the execution purpose. In `Assets`, list the storage path of each required external file first, followed by its source on the next line. If a file is included in the repository, state that fact.

```cpp
/*=============================================================================
  Written by   contributor@example.com
  Task         GPT-2 — completes one prompt.
  Assets       data/tokenizers/gpt2/tokenizer.json
                 from  huggingface.co/openai-community/gpt2
=============================================================================*/
```
