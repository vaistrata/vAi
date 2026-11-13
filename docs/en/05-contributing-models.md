# Writing custom models

> **You are free to create custom models however you like**. However, to include a model in the official distribution, follow the conventions and rules below and then submit a PR.

## Component files

A model has three basic files, with two more when dedicated operations are needed.

| File                                                |          | Description                                            |
| --------------------------------------------------- | -------- | ------------------------------------------------------ |
| `models/<impl>/<model-name>/net.h`                       | Required | Model definition                                       |
| `models/<impl>/<model-name>/nodes/*.h`                     | Optional | Nodes not provided by default                          |
| `models/<impl>/<model-name>/shaders/*.comp`                | Optional | Kernels used by the nodes above                        |
| `models/<impl>/<model-name>/<distribution-name>-weights.h` | Required | Loader for reading a specific distributor's checkpoint |
| `models/<impl>/<model-name>/demo.cpp`                    | Required | Demo that runs the model in practice                   |

`<impl>` is `production` for a maintained implementation and `experimental` for one still in progress.
Checkpoints go in `data/ckpts/<model-name>/<distribution-name>/`, tokenizers in `data/tokenizers/<model-name>/`, and input images in `data/images/`.

The process of creating a model is covered in [Using vAi](03-usage/README.md). See
[NodeGroup](03-usage/node-group.md) for how to group repeated blocks and
[NeuralNet](03-usage/neural-net.md) for how to connect everything and specify the input and output boundaries. This document covers only the additional
rules for adding a model to the repository.

## Common rules

Follow the [common code conventions in the Overview](index.md#common-code-conventions).

## net.h

#### 1. File header stamp

State the author and contents at the very top of the file. `Written by` is
the author's signature (email) and always goes on the first line. In `Model`, list the model being implemented and the supported
sizes. In `Modules`, list every node group defined in this file.

```cpp
/*=============================================================================
  Written by   oneway@vaistrata.com
  Model        GPT-2 (124M / 355M / 774M / 1.5B)
  Modules      MultiHeadAttention, FeedForward, TransformerBlock
=============================================================================*/
```

#### 2. Namespace

Place all models and node groups defined in this file in a dedicated
namespace.

```cpp
namespace <net-name> {

...

} // namespace <model-name>
```

#### 3. Inherit using `struct`

Inherit from `vai::NodeGroup` and `vai::NeuralNet` using `struct`, not
`class`.

```cpp
struct FeedForward : public vai::NodeGroup { ... };
struct Net : public vai::NeuralNet { ... };
```

#### 4. The model struct must be named `Net`

Outside the file, models are distinguished by namespace, as in `model::production::gpt2::Net` and
`model::production::llama3::Net`. If one file must contain multiple models,
separate them with nested namespaces. Code that uses a model shortens the name with an alias such as
`namespace gpt2 = model::production::gpt2;`.

#### 5. Own nodes by value

Declare member `vai::Node` and `vai::NodeGroup` objects by
value. Do not own them through raw pointers or references. This restriction does not apply to
ordinary members that are not nodes.

```cpp
struct Net : public vai::NeuralNet {
    vai::node::TokenEmbedding embedding;          // OK
    MultiHeadAttention mhAttn;             // OK. Node groups are values too
    std::vector<TransformerBlock> blocks;  // OK. A container holding values
    std::unique_ptr<Tokenizer> tokenizer;  // OK. Pointers are allowed for non-nodes
    vai::node::Linear* lmHead;               // X. Raw pointer
    vai::node::RmsNorm& norm;                // X. Reference
};
```

For precautions when placing nodes in a container, see
[Precautions when using containers in NodeGroup](03-usage/node-group.md#precautions-when-using-containers).

## distribution-name-weights.h

Trained weights must be loaded to run a model. Weights are provided in
various formats by different distributors, and vAi currently supports the
`.safetensors` format, which is readily available on HuggingFace.

However, even among `.safetensors` checkpoints, key names, tensor shapes, layer fusion methods,
and other details vary. A loading helper is therefore needed to map and
convert the checkpoint weights to the vAi model structure. Each model contributor must write and
provide this helper. This allows various checkpoints to be
loaded with a single `net.h` implementation.

Name the file after the organization that trained or distributed the checkpoint. For example,
use `openai-weights.h` for a checkpoint distributed by OpenAI. If multiple organizations distribute
checkpoints for the same model, add one file for each distributor.

This file must provide the following:

#### 1. File header stamp

The checkpoint source and storage path must be stated.
Users must be able to tell where to obtain the checkpoint and where to place it
by looking only at the stamp in the file header.

```cpp
/*=============================================================================
  Written by   oneway@vaistrata.com
  Checkpoint   Each size reads data/ckpts/gpt2/openai/<ModelConfig::variant>/model.safetensors,
               taken from the matching HuggingFace repo:
                 124M   openai-community/gpt2
                 355M   openai-community/gpt2-medium
                 774M   openai-community/gpt2-large
                 1.5B   openai-community/gpt2-xl
=============================================================================*/
```

#### 2. `loadWeights` specialization

It must be possible to load a checkpoint externally with the single line
`loadWeights<"distribution-name">(model)`. The call site does not need to know the checkpoint's key names or
actual file path.

The entry point is the function template declared in [`vai/engine/weights.h`](../../vai/engine/weights.h),
and the implementation for each distributor is specialized by passing the distribution name as a string NTTP.

```cpp
namespace model::production::gpt2::openai {

inline void loadWeights(Model& net, const vai::SafetensorsParser& parser)
{
    // Map checkpoint keys to model node slots
}

} // namespace model::production::gpt2::openai


template<>
inline void loadWeights<"openai", model::production::gpt2::Net>(model::production::gpt2::Net& net)
{
    std::string path = std::string(VAI_DATA_DIR "/ckpts/gpt2/openai/")
                     + net.modelCfg.variant + "/model.safetensors";
    model::production::gpt2::openai::loadWeights(net, vai::SafetensorsParser(path.c_str()));
}
```

The checkpoint path is also assembled inside this specialization. Therefore, demos and other code that use the model
do not need to know the checkpoint's specific path or file layout.


#### 3. (Optional) Conversion script

If the source checkpoint is distributed in a format other than `.safetensors`,
provide a conversion script as well. Use the filename
`<distribution-name>-weights-convertor.py`. Do not put a stamp in the script;
include the source and the paths before and after conversion in the `Checkpoint` stamp above.

For example, yolov8 cannot read the `.pt` checkpoint distributed by Ultralytics
as is, so it also provides
[`ultralytics-weights-convertor.py`](../../models/internal/yolov8/ultralytics-weights-convertor.py).
This script converts it to `.safetensors`, which vAi can read.

```cpp
/*=============================================================================
  Written by   oneway@vaistrata.com
  Checkpoint   Each tier reads data/ckpts/yolov8/ultralytics/<tierName>/model.safetensors.
               Ultralytics ships .pt, so the file takes one conversion:
                 download   huggingface.co/Ultralytics/YOLOv8
                              -> data/ckpts/yolov8/ultralytics/<tierName>/model.pt
                 convert    ultralytics-weights-convertor.py
                              -> data/ckpts/yolov8/ultralytics/<tierName>/model.safetensors
=============================================================================*/
```

## demo.cpp

`demo.cpp` is an executable program that shows a representative use of the model. The task is up to the contributor.

#### 1. File header stamp

In `Task`, state what the program does, and in `Assets`, list
the external files required to run it. List each file's storage path first, then identify its source on the next line, indented one level. If a file is already included in the repository, state that fact.

```cpp
/*=============================================================================
  Written by   oneway@vaistrata.com
  Task         GPT-2 — completes one prompt, profiles prefill and decode.
  Assets       data/tokenizers/gpt2/tokenizer.json
                 from  huggingface.co/openai-community/gpt2
=============================================================================*/
```

#### 2. Project registration

Register it in `models/<impl>/<model-name>/CMakeLists.txt`, using
`project(demo-<model-name>)` as the project name. Programs for other tasks go in `examples/<task>-<model-name>/`, as `examples/chat-llama3/` does.

```cmake
cmake_minimum_required(VERSION 3.16)

project(demo-llama3)

# demo-llama3 — one-prompt completion with timings
add_executable(${PROJECT_NAME}
    demo.cpp
)
```


## Dedicated nodes and kernels (optional)

Model-specific nodes and kernels may be added when an operation cannot be expressed with vAi's default nodes, or when a special interface, layer fusion, optimization for a specific target, or similar feature is needed.

Use the `<model-name>-<node-name>` format for filenames and keep the node and kernel names identical when possible.

```
models/<impl>/<model-name>/
  nodes/<model-name>-<node-name>.h        node
  shaders/<model-name>-<node-name>.comp   kernel
```

Kernels in `shaders/` are compiled by `models/CMakeLists.txt` into the `<impl>-<model-name>-shaders` target, and the output directory reaches the code as `<IMPL>_<MODEL-NAME>_SHADER_OUTPUT_DIR`. A program that runs them only adds the dependency.

The following is an example from `models/internal/yolov8/CMakeLists.txt`.

```cmake
add_executable(${PROJECT_NAME}
    demo.cpp
)

add_dependencies(${PROJECT_NAME} internal-yolov8-shaders)
```

The node loads the compiled `.spv` file from that path. The following is an example from `decode.h`.

```cpp
pipe = pipelinePool().request(INTERNAL_YOLOV8_SHADER_OUTPUT_DIR, "decode", { ... });
```

For details on writing nodes and kernels, see
[Writing custom nodes and kernels](07-contributing-nodes.md).

## Promotion to official vAi nodes

Even a node placed in a model folder is promoted to an official node, together with its related kernel, after internal testing if it meets any of the following conditions.

- A general-purpose node that vAi does not currently provide
- A node that already exists in vAi but performs better in a range of environments
- A node that already exists in vAi and has similar performance but better usability

## Checkpoint

Place checkpoints under `data/ckpts/<model-name>/<distribution-name>/` and state the path in the loader's `Checkpoint` stamp.

Do not commit large checkpoints or checkpoints with license restrictions.
The [`.gitignore`](../../.gitignore) blocks `*.safetensors`, `*.json`, and `*.pt` under
`data/`. Only small, self-trained checkpoints may be committed by adding an exception to `.gitignore`.
`data/ckpts/mnist/jroot/model.safetensors` is a representative example.
