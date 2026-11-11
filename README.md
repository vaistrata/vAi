# vAi

**A cross-platform AI inference engine on Vulkan.** It runs a range of models, including vision, language and multimodal models, in one C++ runtime. Model structure and weight mappings are defined in code, with support for adding operators and GPU kernels.

The graph below defines an MNIST model. The `ConvSequence` definition and weight loading code are in the [full example](examples/mnist/main.cpp).

```cpp
struct Model : public vai::NeuralNet {
    ConvSequence convX2;
    vai::node::Flatten flatten;
    vai::node::Linear linear;

    Model(const vai::NeuralNetConfig& cfg)
    : NeuralNet(cfg), convX2({1, 32, 64}, 3)
    , linear({.inDim = 7 * 7 * 64, .outDim = 10, .hasBias = true})
    {
        input("in0") - convX2 - flatten - linear - output("out0");
    }
};
```

```cpp
vai::Tensor output;
model({{"in0", input}}, {{"out0", &output}});
```

## Features

- Portability: Vulkan provides a common execution layer, so supported GPUs use the same model code and API.
- Execution efficiency: GPU commands are reused while execution conditions remain unchanged, reducing repeated CPU work. Kernels are selected for the input shape and device.
- Flexibility: Nodes manage state that changes during inference, such as the KV cache, so one runtime can handle vision models, LLMs, VLMs and VLAs.
- Intuitiveness: Operators are declared as C++ members and connected with `-`, expressing model structure and data flow in code.
- Extensibility: Node subclasses can add [operators and GPU kernels](docs/en/07-contributing-nodes.md) without modifying the engine core.

## Supported models

Size presets and weight loaders are covered in the [model list](docs/en/04-models/README.md#list).

| Model | Task | Example |
|---|---|---|
| MNIST | Image classification | [Quick Start](docs/en/02-quickstart.md) |
| YOLOX | Object detection | [demo.cpp](examples/yolox/demo.cpp) |
| GPT-2 | Text generation | [demo.cpp](examples/gpt2/demo.cpp) |
| Llama 3.2 | Text generation / multi-turn chat | [demo.cpp](examples/llama3/demo.cpp), [chat.cpp](examples/llama3/chat.cpp) |
| Gemma 3 | Text generation / multi-turn chat (text and image input) | [demo.cpp](examples/gemma3/demo.cpp), [chat.cpp](examples/gemma3/chat.cpp) |
| PaliGemma | Text generation from image and text input | [demo.cpp](examples/paligemma/demo.cpp) |
| π0 | Action chunks from images, instructions and state | [demo.cpp](examples/pi0/demo.cpp) |
| π0.5 | Action chunks from images, instructions and state | [demo.cpp](examples/pi05/demo.cpp) |
| Whisper | Speech recognition (transcription) | [demo.cpp](examples/whisper/demo.cpp) |

## Build

Requires the Vulkan development libraries and `glslc` (Vulkan SDK), CMake 3.16+ and a C++23 compiler (GCC 14+ on Linux), with Make or Ninja. The default build produces only the library. On Linux, prefix the configure command with `CC=gcc-14 CXX=g++-14` to select GCC 14.

```bash
git clone --recursive https://github.com/vaistrata/vAi.git
cd vAi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

`VAI_BUILD_EXAMPLES`, `VAI_BUILD_TESTS`, `VAI_BUILD_BENCH` and `VAI_BUILD_PYTHON` all default to `OFF`.

## Documentation

| Section | Contents |
|---|---|
| [Overview](docs/en/index.md) | Features and project structure |
| [Quick Start](docs/en/02-quickstart.md) | Build requirements and limitations, MNIST example |
| [Using vAi](docs/en/03-usage/README.md) | Node, NodeGroup, NeuralNet and Tensor |
| [Supported models](docs/en/04-models/README.md) | Implemented presets, examples and weight loaders |
| [Writing custom models](docs/en/05-contributing-models.md) | Adding a model |
| [Built-in nodes](docs/en/06-nodes.md) | Node reference |
| [Writing custom nodes and kernels](docs/en/07-contributing-nodes.md) | Adding an operator |

## Performance

See the [benchmark documentation](docs/en/04-models/README.md#aggregate-benchmark) for each model's
measurement environment and results. Comparisons require matching GPUs, precision,
input sizes and timing boundaries.

## License

The engine code is licensed under [Apache 2.0](LICENSE). Model weights and external dependencies retain their respective licenses.
