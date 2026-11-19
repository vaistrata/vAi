# vAi Benchmarks

Measures vAi against other inference frameworks under the same conditions.
Every comparison framework is optional (CMake leaves out a bench whose framework it cannot find at the pinned release, see `<model>/CMakeLists.txt`; the vAi bench is always built).

| Benches | Models | Compared against |
|---|---|---|
| [LLM](#llm-benchmarks) | [GPT-2](gpt2/README.md) (124M) · [Llama 3](llama3/README.md) (3.2-1B, 3.2-3B) · [Gemma 3](gemma3/README.md) (270M, 1B, 4B as text) | [llama.cpp](SETUP.md#llamacpp) |
| [VLA](#vla-benchmarks) | [pi0.5](pi05/README.md) (LIBERO) | [vla.cpp](SETUP.md#vlacpp) and [TensorRT Edge-LLM](SETUP.md#tensorrt-edge-llm) on NVIDIA, [MIGraphX](SETUP.md#migraphx) on AMD |

Sizes listed are the ones measured; each bench's `--size` takes more, listed in the model's README.

## Building and running

Each bench lives in `<model>/`, with its own README and `CMakeLists.txt` and one folder per framework (`vai/`, `llama.cpp/`, ...).

Everything a bench reads sits under the repository's `data/`, the path the build compiles in: checkpoints in `data/ckpts/`, tokenizers in `data/tokenizers/`, and files converted for the comparison frameworks in `data/benchmarks/<model>/`; the model's README lists the exact files. Getting a checkpoint in place is covered by the weights header under `models/production/<model>/` and, where there is one, the model's documentation page, not here.

Install the comparison frameworks first, as in [SETUP.md](SETUP.md): CMake looks for them when it configures, and leaves out a bench whose framework it does not find at the pinned release. After installing one later, configure again.

```bash
# Default compiler older than GCC 14 / Clang 18: add -DCMAKE_CXX_COMPILER=g++-14 (see SETUP.md)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVAI_BUILD_BENCH=ON
cmake --build build --config Release
```

The benches land in `build/bin/` (`build/bin/Release/` with a multi-config generator); running one, and its results: see the model's README, linked from the table above.

With more than one GPU, run every framework on the same one; each bench prints the device it uses.

## LLM benchmarks

- Benches, with their results and how to run them: [GPT-2](gpt2/README.md) · [Llama 3](llama3/README.md) · [Gemma 3](gemma3/README.md)
- Comparison framework: [llama.cpp](https://github.com/ggml-org/llama.cpp)

### Requirements

vAi reads the checkpoint where the demos do, the tokenizer from `data/tokenizers/<model>/` and the weights from `data/ckpts/<model>/<source>/<size>/`. The comparison framework reads a file converted from it, in `data/benchmarks/<model>/<size>/`. The model's README lists the exact files. Installing llama.cpp and converting the models: see [SETUP.md](SETUP.md#llm-benchmarks).

| | Format | Found by (CMake) |
|---|---|---|
| vAi | `tokenizer.json` + `.safetensors` | always built |
| llama.cpp | `.gguf` | `llama_DIR` |

### Shared conditions

#### Measurement

Whatever the model, the LLM benches share one code skeleton and one measurement protocol.

| | Shared condition |
|---|---|
| Model size | `--size <name>`: a folder under `data/ckpts/<model>/<source>/` and `data/benchmarks/<model>/`. Defaults: GPT-2 `124M`, Llama 3 `3.2-1B`, Gemma 3 `1B` |
| Input | `common/bench-inputs.h`: a prompt of N tokens is the first N tokens of one paragraph |
| Warmup | One global run at the longest length, then per length: 1 prefill, 8-token decode, 1 prefill |
| Measurement | Per length, timed separately: five prefills, each after an untimed decode step, and five 128-token decodes, each after an untimed prefill |
| Timing | Each timed prefill, and each step of a timed decode, ends with its sampled token copied to host memory |
| Reported | Tokens per second, **mean ± SD**; a row whose SD exceeds 10% of its mean is flagged for a rerun |

#### Matched settings

Where llama.cpp's defaults differ from vAi, the bench sets llama.cpp to match.

| | vAi | llama.cpp default | llama.cpp in the bench |
|---|---|---|---|
| Prefill | The whole prompt in one forward | 512-token chunks | One forward, with `n_ubatch = n_ctx` |
| Token embedding | On the GPU | On the CPU | On the GPU, with `tensor_buft_overrides` |
| Sampling | Top-k 40, then a draw at T = 1, on the GPU | On the CPU | On the GPU, with a sampler chain: top-k 40, then a T = 1 draw |

### Where the frameworks differ

What the benches leave to each framework, to keep in mind when reading the results.

| | vAi | llama.cpp |
|---|---|---|
| Vulkan kernels | `KHR_cooperative_matrix` | `NV_coopmat2` where the driver exposes it, otherwise `KHR_cooperative_matrix` |
| Command submission | One command buffer per forward, recorded when the input shape changes and then submitted whole; later calls only resubmit it | Recorded on every call and submitted in chunks as it is recorded, so recording overlaps GPU work |
| Precision | FP16 weight matrices and KV cache, FP32 norms; GPT-2's linear biases and position embedding FP16 | The same, except GPT-2's linear biases and position embedding, which the converter keeps F32. Tested to cost vAi no precision: see the [GPT-2 README](gpt2/README.md#summary) |

## VLA benchmarks

- Benches, with their results and how to run them: [pi0.5](pi05/README.md)
- Comparison frameworks: [vla.cpp](https://github.com/VinRobotics/vla.cpp) and [TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) on NVIDIA, [MIGraphX](https://github.com/ROCm/AMDMIGraphX) on AMD

### Requirements

vAi reads the checkpoint where the demos do, the tokenizer from `data/tokenizers/paligemma/` and the weights from `data/ckpts/<model>/lerobot/<size>/`. The comparison frameworks read files converted from it, in `data/benchmarks/<model>/<size>/`. The model's README lists the exact files. Installing the comparison frameworks, which run on Linux only, and making their model files: see [SETUP.md](SETUP.md#vla-benchmarks).

| | Format | Found by (CMake) |
|---|---|---|
| vAi | `tokenizer.json` + `.safetensors` | always built |
| vla.cpp | `.gguf` | `vla_ROOT` |
| TensorRT Edge-LLM | TensorRT engines | `TensorRTEdgeLLM_ROOT` |
| MIGraphX | FP16 ONNX bundle | `migraphx_DIR` |

### Shared conditions

#### Measurement

Whatever the framework, the VLA benches share one code skeleton (`runVLABench` in `common/bench-utils.h`) and one measurement protocol.

| | Shared condition |
|---|---|
| Input | `common/bench-inputs.h`: the camera images, the task and the robot state; the noise and the preprocessed images come from the model's reference script when it has written them |
| Accuracy check | One untimed prediction, compared with the PyTorch FP32 reference: `--atol 0.002 --rtol 0.001` by default. A bench that fails it prints no latency; one without the reference prints `accuracy: unverified` |
| Warmup | 3 predictions |
| Measurement | 20 predictions, each timed from uploading its host inputs to the complete action chunk in host memory |
| Reported | Milliseconds per prediction, **mean ± SD**, and the fastest; the maximum absolute error against the reference |

#### Matched settings

vAi and the framework it is compared with run the same configuration, a `VlaConfig` in `<model>/common/vla-config.h`: the checkpoint size, the policy, the cameras, the action chunk and the denoise steps. The comparison bench always runs its own; the vAi bench picks it with `--comparison <framework>`.

| | Matched how |
|---|---|
| Policy | The framework's own: the LeRobot policy puts the discretized robot state in the prompt; openpi's, which TensorRT Edge-LLM serves, leaves it out |
| Action chunk, denoise steps | The framework's: vAi runs whichever the configuration names |
| Noise, images | The reference's, read by every bench, so the comparison covers the model, not the image preprocessing |

### Where the frameworks differ

What the benches leave to each framework, to keep in mind when reading the results.

| | vAi | vla.cpp | TensorRT Edge-LLM | MIGraphX |
|---|---|---|---|---|
| GPU API | Vulkan | CUDA (ggml) | CUDA (TensorRT) | HIP (ROCm 10.1) |
| Model | The checkpoint, read directly | GGUF, converted | Engines built on the GPU from its own FP16 ONNX export | FP16 ONNX graphs, compiled on the GPU when the bench starts. MIGraphX has no official pi0.5 implementation: the export and the runtime loop are the bench's own |
| Precision | FP16 weights, K/V cache and block branches; FP32 vision accumulation and vision / language residuals; FP16 action expert residual | Framework-specific | Framework-specific | As exported, then as MIGraphX compiles it |
| Action output | Normalized, as the model produces it | Un-normalized with the GGUF's dataset statistics; the bench inverts it outside the timed prediction | Normalized | Normalized |
