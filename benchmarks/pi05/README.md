# pi0.5 Benchmarks

## Summary

Runs pi0.5 under vAi and comparison frameworks on the same GPU and compares end-to-end action prediction latency.

- **Frameworks:** vAi vs. vla.cpp and TensorRT Edge-LLM on NVIDIA; vAi vs. MIGraphX on AMD.
- **Checkpoint:** The same pi0.5 LIBERO checkpoint for all frameworks; comparison frameworks read files converted from it.
- **Workload:** Two camera images, a language prompt, and action generation with 10 denoising steps.
- **Conditions and differences:** [benchmarks/README.md](../README.md#vla-benchmarks). GPU APIs, precision handling, and command submission differ and may affect latency.

> **IMPORTANT**
> The frameworks use different execution backends and precision configurations. All results are checked against a PyTorch FP32 reference. Latencies are reported only for predictions that pass the specified accuracy tolerances.

## Model preparation

Begin once the pi0.5 checkpoint is available:

`data/ckpts/pi05/lerobot/<size>/model.safetensors`

The checkpoint comes from the repository listed in [lerobot-weights.h](../../models/production/pi05/lerobot-weights.h).

The vAi benchmark also needs:

- PaliGemma tokenizer: `data/tokenizers/paligemma/tokenizer.json`
- Camera images: `data/images/cam_high.jpg`, `cam_left_wrist.jpg`, `cam_right_wrist.jpg`

The tokenizer is distributed separately through `google/paligemma-3b-pt-224`. Accept the repository license and download it with a Hugging Face token.

```bash
curl -fL --create-dirs \
  -H "Authorization: Bearer $HF_TOKEN" \
  -o data/tokenizers/paligemma/tokenizer.json \
  https://huggingface.co/google/paligemma-3b-pt-224/resolve/main/tokenizer.json
```

### Comparison frameworks

Each comparison framework is installed by its setup script in [`benchmarks/scripts/`](../scripts/). The scripts only build or unpack the framework; they build no model files. Everything goes under `${XDG_CACHE_HOME:-~/.cache}/vai/`:

| Framework | Setup script | Install, linked at | Downloads kept for re-runs |
|---|---|---|---|
| vla.cpp v0.4.0 | [`setup-vlacpp.sh`](../scripts/setup-vlacpp.sh) | `vla.cpp-v0.4.0`, `vla.cpp` | `vla.cpp-v0.4.0-src`: the clone, and the llama.cpp it fetches in `_deps/` |
| TensorRT Edge-LLM 0.11.1 | [`setup-trt-edge-llm.sh`](../scripts/setup-trt-edge-llm.sh) | `TensorRT-Edge-LLM-v0.11.1`, `TensorRT-Edge-LLM` | `TensorRT-Edge-LLM-v0.11.1-src`: the clone with its submodules and the CuTe DSL build environment |
| MIGraphX 2.18.0 + ROCm 10.1.0 runtime | [`setup-migraphx.sh`](../scripts/setup-migraphx.sh) | `migraphx-2.18.0`, `migraphx` | `migraphx-2.18.0-download`: AMD's prebuilt packages and the ROCm tarball of the GPU |

[CMakeLists.txt](CMakeLists.txt) finds each install through its link, without flags; `vla_ROOT`, `TensorRTEdgeLLM_ROOT` or `migraphx_DIR` names another. A later run reuses the downloads without the network; `--uninstall` removes the install, its downloads and the link.

### Model files

The comparison benchmarks read converted files under:

`data/benchmarks/pi05/<size>/`

| Framework | Converted model files | Made by |
|---|---|---|
| vAi | Original `model.safetensors` | — |
| vla.cpp | `model.gguf` | Downloaded: vla.cpp's [vrfai/pi05-libero-gguf](https://huggingface.co/vrfai/pi05-libero-gguf), see [SETUP.md](../SETUP.md#vlacpp-model-conversion) |
| MIGraphX | `prefill.fp16.onnx`, `step.fp16.onnx`, `.weights.bin`, `config.txt` | [`scripts/export-migraphx.sh`](scripts/export-migraphx.sh) |
| TensorRT Edge-LLM | `engines/` | [`scripts/export-trt.sh`](scripts/export-trt.sh) |

- MIGraphX has no official pi0.5 implementation: the ONNX export ([`migraphx/export-onnx.py`](migraphx/export-onnx.py)) and the runtime loop ([`migraphx/net.h`](migraphx/net.h)) are this repo's own, written for this bench, and the MIGraphX results measure them.
- `export-migraphx.sh` creates the Python environment [`migraphx/export-onnx.py`](migraphx/export-onnx.py) checks for (lerobot 0.6.0, transformers 5.5.4, torch 2.8.0 for the CPU, onnx 1.19.0) in `~/.cache/vai/venv-pi05-export` once, then runs the export on the CPU. It needs a C++23 compiler (g++ 14 or clang++ 18) for the input helper; arguments after `--` go to `export-onnx.py`, such as `--no-constant-folding`.
- `export-trt.sh` installs Edge-LLM's exporter (`trtedge`) from the source `setup-trt-edge-llm.sh` keeps into `~/.cache/vai/venv-trt-export` once, exports FP16 ONNX as openpi's `pi05_libero`, adds its normalization statistics, and builds the engines on this GPU.
- Neither overwrites existing model files: remove them to export again.

The converters require additional configuration files from the checkpoint's repository:

```bash
curl -fL -o data/ckpts/pi05/lerobot/<size>/config.json \
  https://huggingface.co/<repository>/resolve/main/config.json

curl -fL -o data/ckpts/pi05/lerobot/<size>/policy_preprocessor.json \
  https://huggingface.co/<repository>/resolve/main/policy_preprocessor.json

curl -fL -o data/ckpts/pi05/lerobot/<size>/policy_postprocessor.json \
  https://huggingface.co/<repository>/resolve/main/policy_postprocessor.json
```

### Configurations

Each comparison runs one `VlaConfig` from [`common/vla-config.h`](common/vla-config.h), the same on both sides: the dataset, the policy, the cameras, the action chunk and the denoise steps. The comparison benchmarks run their own; the vAi benchmark picks one with `--comparison <framework>`.

| Comparison | Framework | `VlaConfig` | Policy | Cameras | Action chunk | Denoise steps |
|---|---|---|---|---:|---:|---:|
| `vlacpp` | vla.cpp | `lerobotLibero` | LeRobot | 2 | 50 | 10 |
| `trt` | TensorRT Edge-LLM | `openpiLibero` | openpi `pi05_libero` | 2 | 10 | 10 |
| `migraphx` | MIGraphX | `lerobotLibero` | LeRobot | 2 | 50 | 10 |

The benchmark uses reference actions, input noise, and preprocessed images generated by [reference.py](scripts/reference.py), run with the configuration's values in the same environment as `export-onnx.py`:

```bash
~/.cache/vai/venv-pi05-export/bin/python benchmarks/pi05/scripts/reference.py --cameras 2 --chunk 50 --steps 10
~/.cache/vai/venv-pi05-export/bin/python benchmarks/pi05/scripts/reference.py --cameras 2 --chunk 10 --steps 10 --policy openpi
```

> **NOTE**
> LeRobot includes the discretized robot state in the prompt. The TensorRT Edge-LLM workload follows openpi's task-only prompt. Action chunk sizes also differ. Compare frameworks only within the same workload.

> **NOTE**
> vla.cpp's GGUF, [vrfai/pi05-libero-gguf](https://huggingface.co/vrfai/pi05-libero-gguf), carries the LIBERO dataset statistics, which vla.cpp applies to the actions it returns. The benchmark reverses that normalization outside the timed prediction, to compare with the reference.

## Building and running

Build as described in [benchmarks/README.md](../README.md#building-and-running).

On an NVIDIA GPU:

```bash
build/bin/bench-pi05-vai --comparison vlacpp
build/bin/bench-pi05-vlacpp

build/bin/bench-pi05-vai --comparison trt
build/bin/bench-pi05-trt
```

On an AMD GPU:

```bash
build/bin/bench-pi05-vai --comparison migraphx
build/bin/bench-pi05-migraphx
```

[`scripts/run-amd.sh`](scripts/run-amd.sh) runs the two AMD benchmarks in turn with the GPU clock pinned to its highest DPM level, restores the level afterwards, and prints the clock and power each one ran at. Arguments after `--` go to both, as in `scripts/run-amd.sh -- --atol 0.2 --rtol 0.1`.

Options:

- `--comparison <name>`: Select the framework the vAi benchmark is compared with, and so its configuration: `migraphx` (default), `vlacpp`, or `trt`.
- `--showActions`: Print the first five generated actions.
- `--atol N`, `--rtol N`: Set accuracy tolerances (defaults: `0.002` and `0.001`).
- vla.cpp: `--model <path>` and `--config <path>` select another GGUF model and runtime configuration.
- MIGraphX: `[bundle-dir]` selects another ONNX export bundle.
- TensorRT Edge-LLM: `--engineDir <path>` selects another engine directory.

*A benchmark prints no latency if its prediction fails the configured accuracy check. Without a reference file, it reports `accuracy: unverified`.*

## Results

Each latency cell is **mean ± SD in milliseconds per prediction**, as printed by the benchmark. `min` is the fastest repetition. Lower latency is better.

The maximum absolute error is calculated against the corresponding PyTorch FP32 reference.

The `vAi` column names the commit the results hold for. They were measured with the precision `8552ebe` chose; `c3fc170` only moves where each tower's precision is set, and the later changes rename the benchmark configurations (`VlaConfig`) and reorganize the setup and export scripts. None changes the model, its precision or the inputs.

### pi0.5 LIBERO

#### NVIDIA GeForce RTX 5090

| OS | Vulkan driver | Compute stack | GPU clock (core / memory) | vAi |
|---|---|---|---|---|
| Ubuntu 24.04.5 LTS | NVIDIA 595.99.02 | CUDA 13.2, TensorRT 10.16.1, TensorRT Edge-LLM 0.11.1, vla.cpp v0.4.0 | Default | `c3fc170` |

| Comparison | Image + text tokens | Framework | Latency (ms) | Min (ms) | Max abs error |
|---|---:|---|---:|---:|---:|
| `vlacpp` | 512 + 154 | **vAi** | **39.19 ± 0.03** | 39.14 | 0.00112 |
| | | vla.cpp | 42.19 ± 0.14 | 42.05 | 0.00142 |
| `trt` | 512 + 17 | vAi | 33.87 ± 0.06 | 33.74 | 0.000755 |
| | | **TensorRT Edge-LLM** | **21.10 ± 0.01** | 21.07 | 0.00417 |

`trt`'s TensorRT Edge-LLM row is measured with `--atol 0.2 --rtol 0.1`; the other rows pass the default tolerances.

#### AMD Strix Halo (Radeon 8060S, gfx1151)

| OS | Vulkan driver | Compute stack | GPU clock (core / memory) | vAi |
|---|---|---|---|---|
| Ubuntu 24.04.4 LTS | RADV, Mesa 25.2.8 | MIGraphX 2.18.0, ROCm 10.1.0 | 2900 / 1000 MHz (`high`) | `c3fc170` |

| Comparison | Image + text tokens | Framework | Latency (ms) | Min (ms) | Max abs error |
|---|---:|---|---:|---:|---:|
| `migraphx` | 512 + 154 | **vAi** | **151.73 ± 3.98** | 148.82 | 0.00116 |
| | | MIGraphX | 195.33 ± 4.17 | 190.27 | 0.00221 |

Each row is the median of three `scripts/run-amd.sh -- --atol 0.2 --rtol 0.1` runs.

#### AMD Strix Point (Radeon 890M, gfx1150)

| OS | Vulkan driver | Compute stack | GPU clock (core / memory) | vAi |
|---|---|---|---|---|
| Ubuntu 26.04.1 LTS | RADV, Mesa 26.0.8 | MIGraphX 2.18.0, ROCm 10.1.0 | 2900 / 2800 MHz (`high`) | `f2ce0ed` |

| Comparison | Image + text tokens | Framework | Latency (ms) | Min (ms) | Max abs error |
|---|---:|---|---:|---:|---:|
| `migraphx` | 512 + 154 | **vAi** | **739.85 ± 4.43** | 732.20 | 0.0119 |
| | | MIGraphX | 896.73 ± 11.17 | 888.84 | 0.0119 |

Each row is one `scripts/run-amd.sh -- --atol 0.2 --rtol 0.1` run. The checkpoint is [`lerobot/pi05_libero_finetuned_v044`](https://huggingface.co/lerobot/pi05_libero_finetuned_v044), the LIBERO fine-tune of the same architecture, read with the pre- and postprocessor configuration of `lerobot/pi05_libero_base`; its maximum absolute errors hold for that checkpoint, not for the rows above.

### Precision configuration

The frameworks use FP16 model weights but differ in intermediate activation storage, accumulation, and residual precision. vAi's are set node by node in each net's constructor in [net.h](../../models/experimental/pi05/net.h).

| Component | vAi | vla.cpp / TensorRT Edge-LLM / MIGraphX |
|---|---|---|
| Model weights | FP16 | FP16 |
| K/V cache | FP16 | Framework-specific |
| Block branch activations | FP16 | Framework-specific |
| Vision GEMM accumulation | FP32 | Backend-specific |
| Vision / language residual | FP32 | Backend-specific |
| Action expert residual | FP16 | Backend-specific |
| Final action update | FP32 | Backend-specific |

> **IMPORTANT**
> Results are specific to the tested checkpoint, GPU, framework versions, precision settings, and workload. A lower latency under one workload does not establish higher performance under different camera counts, action chunks, or denoising steps.
