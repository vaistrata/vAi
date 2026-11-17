# pi0.5 Benchmarks

## Summary

Runs pi0.5 under vAi and each comparison framework on that framework's GPU and compares the latency of one prediction: vla.cpp and TensorRT Edge-LLM on NVIDIA, MIGraphX on AMD.

- Checkpoint: the same for all; the comparison frameworks read [files converted from it](#model-preparation).
- Conditions, and where the frameworks differ: [benchmarks/README.md](../README.md#vla-benchmarks). Their GPU API and command submission differ, which may affect the latency.

## Model preparation

Begin once the pi0.5 checkpoint (`data/ckpts/pi05/lerobot/<size>/model.safetensors`) is in place, taken from the size's repository listed in [lerobot-weights.h](../../models/production/pi05/lerobot-weights.h), with the demo's two other inputs: the camera images (`data/images/cam_high.jpg`, `cam_left_wrist.jpg`, `cam_right_wrist.jpg`, from the dataset [demo.cpp](../../models/production/pi05/demo.cpp) names) and the PaliGemma tokenizer. The vAi bench needs only these files.

The checkpoint carries no tokenizer; it comes from `google/paligemma-3b-pt-224`, a gated repository: accept its license on the model page, then pass a Hugging Face token:

```bash
curl -fL --create-dirs -H "Authorization: Bearer $HF_TOKEN" -o data/tokenizers/paligemma/tokenizer.json \
  https://huggingface.co/google/paligemma-3b-pt-224/resolve/main/tokenizer.json
```

The comparison benches read files converted from the checkpoint as in [SETUP.md](../SETUP.md#vla-benchmarks), in `data/benchmarks/pi05/<size>/`:

| bench | files |
|---|---|
| vla.cpp | `model.gguf` |
| MIGraphX | `prefill.fp32.onnx` and `step.fp32.onnx`, each with its `.weights.bin`, and `config.txt` |
| TensorRT Edge-LLM | `engines/` |
| every bench, vAi included | `reference.bin`, for the accuracy check |

`reference.bin` comes from the [ONNX export](../SETUP.md#model-conversion-2), so run it on NVIDIA too. It is the PyTorch output for the input every bench builds from [`common/bench-inputs.h`](../common/bench-inputs.h), and the graphs keep that input's shapes: a change to the input needs a new export.

The conversions also need these files in the checkpoint folder, from the same repository: `config.json` for all, the two policy files for the vla.cpp converter:

```bash
curl -fL -o data/ckpts/pi05/lerobot/<size>/config.json https://huggingface.co/<repository>/resolve/main/config.json
curl -fL -o data/ckpts/pi05/lerobot/<size>/policy_preprocessor.json https://huggingface.co/<repository>/resolve/main/policy_preprocessor.json
curl -fL -o data/ckpts/pi05/lerobot/<size>/policy_postprocessor.json https://huggingface.co/<repository>/resolve/main/policy_postprocessor.json
```

> [!WARNING]
> The checkpoint ships no dataset statistics, which the vla.cpp converter needs (`--dataset-stats <stats.json>`: `mean`, `std`, `q01` and `q99` of width 32 for `observation.state` and `action`). vla.cpp applies them to the actions it returns, so pass neutral ones: `q01` of -1 and `q99` of 1 leave the actions as the model gives them, as the accuracy check needs.

## Building and running

Build as in [benchmarks/README.md](../README.md#building-and-running), then, on an NVIDIA GPU:

```bash
build/bin/bench-pi05-vai
build/bin/bench-pi05-vlacpp
build/bin/bench-pi05-trt
```

and on an AMD GPU:

```bash
build/bin/bench-pi05-vai
build/bin/bench-pi05-migraphx
```

*A bench whose prediction does not match `reference.bin` prints no latency; without the file it reads `accuracy: unverified`.*

Options:

- `--size <name>`: which checkpoint to measure. The name is the folder under `data/ckpts/pi05/lerobot/` and `data/benchmarks/pi05/`: `3B` (default). Pass the same name to every bench.
- `--showActions`: print the first five actions after the table
- `--atol N`, `--rtol N`: tolerances against `reference.bin` (default: `0.002`, `0.001`)
- vla.cpp bench: `--model <path>` (or `VLA_MODEL_PATH`), another GGUF; `--config <path>` (or `VLA_CONFIG_PATH`), another runtime config than [`vla.cpp/runtime.json`](vla.cpp/runtime.json), which sets FP16 weights
- MIGraphX bench: `--precision fp16|fp32`, FP16 dot and convolution over the FP32 graphs (default) or FP32 throughout; `[bundle-dir]`, another export folder
- TensorRT Edge-LLM bench: `--engineDir <path>` (or `PI05_ENGINE_DIR`), other engines; `EDGELLM_PLUGIN_PATH`, another plugin library

## Results

Each cell is milliseconds per prediction, as the bench prints it, from runs that passed the accuracy check; the Vulkan driver is its name and version from `vulkaninfo --summary` (in `vulkan-tools` or the Vulkan SDK), and the compute stack the comparison frameworks' CUDA and TensorRT versions on NVIDIA, ROCm version on AMD.

### pi0.5 3B

#### {NVIDIA GPU name}

| OS | Vulkan driver | Compute stack | GPU clock (core / memory) | vAi |
|---|---|---|---|---|
| {OS} | {driver} | CUDA {version}, TensorRT {version} | {core} / {memory} | {commit} |

| framework | min (ms) | mean (ms) | p50 (ms) | p90 (ms) |
|---|---:|---:|---:|---:|
| vAi | - | - | - | - |
| vla.cpp | - | - | - | - |
| TensorRT Edge-LLM | - | - | - | - |

#### {AMD GPU name}

| OS | Vulkan driver | Compute stack | GPU clock (core / memory) | vAi |
|---|---|---|---|---|
| {OS} | {driver} | ROCm {version} | {core} / {memory} | {commit} |

| framework | min (ms) | mean (ms) | p50 (ms) | p90 (ms) |
|---|---:|---:|---:|---:|
| vAi | - | - | - | - |
| MIGraphX | - | - | - | - |
