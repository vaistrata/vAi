# pi0.5 Benchmarks

## Summary

Runs pi0.5 under vAi and a comparison framework on the same GPU and compares the latency of one action prediction: vla.cpp and TensorRT Edge-LLM on NVIDIA, MIGraphX on AMD.

- Checkpoint: the same for all, the LIBERO fine-tune of pi0.5 3B; the comparison frameworks read [files converted](#model-preparation) from it.
- Workload: two camera images, a prompt, and an action chunk in 10 denoise steps, as the [configuration](#building-and-running) each comparison runs names it.
- Conditions, and where the frameworks differ: [benchmarks/README.md](../README.md#vla-benchmarks). Their GPU APIs, precision and command submission differ, which may affect latency.

> [!IMPORTANT]
> Every bench checks one prediction against a PyTorch FP32 reference before it times any, and prints no latency when it fails the tolerances.

## Model preparation

Begin once the pi0.5 checkpoint (`data/ckpts/pi05/lerobot/3B/`) and the PaliGemma tokenizer (`data/tokenizers/paligemma/tokenizer.json`) are in place. Which 3B checkpoint goes in the folder listed in [lerobot-weights.h](../../models/production/pi05/lerobot-weights.h) is up to you; the comparisons and the results below use the LIBERO fine-tune, `lerobot/pi05_libero_base`. Download it in full: the reference below loads it through LeRobot, which reads more of the repository than `model.safetensors` and `config.json`. The tokenizer comes from `google/paligemma-3b-pt-224`, whose license must be accepted first, with a Hugging Face token.

```bash
hf download lerobot/pi05_libero_base --local-dir data/ckpts/pi05/lerobot/3B
hf download google/paligemma-3b-pt-224 tokenizer.json --local-dir data/tokenizers/paligemma
```

The benches also read the camera images `data/images/cam_high.jpg`, `cam_left_wrist.jpg` and `cam_right_wrist.jpg`. The vAi bench needs only these files, and the reference below to check its accuracy.

The comparison benches read files converted from the checkpoint, in `data/benchmarks/pi05/3B/`, made as in [SETUP.md](../SETUP.md#vla-benchmarks):

| Framework | Model files |
|---|---|
| vla.cpp | `model.gguf`, downloaded: vla.cpp's [vrfai/pi05-libero-gguf](https://huggingface.co/vrfai/pi05-libero-gguf) |
| TensorRT Edge-LLM | `engines/`, from [`scripts/export-trt.sh`](scripts/export-trt.sh) |
| MIGraphX | `prefill.fp16.onnx`, `step.fp16.onnx`, their `.weights.bin`, `metadata.json`, `config.txt`, from [`scripts/export-migraphx.sh`](scripts/export-migraphx.sh) |

Every bench checks its prediction against the actions [reference.py](scripts/reference.py) writes there, with the noise they start from and the images the model saw. Run it once per configuration, with the configuration's values, in the environment `export-migraphx.sh` creates:

```bash
~/.cache/vai/venv-pi05-export/bin/python benchmarks/pi05/scripts/reference.py --cameras 2 --chunk 50 --steps 10                   # vlacpp, migraphx
~/.cache/vai/venv-pi05-export/bin/python benchmarks/pi05/scripts/reference.py --cameras 2 --chunk 10 --steps 10 --policy openpi   # trt
```

## Building and running

Build as in [benchmarks/README.md](../README.md#building-and-running), then, on NVIDIA:

```bash
build/bin/bench-pi05-vai --comparison vlacpp
build/bin/bench-pi05-vlacpp

build/bin/bench-pi05-vai --comparison trt
build/bin/bench-pi05-trt
```

and on AMD:

```bash
build/bin/bench-pi05-vai --comparison migraphx
build/bin/bench-pi05-migraphx
```

[`scripts/run-amd.sh`](scripts/run-amd.sh) runs the two AMD benches in turn with the GPU clock pinned to its highest DPM level, restores the level afterwards, and prints the clock and power each one ran at. Arguments after `--` go to both, as in `scripts/run-amd.sh -- --atol 0.2 --rtol 0.1`.

Each comparison runs one `VlaConfig` from [`common/vla-config.h`](common/vla-config.h), the same on both sides. A comparison bench runs its own; the vAi bench takes it from `--comparison`:

| Comparison | Framework | `VlaConfig` | Policy | Cameras | Action chunk | Denoise steps |
|---|---|---|---|---:|---:|---:|
| `vlacpp` | vla.cpp | `lerobotLibero` | LeRobot | 2 | 50 | 10 |
| `trt` | TensorRT Edge-LLM | `openpiLibero` | openpi `pi05_libero` | 2 | 10 | 10 |
| `migraphx` | MIGraphX | `lerobotLibero` | LeRobot | 2 | 50 | 10 |

*LeRobot's policy puts the discretized robot state in the prompt; openpi's, which TensorRT Edge-LLM serves, gives the task alone. With the action chunk, that makes `trt` another workload: compare frameworks only within one.*

Options:

- `--comparison <name>` (vAi bench): the framework compared with, and so the configuration: `migraphx` (default), `vlacpp` or `trt`.
- `--showActions`: print the first five actions after the table.
- `--atol N`, `--rtol N`: the accuracy tolerances (defaults `0.002` and `0.001`).
- `--model <path>`, `--config <path>` (vla.cpp bench): another GGUF and runtime configuration.
- `--engineDir <path>` (TensorRT Edge-LLM bench): another engine folder.
- `[bundle-dir]` (MIGraphX bench): another ONNX bundle.

*vla.cpp returns its actions un-normalized with the GGUF's LIBERO statistics; its bench normalizes them again outside the timed prediction, to compare with the reference.*

## Results

Each latency cell is **mean ± SD in milliseconds per prediction**, as printed by the benchmark. `min` is the fastest repetition. Lower latency is better.

The maximum absolute error is calculated against the corresponding PyTorch FP32 reference.

vAi runs the experimental pi0.5 with the per-tower precision under [Precision configuration](#precision-configuration), as [#46](https://github.com/vaistrata/vAi/pull/46) sets it in [net.h](../../models/experimental/pi05/net.h). The results were measured with those dtypes, set then through per-tower switches; the benches have changed since only in their configuration names, paths and scripts, not in the model, its precision or the inputs.

### pi0.5 LIBERO

#### NVIDIA GeForce RTX 5090

| OS | Vulkan driver | Compute stack | GPU clock (core / memory) |
|---|---|---|---|
| Ubuntu 24.04.5 LTS | NVIDIA 595.99.02 | CUDA 13.2, TensorRT 10.16.1, TensorRT Edge-LLM 0.11.1, vla.cpp v0.4.0 | Default |

| Comparison | Image + text tokens | Framework | Latency (ms) | Min (ms) | Max abs error |
|---|---:|---|---:|---:|---:|
| `vlacpp` | 512 + 154 | **vAi** | **39.19 ± 0.03** | 39.14 | 0.00112 |
| | | vla.cpp | 42.19 ± 0.14 | 42.05 | 0.00142 |
| `trt` | 512 + 17 | vAi | 33.87 ± 0.06 | 33.74 | 0.000755 |
| | | **TensorRT Edge-LLM** | **21.10 ± 0.01** | 21.07 | 0.00417 |

`trt`'s TensorRT Edge-LLM row is measured with `--atol 0.2 --rtol 0.1`; the other rows pass the default tolerances.

#### NVIDIA GeForce RTX 4090

| OS | Vulkan driver | Compute stack | GPU clock (core / memory) |
|---|---|---|---|
| Ubuntu 24.04.5 LTS | NVIDIA 615.71.09 | CUDA 13.4, TensorRT 10.16.1, TensorRT Edge-LLM 0.11.1, vla.cpp v0.4.0 | Default |

| Comparison | Image + text tokens | Framework | Latency (ms) | Min (ms) | Max abs error |
|---|---:|---|---:|---:|---:|
| `vlacpp` | 512 + 154 | **vAi** | **52.20 ± 0.33** | 51.81 | 0.0019 |
| | | vla.cpp | 55.55 ± 0.04 | 55.47 | 0.00284 |
| `trt` | 512 + 17 | vAi | 45.02 ± 0.52 | 44.16 | 0.00273 |
| | | **TensorRT Edge-LLM** | **32.01 ± 0.39** | 31.85 | 0.00341 |

Each row is the median of three runs. Only `vlacpp`'s vAi row passes the default tolerances; vla.cpp's row is measured with `--atol 0.005`, and both `trt` rows with `--atol 0.2 --rtol 0.1`. vAi's `trt` row, which passes the default tolerances on the RTX 5090, misses them here by one of 320 values.

#### AMD Strix Halo (Radeon 8060S, gfx1151)

| OS | Vulkan driver | Compute stack | GPU clock (core / memory) |
|---|---|---|---|
| Ubuntu 24.04.4 LTS | RADV, Mesa 25.2.8 | MIGraphX 2.18.0, ROCm 10.1.0 | 2900 / 1000 MHz (`high`) |

| Comparison | Image + text tokens | Framework | Latency (ms) | Min (ms) | Max abs error |
|---|---:|---|---:|---:|---:|
| `migraphx` | 512 + 154 | **vAi** | **151.73 ± 3.98** | 148.82 | 0.00116 |
| | | MIGraphX | 195.33 ± 4.17 | 190.27 | 0.00221 |

Each row is the median of three `scripts/run-amd.sh -- --atol 0.2 --rtol 0.1` runs.

#### AMD Strix Point (Radeon 890M, gfx1150)

| OS | Vulkan driver | Compute stack | GPU clock (core / memory) |
|---|---|---|---|
| Ubuntu 26.04.1 LTS | RADV, Mesa 26.0.8 | MIGraphX 2.18.0, ROCm 10.1.0 | 2900 / 2800 MHz (`high`) |

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
