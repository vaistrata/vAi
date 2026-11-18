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
| MIGraphX | `prefill.fp16.onnx` and `step.fp16.onnx`, each with its `.weights.bin`, and `config.txt`: FP16 graphs (weights, activations, the K/V cache between them) with FP32 images, timestep and actions, their constants folded at export |
| TensorRT Edge-LLM | `engines/` |
| every bench, vAi included | `reference_<cameras>_<chunk>_<steps>.bin`, `noise_…` and `images_…`, for the accuracy check |

Each comparison runs one workload from [`common/workload.h`](common/workload.h), the same on both sides: the LIBERO checkpoint (`libero`), its cameras, its action chunk and its denoise steps. TensorRT Edge-LLM follows openpi's `pi05_libero`, as Edge-LLM's pi0.5 guide builds it; MIGraphX and vla.cpp follow the LeRobot checkpoint's own policy:

| workload | compared with | policy | cameras | chunk | steps | reference |
|---|---|---|---:|---:|---:|---|
| `migraphx` | MIGraphX | LeRobot: the discretized state in the prompt | 2 | 50 | 10 | `reference.py --cameras 2 --chunk 50 --steps 10` |
| `trt` | TensorRT Edge-LLM | openpi `pi05_libero`: the task alone | 2 | 10 | 10 | `reference.py --cameras 2 --chunk 10 --steps 10 --policy openpi` |
| `vlacpp` | vla.cpp | LeRobot: the discretized state in the prompt | 2 | 50 | 10 | `reference.py --cameras 2 --chunk 50 --steps 10` |

The reference files come from [`scripts/reference.py`](scripts/reference.py): LeRobot's FP32 PyTorch policy on the observation in [`common/bench-inputs.h`](../common/bench-inputs.h). The noise it starts from and the pixels its model saw, after LeRobot's preprocessing, are written with it, and every bench takes both: the comparison covers the model, not the image preprocessing. Without them a bench draws its own noise and preprocesses the images itself. The converted models keep the shapes they were built with: a change to a workload needs new ones.

The conversions also need these files in the checkpoint folder, from the same repository: `config.json` for all, the two policy files for the vla.cpp converter:

```bash
curl -fL -o data/ckpts/pi05/lerobot/<size>/config.json https://huggingface.co/<repository>/resolve/main/config.json
curl -fL -o data/ckpts/pi05/lerobot/<size>/policy_preprocessor.json https://huggingface.co/<repository>/resolve/main/policy_preprocessor.json
curl -fL -o data/ckpts/pi05/lerobot/<size>/policy_postprocessor.json https://huggingface.co/<repository>/resolve/main/policy_postprocessor.json
```

> [!NOTE]
> The checkpoint ships no dataset statistics, which the vla.cpp converter needs (`--dataset-stats <stats.json>`: `mean`, `std`, `q01` and `q99` of width 32 for `observation.state` and `action`). vla.cpp applies them to the actions it returns and zeroes the dims past the real action width. The vla.cpp bench reads them back from the GGUF and inverts them outside the timed call, then compares only the real dims with the reference: a GGUF converted with any statistics, such as the published [`vrfai/pi05-libero-gguf`](https://huggingface.co/vrfai/pi05-libero-gguf) with LIBERO's, checks against the model's own actions.

## Building and running

Build as in [benchmarks/README.md](../README.md#building-and-running), then, on an NVIDIA GPU:

```bash
build/bin/bench-pi05-vai --workload vlacpp
build/bin/bench-pi05-vlacpp
build/bin/bench-pi05-vai --workload trt
build/bin/bench-pi05-trt
```

and on an AMD GPU:

```bash
build/bin/bench-pi05-vai --workload migraphx
build/bin/bench-pi05-migraphx
```

*A bench whose prediction does not match its workload's reference prints no latency; without the file it reads `accuracy: unverified`.*

Options:

- vAi bench: `--workload <name>`, the workload of the framework compared with: `migraphx` (default), `trt` or `vlacpp`. The comparison benches always run their own. vAi runs FP16 weights and K/V cache; each block keeps its residual stream FP32 and writes its branch (norms, Q/K/V, attention, FFN hidden) FP16, and the vision tower accumulates its GEMMs in FP32 (`TowerPrecision` in [net.h](../../models/experimental/pi05/net.h)).
- `--showActions`: print the first five actions after the table
- `--atol N`, `--rtol N`: tolerances against the workload's reference (default: `0.002`, `0.001`)
- vla.cpp bench: `--model <path>` (or `VLA_MODEL_PATH`), another GGUF; `--config <path>` (or `VLA_CONFIG_PATH`), another runtime config than [`vla.cpp/runtime.json`](vla.cpp/runtime.json), which sets FP16 weights
- MIGraphX bench: `[bundle-dir]`, another export folder. The graphs run as exported, in FP16. Its GEMM backend is set at configure time: `-DMIGRAPHX_GEMM_PROVIDER=mlir` (default) fuses the dots into MLIR kernels (`MIGRAPHX_DISABLE_MLIR=0`), `hipblaslt` lowers every dot to hipBLASLt with MLIR off (`MIGRAPHX_DISABLE_MLIR=1`); either way a dot left outside MLIR goes to hipBLASLt (`MIGRAPHX_SET_GEMM_PROVIDER=hipblaslt`), and a variable set in the environment wins. MLIR runs fastest on a bundle with its constants folded, as `export-onnx.py` exports by default; `hipblaslt` on one exported with `--no-constant-folding`. The bench prints which it read.
- TensorRT Edge-LLM bench: `--engineDir <path>` (or `PI05_ENGINE_DIR`), other engines; `EDGELLM_PLUGIN_PATH`, another plugin library

## Results

Each latency cell is milliseconds per prediction, mean ± SD, with the fastest rep beside it, as the bench prints it, from runs that passed the accuracy check; max abs error is the bench's against the workload's reference. The Vulkan driver is its name and version from `vulkaninfo --summary` (in `vulkan-tools` or the Vulkan SDK), and the compute stack the comparison frameworks' CUDA and TensorRT versions on NVIDIA, ROCm version on AMD.

### pi0.5 LIBERO

#### NVIDIA GeForce RTX 5090

| OS | Vulkan driver | Compute stack | GPU clock (core / memory) | vAi |
|---|---|---|---|---|
| Ubuntu 24.04.5 LTS | NVIDIA 595.99.02 | CUDA 13.2, vla.cpp v0.4.0 (built for sm_120) | default, not locked (max 3105 / 14001 MHz) | 99128a4 + precision tuning |

| workload | image + text | framework | latency (ms) | min (ms) | max abs error |
|---|---:|---|---:|---:|---:|
| `vlacpp` | 512 + 154 | vAi | 39.24 ± 0.04 | 39.02 | 0.00124 |
|  |  | vla.cpp | 42.19 ± 0.14 | 42.05 | 0.00142 |
| `trt` | 512 + 17 | vAi | - | - | - |
|  |  | TensorRT Edge-LLM | - | - | - |

- Result: vAi is 2.9 ms (7%) faster than vla.cpp and its max abs error is 13% lower, while it is checked over more values (below).
- Accuracy: both pass at `atol 0.002`, `rtol 0.001`, against the reference, noise and images of `reference.py --cameras 2 --chunk 50 --steps 10`. vAi is checked over all 1600 values ([50, 32]). vla.cpp returns only its 7 real action dims and zeroes the rest, so it is checked over those 350 values.
- vla.cpp runs the published [`vrfai/pi05-libero-gguf`](https://huggingface.co/vrfai/pi05-libero-gguf), which carries LIBERO's action quantiles. The bench inverts them outside the timed call, so the actions compared are the model's own.
- Each vAi row is the median, by mean, of three runs: 39.11 – 39.32 ms, max abs error 0.00124 in all three.
- vla.cpp's other runtime options change neither number: `mm_prec f32` 42.28 ms, `flash_attn` 42.15 ms, both 42.21 ms, each with error 0.00142. BF16 weights fail the check (0.00225).

##### vAi precision tuning (NVIDIA)

On NVIDIA a Linear accumulates in its weight's dtype, FP16, unless told otherwise, and it rounds its input to FP16 on the way into the GEMM whatever dtype that input is stored in. Before tuning, vAi stored every activation FP32 (`oDefaultDtype`) and accumulated every GEMM in FP16: 40.74 ± 0.06 ms, error 0.00225, just inside the tolerance.

The tuning sets two switches per tower, `TowerPrecision` in [net.h](../../models/experimental/pi05/net.h):

- `branchFp16`: inside each block, the branch off the residual stream (the norms, Q/K/V, RoPE, attention, the FFN hidden and its gating) writes FP16. The GEMMs round these values to FP16 anyway, so storing them FP16 costs little precision and halves their memory traffic. The residual stream (the output projections that fold the residual in, the action expert's gated residuals, the final norm) and the velocity head stay FP32, so rounding does not build up over the layers or the 10 Euler steps.
- `accF32`: the tower's Linears accumulate in FP32.

Measured with `branchFp16` on in all three towers, one run each (the chosen setting's three runs are above):

| FP32 accumulation in | latency (ms) | max abs error |
|---|---:|---:|
| none | 38.54 ± 0.04 | 0.00187 |
| **vision** (chosen) | **39.22 ± 0.12** | **0.00124** |
| vision + action | 40.79 ± 2.80 | 0.00117 |
| vision + language | 42.63 ± 0.03 | 0.00152 |
| all | 43.50 ± 0.84 | 0.00137 |

- FP32 accumulation in the vision tower costs 0.7 ms and removes most of the error. The language tower's would cost 3–4 ms and the action expert's 1.5 ms, for no clear gain.
- An FP32 K/V cache, tried before `branchFp16`, cost 0.2 ms and left the error unchanged (0.00225), so it stays FP16.
- These dtypes apply on every vendor. AMD already accumulates in FP32; `branchFp16` there is not yet measured, and the AMD row below predates it.

#### AMD Radeon 8060S (gfx1151)

| OS | Vulkan driver | Compute stack | GPU clock (core / memory) | vAi |
|---|---|---|---|---|
| Ubuntu 24.04.4 LTS | RADV, Mesa 25.2.8 | MIGraphX 2.18.0, ROCm 10.1.0 | 2900 / 1000 MHz (`high`) | e979c71 |

| workload | image + text | framework | latency (ms) | min (ms) | max abs error |
|---|---:|---|---:|---:|---:|
| `migraphx` | 512 + 154 | vAi | 182.45 ± 4.95 | 178.01 | 0.000981 |
|  |  | MIGraphX | 195.01 ± 4.39 | 189.54 | 0.00225 |

- Precision: both store FP16 weights and an FP16 K/V cache, and both feed FP16 operands to their matrix multiplies (attention's P included). vAi keeps every activation FP32 between ops (`oDefaultDtype`) and accumulates its GEMMs in FP32, with norms, softmax, epilogues, RoPE, the condition table and the Euler update in FP32. MIGraphX runs its FP16 bundle, constants folded, through MLIR: activations stay FP16 between ops, and only softmax, norm statistics, RoPE's sin/cos and the Euler update run in FP32. Op by op, for both: [docs/readme.md §9](docs/readme.md#9-정밀도).
- Accuracy: both passed at the tolerances the benches ran with, `atol 0.2`, `rtol 0.1`. At `--atol 0.002 --rtol 0.001`, vAi passes (0 of 1600 values outside) and MIGraphX does not (2 of 1600), so its latency is printed only at the looser tolerances.
- Each row is the median, by mean, of three `scripts/run-amd.sh` runs: vAi 179.03 – 183.27 ms, MIGraphX 193.81 – 196.53 ms.
- The APU shares its power with the CPU, so the clock under load stays below the lock: sclk 2457 – 2790 MHz while vAi ran, 2436 – 2898 MHz while MIGraphX did (its samples include its 84 s graph compile).
