# YOLOX Benchmarks

## Summary

Runs YOLOX under vAI and six other runtimes on the same GPU and compares inference latency.

- Checkpoint: one official Megvii YOLOX checkpoint for every runtime. PyTorch reads the `.pth`; the other runtimes use files [converted](#model-preparation) from it.
- Conditions: batch 1, 640x640, `data/images/dog.jpg` letterboxed into the input tensor, and the timed stretch ends at a decoded `[1, 8400, 85]` output. Warmup and repetition counts follow the [shared conditions](../README.md#shared-conditions-vision-frameworks).

| framework | file (`data/benchmarks/yolox/`) |
|---|---|
| vAI | `yolox_<size>.safetensors` |
| PyTorch | `yolox_<size>.pth` |
| ncnn | `yolox_<size>.param/.bin` |
| ExecuTorch | `yolox_<size>.{precision}.pte` |
| MIGraphX | `yolox_<size>.{precision}.onnx` |
| TensorRT | `yolox_<size>.fp32.tf32_on.engine`, `yolox_<size>.fp32.tf32_off.engine`, `yolox_<size>.fp16.engine` |
| ONNX Runtime | `yolox_<size>.{precision}.onnx` |

## Model preparation

Begin once the YOLOX checkpoint (`data/benchmarks/yolox/yolox_l.pth`) is in place. Convert it along the routes in [Model conversion](../README.md#model-conversion). ncnn needs `pnnx` on `PATH`, ExecuTorch needs `vulkaninfo` on `PATH`, and FP16 ONNX export needs the Python `onnxruntime` package. 

Run them from the repository root; they build the Large FP32 and FP16 files into `data/benchmarks/yolox/` as `yolox_l.*`. Swap `l` for another model size (`s` `m` `x`) to measure a different size. The vAi benchmark needs only the `.safetensors` file. The other conversions are for the comparison benchmarks.

Export scripts accept multiple model sizes with `--sizes s l` to convert Small and Large in one invocation.

**vAI**: the `.pth` into `.safetensors`, shared by FP32 and FP16

```bash
python models/experimental/yolox/megvii-weights-convertor.py l
```

**ONNX Runtime · MIGraphX**: the `.pth` into one `.onnx` per precision

```bash
python benchmarks/yolox/onnxruntime/export.py --sizes l --precision fp32
python benchmarks/yolox/onnxruntime/export.py --sizes l --precision fp16
```

**ncnn**: the `.pth` into `.param` + `.bin`, shared by FP32 and FP16

```bash
python -m pip install pnnx
python benchmarks/yolox/ncnn/export.py --sizes l
```

pnnx currently cannot convert the original YOLOX Focus because it does not support slicing with a step of 2.

**ExecuTorch**: the `.pth` into one `.pte` per precision, built for one GPU; `--gpu` takes the index shown by `vulkaninfo --summary` lists 

```bash
python benchmarks/yolox/executorch/export.py --sizes l --precision fp32 --gpu 0
python benchmarks/yolox/executorch/export.py --sizes l --precision fp16 --gpu 0
```

**TensorRT**: each ONNX into an `.engine`

```bash
trtexec --onnx=data/benchmarks/yolox/yolox_l.fp32.onnx --noTF32 --skipInference \
  --saveEngine=data/benchmarks/yolox/yolox_l.fp32.tf32_off.engine
trtexec --onnx=data/benchmarks/yolox/yolox_l.fp16.onnx --fp16 --skipInference \
  --saveEngine=data/benchmarks/yolox/yolox_l.fp16.engine
```

## Building and running

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVAI_BUILD_BENCH=ON
cmake --build build --config Release

build/bin/bench-yolox-vai
python benchmarks/yolox/pytorch/bench.py     # no build needed, runs from the venv
build/bin/bench-yolox-ncnn
build/bin/bench-yolox-executorch
build/bin/bench-yolox-migraphx
build/bin/bench-yolox-tensorrt
build/bin/bench-yolox-onnxruntime
```

A bench whose runtime CMake cannot find is [left out of the build](../README.md#model-files).

*With no arguments a bench takes `data/images/dog.jpg` as input and runs YOLOX Large · FP32 on GPU 0. PyTorch and ONNX Runtime use the CUDA backend by default.*

Options:

- `--size s|m|l|x`: model size (default: `l`)
- `--precision fp32|fp16`: model precision (default: `fp32`)
- `--warmup N`, `--iterations N`: warmup and timed run counts (default: `1000`, `100`)
- `--image PATH`: input image (default: `data/images/dog.jpg`)
- `--gpu N`: GPU index (default: `0`)
- `--backend cuda|rocm`: backend for PyTorch and ONNX Runtime (default: `cuda`). The PyTorch bench also takes `cpu`, at FP32 only
- `--IO fp32|fp16`: vAI input/output precision (default: `fp32`)
- `--tf32 on|off`: whether TensorRT's FP32 run uses TF32 (default: `off`)
- `--help`: the runtime's full option list

## Results

### Environment

**System**

| | 890M (Strix Point) | 8060S (Strix Halo) | RTX 5070 Ti | RX 9070 XT |
|---|---|---|---|---|
| OS | Ubuntu 24.04.4 | Ubuntu 24.04.4 | Ubuntu 24.04 | Ubuntu 24.04 |
| Driver | Mesa RADV 25.2.8 | Mesa RADV 25.2.8 | NVIDIA 595.84 | Mesa RADV 25.2.8 |
| Vulkan device API | 1.4.318 | 1.4.318 | 1.4.329 | 1.4.318 |
| Compute stack | ROCm 7.2.4 | ROCm 7.2.1 | CUDA 12.8 | ROCm 7.2.3 |
| GPU clock | auto | auto | auto | auto |

**Framework versions**

| framework | 890M (Strix Point) | 8060S (Strix Halo) | RTX 5070 Ti | RX 9070 XT |
|---|---|---|---|---|
| PyTorch | 2.9.1+rocm7.2.4 | 2.9.1+rocm7.2.1 | 2.11.0 | 2.13.0.dev20260512 |
| ncnn | 20260526 | 20260113 | 20260526 | 20260526 |
| ExecuTorch | 1.3.1 | 1.4.1 | 1.4.1 | 1.4.1 |
| MIGraphX | 2.15.0 | 2.15.0 | - | 2.15.0 |
| TensorRT | - | - | 10.16.1 | - |
| ONNX Runtime | 1.24.2 (MIGraphX EP) | 1.24.2 (MIGraphX EP) | 1.24.2 (CUDA EP) | 1.30.0 (MIGraphX EP) |

### Inference latency

ncnn assumes wave32 on AMD devices, so its compute shaders were forced to wave32 with `RADV_PERFTEST=cswave32`.

**YOLOX Large FP16 · 640x640 · mean latency (ms)**

| framework | 890M (Strix Point) | 8060S (Strix Halo) | RTX 5070 Ti | RX 9070 XT |
|---|---:|---:|---:|---:|
| vAI (Vulkan, IO FP16) | 26.152 | 7.78 | 2.693 | 3.311 |
| PyTorch (CUDA, ROCm) | 48.590 | 13.129 | 4.703 | 9.035 |
| ncnn (Vulkan) | 62.752 | 13.610 | 4.779 | 6.028 |
| ExecuTorch (Vulkan) | 205.498 | 57.401 | 27.599 | 34.941 |
| MIGraphX (HIP) | 27.417 | 7.720 | - | 3.367 |
| TensorRT (CUDA) | - | - | 1.838 | - |
| ONNX Runtime (CUDA, MIGraphX EP) | 27.703 | 7.833 | 16.670 | 3.404 |

**YOLOX Large FP32 · 640x640 · mean latency (ms)**

| framework | 890M (Strix Point) | 8060S (Strix Halo) | RTX 5070 Ti | RX 9070 XT |
|---|---:|---:|---:|---:|
| vAI (Vulkan) | 74.160 | 24.843 | 8.288 | 9.092 |
| PyTorch (CUDA, ROCm) | 118.224 | 21.636 | 22.071 | 12.525 |
| ncnn (Vulkan) | 144.477 | 45.987 | 10.283 | 35.211 |
| ExecuTorch (Vulkan) | 239.758 | 67.040 | 37.901 | 41.218 |
| MIGraphX (HIP) | 70.346 | 26.828 | - | 12.547 |
| TensorRT (CUDA, TF32 off) | - | - | 6.048 | - |
| ONNX Runtime (CUDA, MIGraphX EP) | 70.280 | 27.035 | 22.290 | 12.826 |

**YOLOX Small FP16 · 640x640 · mean latency (ms)**

| framework | 890M (Strix Point) | 8060S (Strix Halo) | RTX 5070 Ti | RX 9070 XT |
|---|---:|---:|---:|---:|
| vAI (Vulkan, IO FP16) | 7.700 | 1.923 | 0.952 | 1.105 |
| PyTorch (CUDA, ROCm) | 13.516 | 6.782 | 2.259 | 5.270 |
| ncnn (Vulkan) | 18.892 | 3.500 | 2.408 | 2.298 |
| ExecuTorch (Vulkan) | 59.335 | 15.064 | 8.386 | 10.981 |
| MIGraphX (HIP) | 7.412 | 1.703 | - | 1.115 |
| TensorRT (CUDA) | - | - | 0.546 | - |
| ONNX Runtime (CUDA, MIGraphX EP) | 7.594 | 1.708 | 4.610 | 1.179 |

**YOLOX Small FP32 · 640x640 · mean latency (ms)**

| framework | 890M (Strix Point) | 8060S (Strix Halo) | RTX 5070 Ti | RX 9070 XT |
|---|---:|---:|---:|---:|
| vAI (Vulkan) | 17.989 | 5.158 | 2.264 | 2.612 |
| PyTorch (CUDA, ROCm) | 29.985 | 9.006 | 5.035 | 5.258 |
| ncnn (Vulkan) | 38.195 | 8.568 | 3.696 | 8.811 |
| ExecuTorch (Vulkan) | 68.934 | 17.297 | 9.465 | 12.028 |
| MIGraphX (HIP) | 18.422 | 5.758 | - | 3.013 |
| TensorRT (CUDA, TF32 off) | - | - | 1.548 | - |
| ONNX Runtime (CUDA, MIGraphX EP) | 18.467 | 5.923 | 5.168 | 3.073 |

`-` marks a result that was not measured or a device that the framework does not support.
