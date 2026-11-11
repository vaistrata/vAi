# vAi Benchmarks

Measures vAi against other inference frameworks under the same conditions.
Every comparison framework is optional (CMake leaves out a bench whose framework it cannot find; the vAi bench is always built).

## Setup

```
benchmarks/
├── README.md
├── <model>/
│   ├── README.md
│   ├── CMakeLists.txt
│   ├── common/
│   ├── vai/
│   └── <framework>/
└── scripts/
    ├── setup-llamacpp.sh
    └── gpu-clock.sh
```

Every bench reads the same `data/<model>/` the demos do, so getting a checkpoint in place is covered by the model documentation and by the weights header under `models/<model>/`, not here.

### Building and running

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVAI_BUILD_BENCH=ON
cmake --build build --config Release
```

Running a bench, and its results: see the model's own README, linked from each section below.

### Measurement environment

Lock the GPU clock, for measurements that hold still.

**Linux**: `gpu-clock.sh` covers both vendors.

```bash
benchmarks/scripts/gpu-clock.sh status        # the current clock and the lockable range
benchmarks/scripts/gpu-clock.sh lock <MHz>    # NVIDIA: pin the core to <MHz> + memory to its top level
                                              #   pick <MHz> within the range status prints, and pick the
                                              #   clock it sustains under load, not the boost peak
benchmarks/scripts/gpu-clock.sh lock          # or AMD: pin DPM to its top state
# ... run the benches ...
benchmarks/scripts/gpu-clock.sh reset         # back to the default boost / auto
```

*On a machine with both vendors, `GPU_VENDOR=nvidia|amdgpu` picks the one to lock (amdgpu by default).*

**Windows**: no script. `nvidia-smi` ships with the driver and takes the same flags, from an elevated PowerShell.

```powershell
nvidia-smi -L                                              # the GPU index
nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv   # the current clock
nvidia-smi --query-supported-clocks=gr,mem --format=csv    # what can be locked

nvidia-smi -i <gpu idx> -lgc <MHz>,<MHz>    # pin the core, at a clock it sustains under load
nvidia-smi -i <gpu idx> -lmc <MHz>,<MHz>    # pin memory to its top supported level
# ... run the benches ...
nvidia-smi -i <gpu idx> -rgc                # back to the default boost
nvidia-smi -i <gpu idx> -rmc
```

Persistence mode (`-pm 1`), which the Linux script sets, is not supported under WDDM and is left out.

AMD has no command-line equivalent on Windows. Fix the clock in AMD Software: Adrenalin Edition under Performance → Tuning, or record the run as auto-clocked like the [YOLOv8 results](yolov8/README.md#results).

> [!WARNING]
> WSL cannot be benched: Vulkan runs through a D3D12 translation layer (Mesa dzn), so it is not native performance. Use a native Windows build.

## Vision benchmarks

- Benches, with their results and how to run them: [YOLOv8](yolov8/README.md) · [YOLOX](yolox/README.md)
- Comparison frameworks: [PyTorch](https://pytorch.org) · [ncnn](https://github.com/Tencent/ncnn) · [ExecuTorch](https://github.com/pytorch/executorch) · [MIGraphX](https://github.com/ROCm/AMDMIGraphX) · [TensorRT](https://developer.nvidia.com/tensorrt) · [ONNX Runtime](https://onnxruntime.ai)

### Model files

Every bench reads the same checkpoint, each in the format its own runtime takes. The file names are the model's own; see its README.

| | Format (`data/<model>/`) | Found by |
|---|---|---|
| vAi | `.safetensors` | always built |
| PyTorch | `.pt` | Python only, no CMake |
| ncnn | `.param` + `.bin` | `ncnn_DIR` |
| ExecuTorch | `.pte` | `executorch_DIR` |
| MIGraphX | `.onnx` | `migraphx_DIR` (Linux only) |
| TensorRT | `.engine` | `TENSORRT_ROOT` · `CUDAToolkit_ROOT` |
| ONNX Runtime | `.onnx` | `ONNXRUNTIME_ROOT` |

### Shared conditions: vision frameworks

Whatever the runtime, the vision benches share one input and one measurement protocol.

- Input: one image, preprocessed into a tensor before the timing starts and reused for every run, so no preprocessing lands inside the measurement.
- Warmup: 1,000 runs.
- Measurement: 100 runs of the inference stretch, from the prepared input tensor to the model's raw output tensor. Tables report the **mean in ms**.

Batch size, input resolution and output shape are the model's own; its README records them alongside the results.

### Installing the comparison frameworks

> [!NOTE]
> Only needed when building or running that comparison bench.

Where a framework is already installed, skip the steps below and name it at configure time with the variable from the [model files](#model-files) table. When recording results, note the version you actually used alongside the hardware in that model's results table.

<details>
<summary><kbd>PyTorch</kbd></summary>

The PyTorch bench runs straight from Python. Pick the CUDA or ROCm install command on [pytorch.org](https://pytorch.org/get-started/locally/). On Windows, official PyTorch GPU builds are CUDA only.

**Linux**

```bash
python3 -m venv ~/venv/vai-bench
source ~/venv/vai-bench/bin/activate

# run the command shown for Linux · Pip · CUDA or ROCm on pytorch.org
pip install "ultralytics<=8.4.113" pillow safetensors onnx onnxruntime
```

**Windows**

```powershell
python -m venv "$HOME\venv\vai-bench"
& "$HOME\venv\vai-bench\Scripts\Activate.ps1"

# run the command shown for Windows · Pip · CUDA in the official selector
pip install "ultralytics<=8.4.113" pillow safetensors onnx onnxruntime
```

For YOLOX benchmarks and model export, install YOLOX v0.3.0 in the same environment following the [official installation guide](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.3.0#quick-start).

</details>

<details>
<summary><kbd>ncnn</kbd></summary>

Take the package for your OS from [ncnn releases](https://github.com/Tencent/ncnn/releases). Unpack it and point `ncnn_DIR` at the directory holding `ncnnConfig.cmake`.

**Linux** — pick the package matching your Ubuntu version.

```bash
mkdir -p ~/ncnn
curl -L -o ~/ncnn/ncnn-20260113.zip \
  https://github.com/Tencent/ncnn/releases/download/20260113/ncnn-20260113-ubuntu-2404.zip
unzip ~/ncnn/ncnn-20260113.zip -d ~/ncnn

cmake -B build -DCMAKE_BUILD_TYPE=Release -DVAI_BUILD_BENCH=ON \
  -Dncnn_DIR="$HOME/ncnn/ncnn-20260113-ubuntu-2404/lib/cmake/ncnn"
```

**Windows**

```powershell
$NcnnArchive = "$env:TEMP\ncnn-20260113.zip"
$NcnnRoot = "$HOME\ncnn\ncnn-20260113-windows-vs2022\x64"
Invoke-WebRequest `
  https://github.com/Tencent/ncnn/releases/download/20260113/ncnn-20260113-windows-vs2022.zip `
  -OutFile $NcnnArchive
Expand-Archive $NcnnArchive -DestinationPath "$HOME\ncnn" -Force

cmake -B build -DVAI_BUILD_BENCH=ON `
  "-Dncnn_DIR=$NcnnRoot\lib\cmake\ncnn"
```

</details>

<details>
<summary><kbd>ExecuTorch</kbd></summary>

ExecuTorch is installed from source with the Vulkan backend and the C++ extensions the bench uses. The Python environment that converts the model and the C++ runtime come from the same source checkout. After installing, point `executorch_DIR` at the directory holding `executorch-config.cmake`. For the full requirements see the [ExecuTorch source build docs](https://docs.pytorch.org/executorch/stable/using-executorch-building-from-source.html).

**Linux**

```bash
mkdir -p ~/executorch
git clone --depth 1 --recursive \
  https://github.com/pytorch/executorch.git ~/executorch/source
cd ~/executorch/source

python3 -m venv ~/venv/executorch
source ~/venv/executorch/bin/activate
./install_executorch.sh

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/executorch/install" \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_PTHREADPOOL=ON \
  -DEXECUTORCH_BUILD_CPUINFO=ON
cmake --build build --parallel
cmake --install build

cmake -S /path/to/vAi -B /path/to/vAi/build \
  -DCMAKE_BUILD_TYPE=Release -DVAI_BUILD_BENCH=ON \
  -Dexecutorch_DIR="$HOME/executorch/install/lib/cmake/ExecuTorch"
```

**Windows** — allow symbolic links in Developer Mode and run from a Visual Studio 2022 Developer PowerShell.

```powershell
git config --global core.symlinks true
New-Item -ItemType Directory -Force "$HOME\executorch" | Out-Null
git clone --depth 1 --recursive `
  https://github.com/pytorch/executorch.git `
  "$HOME\executorch\source"
Set-Location "$HOME\executorch\source"

python -m venv "$HOME\venv\executorch"
& "$HOME\venv\executorch\Scripts\Activate.ps1"
.\install_executorch.bat

cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -T ClangCL `
  "-DEXECUTORCH_BUILD_PRESET_FILE=$PWD\tools\cmake\preset\windows.cmake" `
  "-DCMAKE_INSTALL_PREFIX=$HOME\executorch\install" `
  -DEXECUTORCH_BUILD_VULKAN=ON
cmake --build build --config Release
cmake --install build --config Release

cmake -S C:\path\to\vAi -B C:\path\to\vAi\build `
  -DVAI_BUILD_BENCH=ON `
  "-Dexecutorch_DIR=$HOME\executorch\install\lib\cmake\ExecuTorch"
```

</details>

<details>
<summary><kbd>MIGraphX</kbd></summary>

MIGraphX is a ROCm framework. Install ROCm first, then the runtime and development packages, and `find_package(migraphx CONFIG)` finds it under `/opt/rocm`. The prebuilt packages are used as they come, so MIGraphX itself is never built here. For details see the [MIGraphX install docs](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/develop/install/install-migraphx.html). The official packages do not support Windows, so that bench is not built there.

```bash
sudo apt update
sudo apt install -y migraphx migraphx-dev
```

Where ROCm sits somewhere other than its default location, name that prefix in `migraphx_DIR` or `CMAKE_PREFIX_PATH`.

</details>

<details>
<summary><kbd>TensorRT</kbd></summary>

Install a Debian, tar or zip package carrying the TensorRT C++ headers and runtime; see the [TensorRT install docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html). `trtexec`, which builds the `.engine`, and the C++ TensorRT the bench links against must be the same version.

**Linux** — from the [NVIDIA TensorRT download page](https://developer.nvidia.com/tensorrt/download), take the local repository package for your OS and CUDA version, register it, then install.

```bash
sudo apt update
sudo apt install tensorrt
```

An APT install is found without any extra path. With a tar package, name `TENSORRT_ROOT` and `CUDAToolkit_ROOT` at configure time.

**Windows** — download the TensorRT zip for your CUDA version and unpack it wherever you like. Add `<TensorRT>/bin` to `PATH` and name the install location at configure time.

```powershell
$TensorRTRoot = "C:\path\to\TensorRT"
$env:PATH = "$TensorRTRoot\bin;$env:PATH"
cmake -B build -DVAI_BUILD_BENCH=ON `
  "-DTENSORRT_ROOT=$TensorRTRoot" `
  "-DCUDAToolkit_ROOT=$env:CUDA_PATH"
```

</details>

<details>
<summary><kbd>ONNX Runtime</kbd></summary>

For CUDA, use the official C/C++ GPU archive from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases) and point `ONNXRUNTIME_ROOT` at the unpacked directory. The MIGraphX EP, for AMD GPUs, needs a source build on Linux. For the official procedures see the [ONNX Runtime C++ docs](https://onnxruntime.ai/docs/get-started/with-cpp.html) and the [MIGraphX EP build docs](https://onnxruntime.ai/docs/build/eps.html).

**Linux** — CUDA package:

```bash
mkdir -p ~/onnxruntime/cuda
tar -xf /path/to/onnxruntime-linux-x64-gpu.tgz \
  -C ~/onnxruntime/cuda --strip-components=1

cmake -B build -DCMAKE_BUILD_TYPE=Release -DVAI_BUILD_BENCH=ON \
  -DONNXRUNTIME_ROOT="$HOME/onnxruntime/cuda"
```

MIGraphX EP package — set `ONNXRUNTIME_ROOT` to `$HOME/onnxruntime/migraphx` when using it:

```bash
mkdir -p ~/onnxruntime
git clone --depth 1 --recursive \
  https://github.com/microsoft/onnxruntime.git ~/onnxruntime/source
cd ~/onnxruntime/source
./build.sh --config Release --build_shared_lib --parallel --skip_tests \
  --use_migraphx --migraphx_home /opt/rocm --target install \
  --cmake_extra_defines CMAKE_INSTALL_PREFIX="$HOME/onnxruntime/migraphx"
```

**Windows** — the Windows bench uses the CUDA package. Unpack the contents of the official zip into `$HOME\onnxruntime\cuda`.

```powershell
$OrtRoot = "$HOME\onnxruntime\cuda"
$env:PATH = "$OrtRoot\lib;$env:PATH"

cmake -B build -DVAI_BUILD_BENCH=ON "-DONNXRUNTIME_ROOT=$OrtRoot"
```

</details>

### Model conversion

*Used to build the per-runtime model files the comparison benches need.*

Every file comes from one source checkpoint. The PyTorch bench and the ONNX / ExecuTorch conversion scripts read that checkpoint directly, and vAi uses the `.safetensors` extracted from it. ONNX Runtime and MIGraphX share one ONNX, which TensorRT also consumes. The YOLOX ncnn exporter reads the checkpoint and creates its own temporary ONNX with Focus folded into the first convolution.

| framework | conversion path |
|---|---|
| PyTorch | the checkpoint, used directly |
| vAi | → `.safetensors` |
| ONNX Runtime · MIGraphX | → `.onnx` |
| ncnn | → `.fp32.onnx` → `.param` + `.bin` |
| ExecuTorch | → `.pte` |
| TensorRT | → `.{precision}.onnx` → `.engine` |

The converters live at `benchmarks/<model>/<framework>/export.py`, and the vAi one under `models/<model>/`. Three of them want a tool on `PATH`: ncnn needs `pnnx`, ExecuTorch needs `vulkaninfo`, and an FP16 ONNX needs the Python `onnxruntime` package.

The commands for a given model are in its own README: [YOLOv8](yolov8/README.md#model-preparation) · [YOLOX](yolox/README.md#model-preparation).

## LLM benchmarks

- Benches, with their results and how to run them: [GPT-2](gpt2/README.md) · [Llama 3](llama3/README.md) · [Gemma 3](gemma3/README.md)
- Comparison framework: [llama.cpp](https://github.com/ggml-org/llama.cpp)

### Model files

| | Files (`data/<model>/`) | Found by |
|---|---|---|
| vAi | `tokenizer.json` + `<size>/model.safetensors` | always built |
| llama.cpp | `<size>/model-f16.gguf` ([GGUF Conversion](#gguf-conversion)) | [Installing llama.cpp](#installing-llamacpp) |

### Shared conditions: vAi vs llama.cpp

Whatever the model, the LLM benches share one code skeleton and one measurement protocol.

- Model size: chosen with `--size <name>`. The name is the checkpoint folder name under `data/<model>/` as is, so the vAi bench and the llama.cpp bench take the same value, and both print `model: <model> <size>` above the table. Omitted, GPT-2 defaults to `124M`, Llama 3 to `3.2-1B` and Gemma 3 to `4B`.
- Prompts: the prompts defined in `common/bench-prompts.h`.
- Warmup: one global warmup at the longest length, then per length 3 prefills and an 8-token decode.
- Measurement: per length, 15 prefills and five 128-token decodes, timed separately (the prefill opening a decode excluded). Tables report throughput in tokens per second, as **median ± SEM**.
- Vulkan kernel path: llama.cpp keeps its own kernel selection, NVIDIA-only `NV_coopmat2` where the driver exposes it, otherwise `KHR_cooperative_matrix`. vAi only uses `KHR_cooperative_matrix`, so on NVIDIA the two run different kernels.
- Prefill execution unit: vAi processes the whole prompt in a single forward pass, so llama.cpp is matched with `n_ubatch = n_ctx`.
- Logits extent: both benches sample the last token only, and the vAi graph slices off the last row before the lm head, so llama.cpp is matched with `n_outputs_max = 1`.

### Installing llama.cpp

> [!NOTE]
> Only needed when building the llama.cpp comparison bench.

llama.cpp is installed into a user cache outside the repo and registered in the CMake package registry, so `find_package(llama)` finds it with no extra flags.

<details>
<summary><kbd>Linux</kbd></summary>

```bash
./benchmarks/scripts/setup-llamacpp.sh               # clone + static Vulkan build + install + register, ~2-3 min
./benchmarks/scripts/setup-llamacpp.sh --uninstall   # remove the install, the source and the registration
```

Paths, prerequisites and the other options: `--help`, or the header of [setup-llamacpp.sh](scripts/setup-llamacpp.sh).

</details>

<details>
<summary><kbd>Windows</kbd></summary>

```powershell
# 0. Install location
$LLAMA = "$env:LOCALAPPDATA\vai\llama.cpp-b10426"

# 1. Clone the pinned tag
git clone --depth 1 --branch b10426 https://github.com/ggml-org/llama.cpp "$LLAMA-src"

# 2. configure — Vulkan backend, static libraries, libraries only
#    LLAMA_BUILD_NUMBER cannot be omitted: a shallow clone carries no tag info, so it comes out as 0
cmake -S "$LLAMA-src" -B "$LLAMA-src\build" `
    -DGGML_VULKAN=ON `
    -DBUILD_SHARED_LIBS=OFF `
    -DLLAMA_BUILD_TESTS=OFF `
    -DLLAMA_BUILD_EXAMPLES=OFF `
    -DLLAMA_BUILD_TOOLS=OFF `
    -DLLAMA_BUILD_SERVER=OFF `
    -DLLAMA_BUILD_APP=OFF `
    -DLLAMA_CURL=OFF `
    -DLLAMA_BUILD_NUMBER=10426 `
    -DCMAKE_INSTALL_PREFIX="$LLAMA"

# 3. Build + install
cmake --build   "$LLAMA-src\build" --config Release
cmake --install "$LLAMA-src\build" --config Release

# 4. Register in the CMake user package registry — so find_package(llama) finds it with no flags
New-Item -Path HKCU:\Software\Kitware\CMake\Packages\llama -Force | Out-Null
New-ItemProperty -Path HKCU:\Software\Kitware\CMake\Packages\llama `
    -Name vai-b10426 -Value "$LLAMA\lib\cmake\llama" -Force | Out-Null

# 5. Keep the GGUF conversion code next to the install
New-Item -ItemType Directory -Force "$LLAMA\convert" | Out-Null
Copy-Item "$LLAMA-src\convert_hf_to_gguf.py", "$LLAMA-src\LICENSE" "$LLAMA\convert\"
Copy-Item -Recurse "$LLAMA-src\conversion" "$LLAMA\convert\conversion"
Copy-Item -Recurse "$LLAMA-src\gguf-py" "$LLAMA\convert\gguf-py"
Copy-Item "$LLAMA-src\requirements\requirements-convert_hf_to_gguf.txt", `
          "$LLAMA-src\requirements\requirements-convert_legacy_llama.txt" "$LLAMA\convert\"

# 6. Clean up the source tree (optional, ~600 MB)
Remove-Item -Recurse -Force "$LLAMA-src"
```

</details>

### GGUF conversion

> [!NOTE]
> Only needed when running the llama.cpp comparison bench (`bench-<model>-llamacpp`).

The llama.cpp bench uses `data/<model>/<size>/model-f16.gguf`.

- Conversion: llama.cpp's official `convert_hf_to_gguf.py`
- Where the conversion code lives: [Installing llama.cpp](#installing-llamacpp) puts it next to the install (`convert_hf_to_gguf.py` + `conversion/` + `gguf-py/` + two requirements files + LICENSE, all MIT).

**1. Setup**: shared across models, once.

```bash
CONVERT=~/.cache/vai/llama.cpp-b10426/convert                  # Windows: $env:LOCALAPPDATA\vai\llama.cpp-b10426\convert
pip install -r $CONVERT/requirements-convert_hf_to_gguf.txt
```

**2. Convert**:
- Input: the `data/<model>/<size>/` folder (with `config.json` and the tokenizer files alongside, see [GPT-2](gpt2/README.md#gguf-conversion) · [Llama 3](llama3/README.md#gguf-conversion) · [Gemma 3](gemma3/README.md#gguf-conversion))

```bash
PYTHONPATH=$CONVERT python $CONVERT/convert_hf_to_gguf.py data/<model>/<size> --outtype f16 --outfile data/<model>/<size>/model-f16.gguf
```

## Diffusion benchmarks

- Benches, with their results and how to run them: [Z-Image](zimage/README.md)
- Comparison framework: [PyTorch](https://pytorch.org) through [diffusers](https://github.com/huggingface/diffusers)

### Shared conditions: vAi vs diffusers

- Measurement: one DiT forward, i.e. one denoising step, on the same latent, caption features and timestep for both. The caption encoder and the VAE are outside it.
- Timing: the median of 10 forwards after warmup, from building the host inputs to reading the output back (vAi), or CUDA-synchronized (PyTorch).
- Accuracy: both sides are checked against one fp32 diffusers forward of the same inputs, which `torch/bench.py` stores and the vAi bench reads.
