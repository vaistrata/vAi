# vAi Benchmark Setup

What the benches compare, and how, is in [README.md](README.md). The vAi benches need only the [GPU clock](#gpu-clock); a comparison bench also needs its framework installed and the models converted, in that framework's section:

| Benches | Framework | Installation | Model conversion |
|---|---|---|---|
| [LLM](#llm-benchmarks) | [llama.cpp](#llamacpp) | [setup-llamacpp.sh](#installation) | [GGUF](#model-conversion) |
| [VLA](#vla-benchmarks) | [vla.cpp](#vlacpp) | [setup-vlacpp.sh](#vlacpp-installation) | [GGUF](#vlacpp-model-conversion) |
| [VLA](#vla-benchmarks) | [TensorRT Edge-LLM](#tensorrt-edge-llm) | [setup-trt-edge-llm.sh](#tensorrt-edge-llm-installation) | [TensorRT engines](#tensorrt-edge-llm-model-conversion) |
| [VLA](#vla-benchmarks) | [MIGraphX](#migraphx) | [setup-migraphx.sh](#migraphx-installation) | [FP16 ONNX bundle](#migraphx-model-conversion) |

Each framework's setup script installs a pinned release into a user cache outside the repo (`~/.cache/vai/`), where CMake finds it with no extra flags. It installs nothing system-wide: install the **Prerequisites** in the framework's section first. `--uninstall` removes what it installed, and `--help` lists its paths and options. To use your own build of the pinned release instead, point the variable in the `Found by (CMake)` column of the README's Requirements at it.

The vAi sources need GCC 14+ or Clang 18+. Where the default compiler is older, name one when configuring, e.g. `-DCMAKE_CXX_COMPILER=g++-14`.

The shaders need a recent `glslc`. Ubuntu 24.04's (shaderc 2023.8) fails to optimize `linear-cm-amd.comp` ("Cooperative matrix type must be B Type"); the LunarG Vulkan SDK's, or conda-forge's `shaderc` (2026.4 works), builds it. Name it when configuring, e.g. `-DVulkan_GLSLC_EXECUTABLE=<prefix>/bin/glslc`.

> [!WARNING]
> WSL cannot be benched: its Vulkan runs through a D3D12 translation layer (Mesa dzn). Use a native Windows build.

## GPU clock

Lock the GPU clock where possible, for measurements that hold still. Where it cannot be locked, measure without it and record that in the results table.

A lock is an upper bound: power and thermal limits still override it, so on a laptop or APU pick a clock the GPU sustains under load, not the boost peak.

**NVIDIA**: Linux with `sudo`, Windows from an elevated PowerShell.

```bash
nvidia-smi -L                                              # the GPU index
nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv   # the current clock
nvidia-smi --query-supported-clocks=gr,mem --format=csv    # what can be locked

nvidia-smi -i <gpu idx> -lgc <MHz>,<MHz>                   # pin the core, at a clock it sustains under load
nvidia-smi -i <gpu idx> -lmc <MHz>,<MHz>                   # pin memory to its top level: -lgc alone leaves it to the performance state

# ... run the benches ...

# back to the default boost
nvidia-smi -i <gpu idx> -rgc
nvidia-smi -i <gpu idx> -rmc
```

On a headless Linux box, `nvidia-smi -pm 1` keeps the lock between bench processes (not supported under WDDM).

To check that a lock held, compare the Clocks Event Reasons Counters of `nvidia-smi -q -d PERFORMANCE` before and after a run; a grown counter means the clock dropped below the lock.

**AMD, Linux**: `high` pins every clock to the top of its DPM table. An APU shares its power with the CPU, so check the clocks during a run as well.

```bash
ls /sys/class/drm/card*/device/power_dpm_force_performance_level   # the AMD card(s)
echo high | sudo tee /sys/class/drm/card<N>/device/power_dpm_force_performance_level
grep -H '\*' /sys/class/drm/card<N>/device/pp_dpm_*clk             # each clock's active level: the top one

# ... run the benches ...

echo auto | sudo tee /sys/class/drm/card<N>/device/power_dpm_force_performance_level
```

**AMD, Windows**: no command-line equivalent. Fix the clock in AMD Software: Adrenalin Edition under Performance → Tuning, or record the run as auto-clocked in the results table.

## LLM benchmarks

### llama.cpp

#### Installation

**Prerequisites**

- The SPIRV-Headers CMake package, which the Vulkan backend needs (Ubuntu: `spirv-headers`; the LunarG Vulkan SDK carries it)
- A glslc that supports `GL_NV_cooperative_matrix2`, such as the LunarG Vulkan SDK's: llama.cpp builds its NVIDIA coopmat2 kernels only then
- With Clang: its OpenMP runtime (`libomp-dev`, e.g. `libomp-18-dev`), which llama.cpp's CMake package asks for

[setup-llamacpp.sh](scripts/setup-llamacpp.sh) installs llama.cpp `v0.8.0` on its Vulkan backend as static libraries and registers it in the CMake package registry for `find_package(llama)`.

*The pin moves to llama.cpp v1.0.0 in a later update.*

**Linux**

```bash
./benchmarks/scripts/setup-llamacpp.sh               # clone + static Vulkan build + install + register
./benchmarks/scripts/setup-llamacpp.sh --uninstall
```

**Windows**: no script, so the same steps by hand.

<details>
<summary><kbd>Windows</kbd></summary>

```powershell
# PowerShell

# 0. Install location
$LLAMA = "$env:LOCALAPPDATA\vai\llama.cpp-v0.8.0"

# 1. Clone the pinned release
git clone --depth 1 --branch v0.8.0 https://github.com/ggml-org/llama.cpp "$LLAMA-src"

# 2. Configure: Release, Vulkan backend, static libraries, libraries only
#    CMAKE_BUILD_TYPE is for a single-config generator such as Ninja; Visual Studio takes --config Release below.
#    LLAMA_BUILD_NUMBER cannot be omitted: a shallow clone carries no tag info, so it comes out as 0.
#    11960 is the build number of b11960, tagged on the same commit as v0.8.0
cmake -S "$LLAMA-src" -B "$LLAMA-src\build" `
    -DCMAKE_BUILD_TYPE=Release `
    -DGGML_VULKAN=ON `
    -DBUILD_SHARED_LIBS=OFF `
    -DLLAMA_BUILD_TESTS=OFF `
    -DLLAMA_BUILD_EXAMPLES=OFF `
    -DLLAMA_BUILD_TOOLS=OFF `
    -DLLAMA_BUILD_SERVER=OFF `
    -DLLAMA_BUILD_APP=OFF `
    -DLLAMA_BUILD_MTMD=ON `
    -DLLAMA_OPENSSL=OFF `
    -DLLAMA_BUILD_IS_DEV=OFF `
    -DLLAMA_BUILD_NUMBER=11960 `
    -DCMAKE_INSTALL_PREFIX="$LLAMA"

# 3. Build + install
cmake --build   "$LLAMA-src\build" --config Release
cmake --install "$LLAMA-src\build" --config Release

# 4. Register in the CMake user package registry, so find_package(llama) finds it with no flags.
#    Earlier vai-* entries go first: with several, which one find_package picks is undefined.
New-Item -Path HKCU:\Software\Kitware\CMake\Packages\llama -Force | Out-Null
Remove-ItemProperty -Path HKCU:\Software\Kitware\CMake\Packages\llama -Name vai-* -ErrorAction SilentlyContinue
New-ItemProperty -Path HKCU:\Software\Kitware\CMake\Packages\llama `
    -Name vai-v0.8.0 -Value "$LLAMA\lib\cmake\llama" -Force | Out-Null

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

#### Model conversion

Each checkpoint is converted to `data/benchmarks/<model>/<size>/model.gguf` with llama.cpp's official `convert_hf_to_gguf.py`, which the installation above keeps in the install's `convert/` folder.

**1. Setup**: a virtual environment, shared across models, once.

```bash
CONVERT="${XDG_CACHE_HOME:-$HOME/.cache}/vai/llama.cpp-v0.8.0/convert"   # Windows: $env:LOCALAPPDATA\vai\llama.cpp-v0.8.0\convert
python3 -m venv .venv-llamacpp-convert                                   # Windows: its Python is .venv-llamacpp-convert\Scripts\python
.venv-llamacpp-convert/bin/python -m pip install -r $CONVERT/requirements-convert_hf_to_gguf.txt
```

**2. Convert**: once the checkpoint folder holds the files the model's README lists ([GPT-2](gpt2/README.md#model-preparation) · [Llama 3](llama3/README.md#model-preparation) · [Gemma 3](gemma3/README.md#model-preparation)).

```bash
mkdir -p data/benchmarks/<model>/<size>   # Windows: New-Item -ItemType Directory -Force data\benchmarks\<model>\<size>
# Windows: the same command with .venv-llamacpp-convert\Scripts\python and step 1's $CONVERT
.venv-llamacpp-convert/bin/python $CONVERT/convert_hf_to_gguf.py data/ckpts/<model>/<source>/<size> --outtype f16 --outfile data/benchmarks/<model>/<size>/model.gguf
```

## VLA benchmarks

The VLA comparison frameworks run on Linux only, and so do their setup scripts. Each script only builds or unpacks its framework, under `${XDG_CACHE_HOME:-~/.cache}/vai/`, and links the install there, where CMake finds it; what it downloads stays next to the install, so a later run reuses it without the network. The model files come from the model's own export scripts, in `<model>/scripts/`, once the framework is installed.

Each section below gives the framework's official way to make its model files first, then the script in this repo that runs those steps for the bench. The examples use pi0.5's LIBERO checkpoint, `data/ckpts/pi05/lerobot/libero/`; the model's README lists the files it needs ([pi0.5](pi05/README.md#model-preparation)).

### vla.cpp

[vla.cpp](https://github.com/VinRobotics/vla.cpp) runs the pi0.5 benches on CUDA: it has no Vulkan backend for pi0.5.

#### vla.cpp installation

**Prerequisites**

- git, CMake 3.22+ and a C++17 compiler
- CUDA 12 or 13 (`nvcc`; `CUDA_HOME` where it is not on `PATH`)
- Network access on the first run: vla.cpp fetches its llama.cpp when it configures

[setup-vlacpp.sh](scripts/setup-vlacpp.sh) installs vla.cpp `v0.4.0` on its CUDA backend as shared libraries and links it at `~/.cache/vai/vla.cpp`, where CMake reads `vla_ROOT`. It keeps the clone, and the llama.cpp fetched into its `_deps/`, in `~/.cache/vai/vla.cpp-v0.4.0-src`; only the build trees go after the install.

It builds for one CUDA architecture, the SM without its dot (86 RTX 30, 89 RTX 40, 90 H100, 120 RTX 50), as vla.cpp's README passes it in `CMAKE_CUDA_ARCHITECTURES`. The script asks for it, offering the one `nvidia-smi` reports for GPU `GPU_INDEX` (default 0); `--cuda-arch` or `CUDA_ARCH` answers it up front, and without a terminal the GPU's own is taken. A later run with another architecture rebuilds.

```bash
./benchmarks/scripts/setup-vlacpp.sh                  # asks for the architecture; clone + CUDA build + install + link, ~20 min
./benchmarks/scripts/setup-vlacpp.sh --cuda-arch 89   # without asking
./benchmarks/scripts/setup-vlacpp.sh --keep-build     # keep the build trees too, for a faster rebuild
./benchmarks/scripts/setup-vlacpp.sh --uninstall
```

#### vla.cpp model conversion

**Used by the bench**: the LIBERO GGUF vla.cpp's README lists on Hugging Face, [vrfai/pi05-libero-gguf](https://huggingface.co/vrfai/pi05-libero-gguf) (`pi05-libero.gguf`, converted from `lerobot/pi05_libero_base`), downloaded to `data/benchmarks/pi05/libero/model.gguf`. The bench loads it in FP16 through its own runtime config, `pi05/vla.cpp/runtime.json`.

```bash
curl -fL --create-dirs -o data/benchmarks/pi05/libero/model.gguf \
  https://huggingface.co/vrfai/pi05-libero-gguf/resolve/main/pi05-libero.gguf
```

**Official conversion**, for another checkpoint ([docs/MODELS.md](https://github.com/VinRobotics/vla.cpp/blob/v0.4.0/docs/MODELS.md)): the converters ship in the source's `scripts/`, one per architecture, and run in a virtual environment with the source's `convert` extra. A pi0.5 checkpoint carries no normalizer statistics, so `convert_pi05_to_gguf.py` takes them from the dataset's `meta/stats.json`: a local file with `--dataset-stats`, or one downloaded from a Hugging Face dataset with `--dataset-repo`.

```bash
SRC="${XDG_CACHE_HOME:-$HOME/.cache}/vai/vla.cpp-v0.4.0-src"   # the source setup-vlacpp.sh keeps
python3 -m venv .venv-vlacpp-convert
.venv-vlacpp-convert/bin/python -m pip install -e "$SRC[convert]"

mkdir -p data/benchmarks/pi05/libero
.venv-vlacpp-convert/bin/python $SRC/scripts/convert_pi05_to_gguf.py \
    --ckpt data/ckpts/pi05/lerobot/libero \
    --out data/benchmarks/pi05/libero/model.gguf \
    --dataset-repo lerobot/libero
```

**In this repo**: no script; the bench takes the downloaded file above.

### TensorRT Edge-LLM

[TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) runs the pi0.5 benches through its experimental pi0.5 runtime, on engines built on the GPU that runs them.

#### TensorRT Edge-LLM installation

**Prerequisites**

- git, CMake and a CUDA-compatible C++17 compiler
- Python 3 with its venv module (Ubuntu: `python3-venv`)
- CUDA 12 or 13 (`nvcc`; `CUDA_HOME` where it is not on `PATH`)
- A TensorRT C++ SDK built for that CUDA, with its plugins and ONNX parser (`TRT_PACKAGE_DIR` for an extracted SDK, default `/usr`)

From NVIDIA's CUDA apt repository, pin every package to one release, `libnvinfer-safe-headers-dev` included: left unpinned, apt picks the newest release for it, and the install fails on its dependencies. The results were measured with TensorRT 10.16.1:

```bash
V=10.16.1.11-1+cuda13.2
sudo apt-get install libnvinfer-dev=$V libnvinfer-headers-dev=$V libnvinfer-safe-headers-dev=$V libnvinfer10=$V \
    libnvinfer-plugin-dev=$V libnvinfer-headers-plugin-dev=$V libnvinfer-plugin10=$V \
    libnvonnxparsers-dev=$V libnvonnxparsers10=$V
```

[setup-trt-edge-llm.sh](scripts/setup-trt-edge-llm.sh) installs TensorRT Edge-LLM `v0.11.1`: its C++ libraries, the experimental pi0.5 runtime and the policy tools (`pi05_policy_build`), with its CuTe DSL kernels for the SM of GPU `GPU_INDEX` (default 0; `--cuda-arch` or `CUDA_ARCH` names another, and a later run with another SM rebuilds). It links the install at `~/.cache/vai/TensorRT-Edge-LLM`, where CMake reads `TensorRTEdgeLLM_ROOT`, and keeps the clone with its submodules in `~/.cache/vai/TensorRT-Edge-LLM-v0.11.1-src`, from which the export below installs its exporter.

```bash
./benchmarks/scripts/setup-trt-edge-llm.sh                  # clone + CuTe DSL kernels + build + install + link
./benchmarks/scripts/setup-trt-edge-llm.sh --cuda-arch 120  # another SM than GPU 0's
./benchmarks/scripts/setup-trt-edge-llm.sh --keep-build     # keep the build tree too, for a faster rebuild
./benchmarks/scripts/setup-trt-edge-llm.sh --uninstall
```

#### TensorRT Edge-LLM model conversion

**Official** ([installation](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/v0.11.1/docs/source/user_guide/getting_started/installation.md) · [pi0.5](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/v0.11.1/docs/source/user_guide/examples/vla/pi05.md)): the exporter is the source's Python package with its `export` extra, and runs on the CPU without the native build. It exports the checkpoint to FP16 ONNX as an openpi policy configuration (`pi05_libero` for the LIBERO checkpoint); the configuration's normalization statistics go in the export's `assets/`, and `pi05_policy_build` builds the engines from it on this GPU.

```bash
SRC="${XDG_CACHE_HOME:-$HOME/.cache}/vai/TensorRT-Edge-LLM-v0.11.1-src"   # the source setup-trt-edge-llm.sh keeps
EDGELLM="${XDG_CACHE_HOME:-$HOME/.cache}/vai/TensorRT-Edge-LLM"            # the install it links
python3 -m venv .venv-trt-export
.venv-trt-export/bin/python -m pip install -e "$SRC[export]"

ONNX_DIR=data/benchmarks/pi05/libero/trt-onnx
PYTHONNOUSERSITE=1 .venv-trt-export/bin/trtedge export data/ckpts/pi05/lerobot/libero "$ONNX_DIR" \
    --dtype float16 --pi05-policy-config pi05_libero
curl --fail --location --create-dirs --output "$ONNX_DIR/assets/norm_stats.json" \
    https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json
EDGELLM_PLUGIN_PATH="$EDGELLM/lib/libNvInfer_edgellm_plugin.so" LD_LIBRARY_PATH="$EDGELLM/lib" \
    "$EDGELLM/bin/pi05_policy_build" --onnx-dir "$ONNX_DIR" --engine-dir data/benchmarks/pi05/libero/engines
```

**In this repo**: [pi05/scripts/export-trt.sh](pi05/scripts/export-trt.sh) runs these steps. It installs the exporter into `~/.cache/vai/venv-trt-export` once, reuses a finished export, and never overwrites an engine folder: remove the folder to build again.

```bash
./benchmarks/pi05/scripts/export-trt.sh               # --size libero: pi05_libero
./benchmarks/pi05/scripts/export-trt.sh --size 3B     # pi05_aloha, with pi05_base's statistics
```

### MIGraphX

[MIGraphX](https://github.com/ROCm/AMDMIGraphX) runs the pi0.5 benches on AMD GPUs, on ONNX graphs it compiles when the bench starts.

#### MIGraphX installation

**Prerequisites**

- Linux x86_64 with the amdgpu kernel driver of a GPU that ROCm 10.1 supports
- `curl`, `dpkg-deb`, `tar` and `sha256sum`
- A system ROCm, for the HIP headers the bench builds with (`ROCM_PATH`, default `/opt/rocm`)

[setup-migraphx.sh](scripts/setup-migraphx.sh) installs MIGraphX `2.18.0` from AMD's prebuilt files, with no build: its Ubuntu 24.04 packages and, from the ROCm `10.1.0` tarball of the GPU (`--arch`, default the first one), only the runtime MIGraphX loads. It unpacks them without apt or root, links the install at `~/.cache/vai/migraphx`, where CMake reads `migraphx_DIR` ahead of the MIGraphX a system ROCm carries, and keeps the downloads, about 2 GB, in `~/.cache/vai/migraphx-2.18.0-download`. The bench then runs on ROCm 10.1 whichever ROCm the system has.

```bash
./benchmarks/scripts/setup-migraphx.sh                  # download + verify + unpack + link
./benchmarks/scripts/setup-migraphx.sh --arch gfx1151   # name the GPU
./benchmarks/scripts/setup-migraphx.sh --uninstall
```

#### MIGraphX model conversion

**Official** ([MIGraphX](https://github.com/ROCm/AMDMIGraphX) · [ONNX export in PyTorch](https://docs.pytorch.org/docs/stable/onnx.html)): MIGraphX has no model-specific exporter; it parses an ONNX graph and compiles it for the GPU. A PyTorch model reaches it through `torch.onnx.export`.

> [!NOTE]
> Neither MIGraphX nor LeRobot provides a pi0.5 implementation or export for MIGraphX. The ONNX export and the MIGraphX runtime the bench uses are this repo's own: the export below, and the prefill-then-steps loop in `pi05/migraphx/net.h`. The MIGraphX results measure that implementation, not an official one.

**In this repo**: [pi05/migraphx/export-onnx.py](pi05/migraphx/export-onnx.py), written for this bench, exports LeRobot's pi0.5 policy as two static graphs, prefill and one denoise step, checks the split against LeRobot's own `sample_actions` in FP32, and writes them in FP16. [pi05/scripts/export-migraphx.sh](pi05/scripts/export-migraphx.sh) runs it: it creates the Python environment the export checks for (lerobot 0.6.0, transformers 5.5.4, torch 2.8.0 for the CPU, onnx 1.19.0) in `~/.cache/vai/venv-pi05-export` once, finds a C++23 compiler for its input helper (g++ 14 or Clang 18; `CXX` names another), and never overwrites a bundle: remove its files to export again.

```bash
./benchmarks/pi05/scripts/export-migraphx.sh                            # constants folded, for the default MLIR GEMM path
./benchmarks/pi05/scripts/export-migraphx.sh -- --no-constant-folding   # for the hipBLASLt GEMM path
```
