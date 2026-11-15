# vAi Benchmark Setup

What the benches compare, and how, is in [README.md](README.md). The vAi benches need only the [GPU clock](#gpu-clock); a comparison bench also needs its framework installed and the models converted, in that framework's section:

| Benches | Framework | Installation | Model conversion |
|---|---|---|---|
| [LLM](#llm-benchmarks) | [llama.cpp](#llamacpp) | [setup-llamacpp.sh](#installation) | [GGUF](#model-conversion) |

Each framework's setup script installs a pinned release into a user cache outside the repo (`~/.cache/vai/`), where CMake finds it with no extra flags. It installs nothing system-wide: install the **Prerequisites** in the framework's section first. `--uninstall` removes what it installed, and `--help` lists its paths and options. To use your own build of the pinned release instead, point the variable in the `Found by (CMake)` column of the README's Requirements at it.

The vAi sources need GCC 14+ or Clang 18+. Where the default compiler is older, name one when configuring, e.g. `-DCMAKE_CXX_COMPILER=g++-14`.

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
