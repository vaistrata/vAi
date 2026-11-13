# Quickstart

Build vAi and run the MNIST classifier example.

The MNIST classifier, from model definition to weight loading and execution, is contained in a single file, [`examples/mnist/main.cpp`](../../examples/mnist/main.cpp). The checkpoint is in [`data/ckpts/mnist/jroot/`](../../data/ckpts/mnist/jroot/), and the input images are in [`data/images/`](../../data/images/).

## 1. Installation

### 1.1 Requirements

| | |
|---|---|
| GPU | Vulkan conformant (NVIDIA / AMD / Intel / Qualcomm) |
| Vulkan SDK | 1.3 or later, including glslc |
| CMake | 3.20 or later |
| Compiler | C++23 (gcc 14 or later) |

### 1.2 Clone

eva (the Vulkan wrapper) is a submodule and must be cloned along with the repository.

```bash
git clone --recursive https://github.com/vaistrata/vAi.git
cd vAi
git submodule update --init --recursive
```

### 1.3 Build

```bash
cmake -S . -B build -DVAI_BUILD_EXAMPLES=ON
cmake --build build --target demo-mnist
```

Build options:

| Option | Default | Description |
|---|---|---|
| `VAI_BUILD_MODEL_DEMO` | `ON` | Model demos (`demo-gpt2`, `demo-yolox`, etc.) |
| `VAI_BUILD_EXAMPLES` | `OFF` | Examples outside the models (`demo-mnist`, `chat-llama3`, `chat-gemma3`) |
| `VAI_BUILD_TESTS` | `OFF` | Kernel unit tests, A/B |
| `VAI_BUILD_BENCH` | `OFF` | Benchmarks against frameworks such as llama.cpp and ncnn |
| `VAI_BUILD_PYTHON` | `OFF` | Python bindings |

### 1.4 Run

```bash
./build/bin/demo-mnist
```

### 1.5 Results

An example output is shown below. Since the input is `0.png`, check that class 0 has the largest value among the 10 logits.

```
=== Output logits ===
data[0] = 23.026661
data[1] = -24.376106
data[2] = -1.872803
data[3] = -15.056368
data[4] = -14.481293
data[5] = -13.030799
data[6] = -0.130337
data[7] = -14.873937
data[8] = 1.507278
data[9] = -2.433280
```
