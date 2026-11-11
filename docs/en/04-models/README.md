# Supported models

These are models implemented with vAi.

```
models/<model>/
├── model.h                      Model and NodeGroup definitions
├── <distribution-name>-weights.h Distribution checkpoint to graph slot mapping (loadWeights<"distribution-name">)
├── nodes/                        Dedicated model nodes (if any)
└── shaders/                      Kernels for dedicated nodes (if any)

examples/<model>/
├── demo.cpp                      demo-<model> - model inference and inference timing
└── CMakeLists.txt

data/<model>/                     Checkpoints and related data
```

## Weight loading

```cpp
// GPT-2 example
#include "model.h"
#include "openai-weights.h"

gpt2::Model model({ .device = device, .defaultDtype = Dtype::Fp16 },
                  gpt2::ModelConfig::_124M());

loadWeights<"openai">(model);
```

[`vai/engine/weights.h`](../../../vai/engine/weights.h) only declares `loadWeights<Name, Model>`, and each model provides a specialization for each distribution name.

See [Writing custom models](../05-contributing-models.md) for information on writing a new model.

## List

### Vision

| Model | Model size |
|---|---|
| [YOLOv8](yolov8.md) | NANO(n), SMALL(s), MEDIUM(m), LARGE(l), XLARGE(x) |
| [YOLOX](yolox.md) | SMALL(s), MEDIUM(m), LARGE(l), XLARGE(x) |

### LLM

| Model | Supported model sizes |
|---|---|
| [GPT-2](gpt2.md) | 124M, 355M, 774M, 1.5B |
| [Llama 3](llama3.md) | 3.2-1B, 3.2-3B |

### Speech

| Model | Supported model sizes |
|---|---|
| [Whisper](whisper.md) | tiny, base, small, medium, large-v3, large-v3-turbo |

## Aggregate benchmark

### Vision

Measured with vAi, at **640x640, FP16**.

| Model | Metric | 890M (Strix Point) | 8060S (Strix Halo) | RTX 5070 Ti | RX 9070 XT |
|---|---|---:|---:|---:|---:|
| YOLOv8n | Latency (ms) | 3.238 | 0.759 | 0.616 | 0.657 |

### LLM

Measured with vAi, using a **512-token prompt**.

| Model | Metric | 890M (Strix Point) | 8060S (Strix Halo) | UHD Graphics (TGL GT1) | RTX 3060 Laptop | RTX 5070 | RTX 4090 | RTX 5090 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 124M | prefill (t/s) | 10,899 | 50,777 | 1,201 | 61,236 | 154,785 | 269,911 | 308,638 |
|  | decode (t/s) | 130.5 | 500.9 | 94.06 | 743.2 | 1,319 | 1,631 | 1,756 |
| Llama 3 3.2-1B | prefill (t/s) | 1,653 | 4,932 | 123.1 | 8,761 | 25,313 | 51,332 | 56,603 |
|  | decode (t/s) | 16.46 | 81.66 | 9.63 | 115.3 | 213.2 | 311.0 | 455.4 |
| Llama 3 3.2-3B | prefill (t/s) | - | 1,726 | - | - | 9,091 | 17,219 | 21,013 |
|  | decode (t/s) | - | 33.87 | - | - | 87.74 | 128.0 | 194.0 |

Results against other frameworks are available in the [YOLOv8 benchmark summary](yolov8.md#benchmark-summary), [YOLOX benchmark summary](yolox.md#benchmark-summary), [GPT-2 benchmark summary](gpt2.md#benchmark-summary), and [Llama 3 benchmark summary](llama3.md#benchmark-summary). See [`benchmarks/`](../../../benchmarks/) for the measurement environment and conditions, and reproduction instructions.
