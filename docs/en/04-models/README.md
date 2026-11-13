# Supported models

These models are implemented in vAi.

```
models/<model>/
├── model.h                      Model and NodeGroup definitions
├── <distribution>-weights.h     Distribution checkpoint -> graph slot mapping (loadWeights<"distribution">)
├── nodes/                       Model-specific nodes (if any)
└── shaders/                     Kernels for model-specific nodes (if any)

examples/<model>/
├── demo.cpp                     demo-<model> - model inference and timing
└── CMakeLists.txt

data/<model>/                    Checkpoints and related data
```

## Loading weights

```cpp
// GPT-2 example
#include "net.h"
#include "openai-weights.h"

namespace gpt2 = model::production::gpt2;

gpt2::Net net({ .device = device, .defaultDtype = Dtype::Fp16 },
              gpt2::ModelConfig::_124M());

loadWeights<"openai">(net);
```

[`vai/engine/weights.h`](../../../vai/engine/weights.h) only declares `loadWeights<Name, Model>`; each model provides specializations for its distributions.

For new models, see [Writing custom models](../05-contributing-models.md).

## Model list

### Vision

| Model | Model sizes |
|---|---|
| [YOLOv8](yolov8.md) | NANO(n), SMALL(s), MEDIUM(m), LARGE(l), XLARGE(x) |
| [YOLOX](yolox.md) | SMALL(s), MEDIUM(m), LARGE(l), XLARGE(x) |

### LLM

| Model | Supported model sizes |
|---|---|
| [GPT-2](gpt2.md) | 124M, 355M, 774M, 1.5B |
| [Llama 3](llama3.md) | 3.2-1B, 3.2-3B |

## Combined benchmarks

### Vision

vAi measurements at **640x640, FP16**.

| Model | Metric | 890M (Strix Point) | 8060S (Strix Halo) | RTX 5070 Ti | RX 9070 XT |
|---|---|---:|---:|---:|---:|
| YOLOv8n | Latency (ms) | 3.238 | 0.759 | 0.616 | 0.657 |

### LLM

vAi measurements with a **512-token prompt**.

| Model | Metric | 890M (Strix Point) | 8060S (Strix Halo) | UHD Graphics (TGL GT1) | RTX 3060 Laptop | RTX 5070 | RTX 4090 | RTX 5090 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 124M | prefill (t/s) | 10,899 | 50,777 | 1,201 | 61,236 | 154,785 | 269,911 | 308,638 |
|  | decode (t/s) | 130.5 | 500.9 | 94.06 | 743.2 | 1,319 | 1,631 | 1,756 |
| Llama 3 3.2-1B | prefill (t/s) | 1,653 | 4,932 | 123.1 | 8,761 | 25,313 | 51,332 | 56,603 |
|  | decode (t/s) | 16.46 | 81.66 | 9.63 | 115.3 | 213.2 | 311.0 | 455.4 |
| Llama 3 3.2-3B | prefill (t/s) | - | 1,726 | - | - | 9,091 | 17,219 | 21,013 |
|  | decode (t/s) | - | 33.87 | - | - | 87.74 | 128.0 | 194.0 |

Comparisons with other frameworks are in the benchmark summaries for [YOLOv8](yolov8.md#benchmark-summary), [YOLOX](yolox.md#benchmark-summary), [GPT-2](gpt2.md#benchmark-summary), and [Llama 3](llama3.md#benchmark-summary). For measurement environments, conditions, and reproduction instructions, see [`benchmarks/`](../../../benchmarks/).
