# Supported models

These models are implemented in vAi.

```
models/<impl>/<model>/
├── net.h                        Model and NodeGroup definitions
├── <distribution>-weights.h     Distribution checkpoint -> graph slot mapping (loadWeights<"distribution">)
├── nodes/                       Model-specific nodes (if any)
├── shaders/                     Kernels for model-specific nodes (if any)
├── demo.cpp                     demo-<model> - model inference and timing
└── CMakeLists.txt

data/ckpts/<model>/<distribution>/   Checkpoints
data/tokenizers/<model>/             Tokenizer
data/images/                         Input images
```

## Loading weights

```cpp
// GPT-2 example
#include "net.h"
#include "openai-weights.h"

namespace gpt2 = model::production::gpt2;

gpt2::Net net({ .device = device, .wDefaultDtype = Dtype::Fp16 },
              gpt2::ModelConfig::_124M());

loadWeights<"openai">(net);
```

[`vai/engine/weights.h`](../../../vai/engine/weights.h) only declares `loadWeights<Name, Model>`; each model provides specializations for its distributions.

For new models, see [Writing custom models](../contributing/models.md).

## Model list

### Vision

| Model | Model sizes |
|---|---|
| [YOLOX](yolox.md) | SMALL(s), MEDIUM(m), LARGE(l), XLARGE(x) |

### Speech

| Model | Supported model sizes |
|---|---|
| [Whisper](whisper.md) | tiny, base, small, medium, large-v3, large-v3-turbo |

### LLM / VLM

| Model | Supported model sizes |
|---|---|
| [GPT-2](gpt2.md) | 124M, 355M, 774M, 1.5B |
| [Llama 3](llama3.md) | 3.2-1B, 3.2-3B |
| [Gemma 3](gemma3.md) | 270M, 1B, 4B, 12B (4B and 12B also take images) |
| [PaliGemma](paligemma.md) | 3B-224, 3B-448 (take images) |

### VLA

| Model | Supported model sizes |
|---|---|
| [π0](pi0.md) | 3B |
| [ACT](act.md) | Single configuration (LeRobot defaults) |
