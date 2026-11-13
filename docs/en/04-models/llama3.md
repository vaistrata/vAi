# Llama 3

Llama 3 (Grattafiori et al., 2024) is a decoder-only transformer language model developed by Meta. It stacks pre-norm blocks with GQA self-attention and SwiGLU feed-forward layers. It uses RMSNorm for normalization and RoPE for positional information, with no bias. The model predicts the next token by computing a probability distribution over a vocabulary of 128,256 tokens.

**References**: [Paper](https://arxiv.org/abs/2407.21783), [Checkpoint (HuggingFace)](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), [Repository](https://github.com/meta-llama/llama-models)

## Configuration

```cpp
namespace llama3 = model::production::llama3;

auto modelCfg = llama3::ModelConfig::_3_2_1B();

llama3::Net net({
    .device       = device,
    .defaultDtype = Dtype::Fp16,
    .ioDtypeHint  = Dtype::Fp32,
}, modelCfg, {
    .maxRuntimeContext = 4096,
});
```

### NeuralNetConfig

Llama 3 uses the defaults, except that each `RmsNorm` forces `wDtype = fp32`.
For details, see [NeuralNet](../03-usage/neural-net.md).

### ModelConfig

| Factory | `name` | `embedDim` | `ffnHiddenDim` | `numHeads` | `numKVHeads` | `numLayers` |
|---|---|---|---|---|---|---|
| `_3_2_1B()` | `"3.2-1B"` | 2048 | 8192 | 32 | 8 | 16 |
| `_3_2_3B()` | `"3.2-3B"` | 3072 | 8192 | 24 | 8 | 28 |

Both sizes use `vocabSize` 128,256 and `maxTrainedContext` 131,072. They also share the same `ropeConfig` value (`ropeCommon(32.0f)`). The `factor` of 32 is the RoPE frequency scaling factor for the Llama 3 family.

### RuntimeConfig

This only determines the amount of KV cache to allocate. If omitted, it uses `maxTrainedContext`; specifying a value allocates only that amount to save memory. Allocating the full trained context of 131,072 requires too much memory, so the demo and benchmarks reduce it to 4,096 (the same value as `n_ctx` in the llama.cpp benchmark).

| Field | Default |
|---|---|
| `maxRuntimeContext` | `0` |

## vAi graph

In vAi, a model is a DAG built by connecting nodes with `-`. For details, see [Node](../03-usage/node.md) and [NodeGroup](../03-usage/node-group.md).

*`Tq`: number of input tokens, `D`: embedding dimension, `H`: number of query heads, `Hkv`: number of KV heads.*

### Model architecture

```cpp
input("inTokenIds") - embedding - layers[0];
for (size_t i = 1; i < layers.size(); ++i)
    layers[i - 1] - layers[i];
layers.back() - lastRowSlice - finalNorm - lmHead - topkSampling - output("nextTokenId");
```

### NodeGroup

These [node groups](../03-usage/node-group.md) are defined for the Llama 3 implementation.
`TransformerLayer` contains nested `MultiHeadAttention` and `GatedFeedForward` groups.

#### TransformerLayer

```cpp
input - norm1 - "in0"/mhAttn      - norm2 - "in0"/ffn;
input         - "residual"/mhAttn         - "residual"/ffn;
```

The final projection in each sublayer accepts the residual through `acceptResidual` and adds it within the kernel, without a separate `Add`.

#### MultiHeadAttention

With GQA, K and V are projected into fewer heads than the queries (`H == 32`, `Hkv == 8`). RoPE applies only to Q and K; V passes through unchanged.

```cpp
input - projQ - splitQ - perQ - ropeQ - "Q"/attn - mergeOut - projOut;
input - projK - splitK - perK - ropeK - "K"/attn;
input - projV - splitV - perV         - "V"/attn;
```

With `acceptResidual = true`, `toOut` receives the block's residual through `residual` and adds it to the projection result.

`Rope` shares the KV cache token counter, so rotation uses the cumulative absolute position even during decode steps, rather than the offset within the current forward pass.

#### GatedFeedForward

Since `gate` and `value` read the same input and differ only in their weights, a single `Linear` computes both at twice the width. `GatedMul` applies SiLU to the first half and multiplies it by the second half.

```cpp
gateValue - gatedMul - out;   // 2048 -> 16384 -> 8192 -> 2048
```

`out` also sets `acceptResidual = true` and receives the block's residual through `residual`.

### Nodes used

[`TokenEmbedding`](../06-nodes.md#tokenembedding),
[`RmsNorm`](../06-nodes.md#rmsnorm),
[`Linear`](../06-nodes.md#linear),
[`Reshape`](../06-nodes.md#reshape),
[`Permute`](../06-nodes.md#permute),
[`Rope`](../06-nodes.md#rope),
[`Attention`](../06-nodes.md#attention),
[`GatedMul`](../06-nodes.md#gatedmul),
[`Identity`](../06-nodes.md#identity),
[`Slice`](../06-nodes.md#slice),
[`TopKSampling`](../06-nodes.md#topksampling)

## Checkpoints

Place checkpoints in size-specific folders (`ModelConfig::variant`), with the tokenizer in their parent folder.
Both sizes share `tokenizer.json`.

```
data/
├── ckpts/llama3/meta/
│   ├── 3.2-1B/
│   │   └── model.safetensors
│   ├── 3.2-3B/
│   │   └── model.safetensors
│   └── ...
└── tokenizers/llama3/
    └── tokenizer.json
```

Download the files from the corresponding HuggingFace repository and use them without conversion. The weights are subject to the [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE).

```bash
mkdir -p data/ckpts/llama3/meta/3.2-1B data/tokenizers/llama3
curl -L -o data/tokenizers/llama3/tokenizer.json https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/tokenizer.json
curl -L -o data/ckpts/llama3/meta/3.2-1B/model.safetensors \
  https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/model.safetensors
```

Download `model.safetensors` and `tokenizer.json` directly from the HuggingFace repository's **Files and versions** tab or use the commands above.

> The original `meta-llama/*` repositories are gated and require approval and a token. The commands above use an [unsloth mirror](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct) containing the same tensors. The same license applies to mirror downloads. **An Instruct checkpoint trained on chat tokens is required**.

### Loading weights

The code that maps checkpoint tensors to graph slots is in [`models/llama3/meta-weights.h`](../../../models/llama3/meta-weights.h).
Loading takes one line at the call site. The specialization constructs the path (`VAI_DATA_DIR "/llama3/" + modelCfg.variant`), so only the size needs to be selected. For distribution tags, see [`distribution-weights.h` in Writing custom models](../05-contributing-models.md#distribution-weightsh).

```cpp
loadWeights<"meta">(net);
```

## Running

### Directory structure

```
models/llama3/
├── model.h                       Model and NodeGroup definitions
└── meta-weights.h                Meta (HuggingFace `meta-llama/*`) checkpoint mapping

examples/llama3/
├── demo.cpp                      demo-llama3, complete one prompt and measure timing
├── chat.cpp                      chat-llama3, multi-turn conversation
└── CMakeLists.txt

examples/chat-llama3/
├── main.cpp                      chat-llama3: multi-turn conversation
└── CMakeLists.txt

data/
├── ckpts/llama3/meta/
│   └── <variant>/                    ModelConfig::variant (3.2-1B, 3.2-3B)
│       └── model.safetensors
└── tokenizers/llama3/
    └── tokenizer.json
```

### Build and run

```bash
cmake -B build -DVAI_BUILD_EXAMPLES=ON
cmake --build build --config RelWithDebInfo
```

### demo-llama3

```bash
./build/bin/demo-llama3
```

The demo generates text that continues the text in `prompt` in [`demo.cpp`](../../../examples/llama3/demo.cpp). It measures `prefill` and `decode` times separately and prints them with the generated text.

### chat-llama3

```bash
./build/bin/chat-llama3
```

This runs a multi-turn conversation from [`chat.cpp`](../../../examples/llama3/chat.cpp). Conversation history stays in the KV cache, and the session ends on an empty line or when the cache is full.

## Benchmark summary

These results compare llama.cpp (Vulkan) and vAi with a **512-token prompt**. For full results by GPU and prompt length, measurement environments and conditions, and reproduction instructions, see [`benchmarks/llama3/README.md`](../../../benchmarks/llama3/README.md).

**Llama 3.2-1B**

| | | 890M (Strix Point) | 8060S (Strix Halo) | UHD Graphics (TGL GT1) | RTX 3060 Laptop | RTX 5070 | RTX 4090 | RTX 5090 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vAi | prefill (t/s) | 1,653 | 4,932 | 123.1 | 8,761 | 25,313 | 51,332 | 56,603 |
|  | decode (t/s) | 16.46 | 81.66 | 9.63 | 115.3 | 213.2 | 311.0 | 455.4 |
| llama.cpp | prefill (t/s) | 1,290 | 2,975 | 122.3 | 10,020 | 29,671 | 50,166 | 66,053 |
|  | decode (t/s) | 16.09 | 66.83 | 9.39 | 110.5 | 221.5 | 290.6 | 424.0 |

**Llama 3.2-3B**

| | | 8060S (Strix Halo) | RTX 5070 | RTX 4090 | RTX 5090 |
|---|---|---:|---:|---:|---:|
| vAi | prefill (t/s) | 1,726 | 9,091 | 17,219 | 21,013 |
|  | decode (t/s) | 33.87 | 87.74 | 128.0 | 194.0 |
| llama.cpp | prefill (t/s) | 1,609 | 12,644 | 22,849 | 27,118 |
|  | decode (t/s) | 32.63 | 88.81 | 124.8 | 185.7 |
