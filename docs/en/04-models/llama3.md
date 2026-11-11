# Llama 3

Llama 3 (Grattafiori et al., 2024) is a decoder-only Transformer-based language model developed by Meta. Its architecture stacks pre-norm layers with GQA self-attention, SwiGLU, and feed-forward layers. It uses RMSNorm for normalization and RoPE for positional information, with no bias terms. The model predicts the next token by computing a probability distribution over a vocabulary of 128,256 tokens.

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

Llama 3 uses the defaults as they are; the loader reads every `RmsNorm`'s scale at fp32.
See [NeuralNet](../03-usage/neural-net.md) for details.

### ModelConfig

| Factory | `name` | `embedDim` | `ffnHiddenDim` | `numHeads` | `numKVHeads` | `numLayers` |
|---|---|---|---|---|---|---|
| `_3_2_1B()` | `"3.2-1B"` | 2048 | 8192 | 32 | 8 | 16 |
| `_3_2_3B()` | `"3.2-3B"` | 3072 | 8192 | 24 | 8 | 28 |

`vocabSize` 128,256 and `maxTrainedContext` 131,072 are shared by both sizes. Both sizes also use the same value for `ropeConfig` (`ropeCommon(32.0f)`). The `factor` value of 32 is the RoPE frequency scaling factor for the Llama 3 family.

### RuntimeConfig

This only determines how much KV cache to reserve. If omitted, it uses the `maxTrainedContext` value as is; if a value is specified, it allocates only that amount to save memory. Reserving the full training context of 131,072 would require too much memory, so the demos and benchmarks reduce it to 4,096 (the same value as `n_ctx` in the llama.cpp benchmark).

| Field | Default |
|---|---|
| `maxRuntimeContext` | `0` |

## vAi graph

In vAi, the model is a DAG formed by connecting nodes with `-`. See the [Node](../03-usage/node.md) and [NodeGroup](../03-usage/node-group.md) documentation for details.

*`Tq`: number of input tokens, `D`: embedding dimension, `H`: number of query heads, `Hkv`: number of KV heads.*

### Model architecture

```cpp
input("inTokenIds") - embedding - layers[0];
for (size_t i = 1; i < layers.size(); ++i)
    layers[i - 1] - layers[i];
layers.back() - lastRowSlice - finalNorm - lmHead - topkSampling - output("nextTokenId");
```

### NodeGroup

These are the [NodeGroups](../03-usage/node-group.md) defined to implement Llama 3.
`TransformerLayer` has a nested structure containing `MultiHeadAttention` and `GatedFeedForward`.

#### TransformerLayer

```cpp
input - norm1 - "in0"/mhAttn - norm2 - "in0"/ffn;
input         - "residual"/mhAttn         - "residual"/ffn;
```

Instead of using a separate `Add`, the final projection of each sublayer accepts the residual through `acceptResidual` and adds it inside the kernel.

#### MultiHeadAttention

Because GQA is used, K and V are projected to fewer heads than the queries (`H == 32`, `Hkv == 8`). RoPE is applied only to Q and K, while V passes through unchanged.

```cpp
input - projQ - splitQ - perQ - ropeQ - "Q"/attn - mergeOut - projOut;
input - projK - splitK - perK - ropeK - "K"/attn;
input - projV - splitV - perV         - "V"/attn;
```

Because `projOut` has `acceptResidual = true`, it accepts the block residual as `residual` and adds it to the projection result.

`Rope` shares the KV cache token counter, so even during a decode step, it rotates using the accumulated absolute position rather than the offset within the current forward.

#### GatedFeedForward

`gate` and `value` read the same input and differ only in their weights, so a single `Linear` computes both at once with twice the width. `GatedMul` applies SiLU to the first half and multiplies it by the second half.

```cpp
gateValue - gatedMul - out;   // 2048 -> 16384 -> 8192 -> 2048
```

`out` also has `acceptResidual = true` and accepts the block residual as `residual`.

### Nodes used

[`TokenEmbedding`](../06-nodes.md#tokenembedding) ·
[`RmsNorm`](../06-nodes.md#rmsnorm) ·
[`Linear`](../06-nodes.md#linear) ·
[`Reshape`](../06-nodes.md#reshape) ·
[`Permute`](../06-nodes.md#permute) ·
[`Rope`](../06-nodes.md#rope) ·
[`Attention`](../06-nodes.md#attention) ·
[`GatedMul`](../06-nodes.md#gatedmul) ·
[`Identity`](../06-nodes.md#identity) ·
[`Slice`](../06-nodes.md#slice) ·
[`TopKSampling`](../06-nodes.md#topksampling)

## Checkpoint

Checkpoints are placed in a folder for each size (`ModelConfig::variant`) under `data/ckpts/llama3/meta/`, and the tokenizer in `data/tokenizers/llama3/`.
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

Download the files from the corresponding HuggingFace repository and use them as is without a separate conversion. The weights are licensed under the [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE).

```bash
mkdir -p data/ckpts/llama3/meta/3.2-1B data/tokenizers/llama3
curl -L -o data/tokenizers/llama3/tokenizer.json https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/tokenizer.json
curl -L -o data/ckpts/llama3/meta/3.2-1B/model.safetensors \
  https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/model.safetensors
```

Download `model.safetensors` and `tokenizer.json` directly from the **Files and versions** tab of the HuggingFace repository, or use the commands above.

> The original `meta-llama/*` repository is gated and requires approval and a token. The commands above use an [unsloth mirror](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct) containing the same tensors. The license remains the same when the files are downloaded from the mirror. **An Instruct checkpoint trained with chat tokens is required**.

### Weight loading

The code that maps checkpoints to graph slots is in [`models/production/llama3/meta-weights.h`](../../../models/production/llama3/meta-weights.h).
Only one line is needed at the call site. The specialization assembles the path (`VAI_DATA_DIR "/ckpts/llama3/meta/" + modelCfg.variant`), so only the size needs to be set. For how the distribution-name tag works, see [`distribution-name-weights.h` in Writing custom models](../05-contributing-models.md#distribution-name-weightsh).

```cpp
loadWeights<"meta">(net);
```

## Execution

### Folder structure

```
models/production/llama3/
├── net.h                         Model definition and NodeGroup definitions
├── meta-weights.h                Mapping for Meta (HuggingFace `meta-llama/*`) checkpoints
├── demo.cpp                      demo-llama3: complete one prompt + timing
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

[`demo.cpp`](../../../models/production/llama3/demo.cpp) continues the sentence in `prompt` to generate a new sentence. It measures the `prefill` and `decode` times separately and prints them together with the generated sentence.

### chat-llama3

```bash
./build/bin/chat-llama3
```

This is the multi-turn conversation in [`main.cpp`](../../../examples/chat-llama3/main.cpp). The conversation history remains in the KV cache, and the conversation ends when an empty line is entered or the cache fills up.

## Benchmark summary

These results compare llama.cpp (Vulkan) and vAi using a **512-token prompt**. See [`benchmarks/llama3/README.md`](../../../benchmarks/llama3/README.md) for the complete results by GPU and prompt length, the measurement environment and conditions, and reproduction instructions.

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
