# GPT-2

GPT-2 (Radford et al., 2019) is a decoder-only Transformer-based language model developed by OpenAI. Its architecture stacks pre-norm layers with causal self-attention and GELU feed-forward layers. The model predicts the next token by computing a probability distribution over a vocabulary of 50,257 tokens.

**References**: [Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [Checkpoint (HuggingFace)](https://huggingface.co/openai-community/gpt2), [Repository](https://github.com/openai/gpt-2)

## Configuration

```cpp
namespace gpt2 = model::production::gpt2;

auto modelCfg = gpt2::ModelConfig::_124M();

gpt2::Net net({
    .device       = device,
    .defaultDtype = Dtype::Fp16,
    .ioDtypeHint  = Dtype::Fp32,
}, modelCfg);
```

### NeuralNetConfig

GPT-2 uses the defaults as they are; the loader reads every `LayerNorm`'s coefficients at fp32.
See [NeuralNet](../03-usage/neural-net.md) for details.

### ModelConfig

| Factory | `name` | `embedDim` | `numHeads` | `numLayers` |
|---|---|---|---|---|
| `_124M()` | `"124M"` | 768 | 12 | 12 |
| `_355M()` | `"355M"` | 1024 | 16 | 24 |
| `_774M()` | `"774M"` | 1280 | 20 | 36 |
| `_1_5B()` | `"1.5B"` | 1600 | 25 | 48 |

`vocabSize` 50,257 and `maxTrainedContext` 1,024 are shared by all configurations.

### RuntimeConfig

This only determines how much KV cache to reserve. If omitted, it uses the `maxTrainedContext` value as is. If a value is specified, it allocates only that amount to save memory.

| Field | Default |
|---|---|
| `maxRuntimeContext` | `0` |

## vAi graph

In vAi, the model is a DAG formed by connecting nodes with `-`. See the [Node](../03-usage/node.md) and [NodeGroup](../03-usage/node-group.md) documentation for details.

*`Tq`: number of input tokens, `D`: embedding dimension, `H`: number of heads.*

### Model architecture

```cpp
input("inTokenIds") - embedding - layers[0];
for (size_t i = 1; i < layers.size(); ++i)
    layers[i - 1] - layers[i];
layers.back() - lastRowSlice - finalNorm - lmHead - topkSampling - output("nextTokenId");
```

### NodeGroup

These are the [NodeGroups](../03-usage/node-group.md) defined to implement GPT-2.
`TransformerLayer` has a nested structure containing `MultiHeadAttention` and `FeedForward`.

#### TransformerLayer

```cpp
input - norm1 - "in0"/mhAttn - norm2 - "in0"/ffn;
input         - "residual"/mhAttn         - "residual"/ffn;
```

Instead of using a separate `Add`, the final projection of each sublayer accepts the residual through `acceptResidual` and adds it inside the kernel.

#### MultiHeadAttention

Q, K, and V are computed with a single `{D, 3D}` projection and then split into three pieces along the head axis.

```cpp
projQKV - splitHeads - perQKV - sliceQ - "Q"/attn - mergeOut - projOut;
                     perQKV - sliceK - "K"/attn;
                     perQKV - sliceV - "V"/attn;
```

Because `projOut` has `acceptResidual = true`, it accepts the block residual as `residual` and adds it to the projection result.

#### FeedForward

This group connects two `Linear`s. GELU is attached as the epilogue of the first `Linear` rather than implemented as a separate node, so it is processed inside the GEMM.

```cpp
fc1 - fc2;   // 768 -> 3072 -> 768
```

`fc2` also has `acceptResidual = true` and accepts the block residual as `residual`.

### Nodes used

[`TokenEmbedding`](../06-nodes.md#tokenembedding) ·
[`LayerNorm`](../06-nodes.md#layernorm) ·
[`Linear`](../06-nodes.md#linear) ·
[`Reshape`](../06-nodes.md#reshape) ·
[`Permute`](../06-nodes.md#permute) ·
[`Attention`](../06-nodes.md#attention) ·
[`Identity`](../06-nodes.md#identity) ·
[`Slice`](../06-nodes.md#slice) ·
[`TopKSampling`](../06-nodes.md#topksampling)

## Checkpoint

Checkpoints are placed in a folder for each size (`ModelConfig::variant`) under `data/ckpts/gpt2/openai/`, and the tokenizer in `data/tokenizers/gpt2/`.
All four sizes share `tokenizer.json`.

```
data/
├── ckpts/gpt2/openai/
│   ├── 124M/
│   │   └── model.safetensors
│   ├── 355M/
│   │   └── model.safetensors
│   └── ...
└── tokenizers/gpt2/
    └── tokenizer.json
```

Download them from the corresponding HuggingFace repository and use them as is without a separate conversion. The weights are licensed under the MIT License.

```bash
mkdir -p data/ckpts/gpt2/openai/124M data/tokenizers/gpt2
curl -L -o data/tokenizers/gpt2/tokenizer.json https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json
curl -L -o data/ckpts/gpt2/openai/124M/model.safetensors \
  https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
```

Download `model.safetensors` and `tokenizer.json` directly from the **Files and versions** tab of the HuggingFace repository, or use the commands above.

### Weight loading

The code that maps checkpoints to graph slots is in [`models/production/gpt2/openai-weights.h`](../../../models/production/gpt2/openai-weights.h).
Only one line is needed at the call site. The specialization assembles the path (`VAI_DATA_DIR "/ckpts/gpt2/openai/" + modelCfg.variant`),
so only the size needs to be set. For how the distribution-name tag works, see [`distribution-name-weights.h` in Writing custom models](../05-contributing-models.md#distribution-name-weightsh).

```cpp
loadWeights<"openai">(net);
```

## Execution

### Folder structure

```
models/production/gpt2/
├── net.h                         Model definition and NodeGroup definitions
├── openai-weights.h              Mapping for OpenAI (HuggingFace `openai-community/*`) checkpoints
├── demo.cpp                      demo-gpt2: complete one prompt + timing
└── CMakeLists.txt

data/
├── ckpts/gpt2/openai/
│   └── <variant>/                    ModelConfig::variant: 124M, 355M, 774M, 1.5B
│       └── model.safetensors
└── tokenizers/gpt2/
    └── tokenizer.json
```

### Build and run

```bash
cmake -B build
cmake --build build --config RelWithDebInfo

# Run
./build/bin/demo-gpt2
```

The model continues the sentence in `prompt` to generate a new sentence. It measures the `prefill` and `decode` times separately and prints them together with the generated sentence.

## Benchmark summary

These results compare llama.cpp (Vulkan) and vAi using a **512-token prompt**. See [`benchmarks/gpt2/README.md`](../../../benchmarks/gpt2/README.md) for the complete results by GPU and prompt length,
the measurement environment and conditions, and reproduction instructions.

| | | 890M (Strix Point) | 8060S (Strix Halo) | UHD Graphics (TGL GT1) | RTX 3060 Laptop | RTX 5070 | RTX 4090 | RTX 5090 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vAi | prefill (t/s) | 10,899 | 50,777 | 1,201 | 61,236 | 154,785 | 269,911 | 308,638 |
|  | decode (t/s) | 130.5 | 500.9 | 94.06 | 743.2 | 1,319 | 1,631 | 1,756 |
| llama.cpp | prefill (t/s) | 7,803 | 41,159 | 1,065 | 50,910 | 131,056 | 155,877 | 153,821 |
|  | decode (t/s) | 133.6 | 595.4 | 80.90 | 592.5 | 1,020 | 979.0 | 940.5 |
