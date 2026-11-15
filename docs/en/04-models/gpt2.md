# GPT-2

GPT-2 (Radford et al., 2019) is a decoder-only transformer language model developed by OpenAI. It stacks pre-norm blocks with causal self-attention and GELU feed-forward layers. The model predicts the next token by computing a probability distribution over a vocabulary of 50,257 tokens.

**References**: [Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [Checkpoint (HuggingFace)](https://huggingface.co/openai-community/gpt2), [Repository](https://github.com/openai/gpt-2)

## Configuration

```cpp
namespace gpt2 = model::production::gpt2;

auto modelCfg = gpt2::ModelConfig::_124M();

gpt2::Net net({
    .device        = device,
    .wDefaultDtype = Dtype::Fp16,
    .oDefaultDtype = Dtype::Fp32,
}, modelCfg);
```

### NeuralNetConfig

GPT-2 uses the defaults; the loader reads each `LayerNorm`'s coefficients at fp32.
For details, see [NeuralNet](../03-usage/neural-net.md).

### ModelConfig

| Factory | `variant` | `embedDim` | `numQHeads` | `nBlocks` |
|---|---|---|---|---|
| `_124M()` | `"124M"` | 768 | 12 | 12 |
| `_355M()` | `"355M"` | 1024 | 16 | 24 |
| `_774M()` | `"774M"` | 1280 | 20 | 36 |
| `_1_5B()` | `"1.5B"` | 1600 | 25 | 48 |

All sizes use `vocabSize` 50,257 and `maxTrainedContext` 1,024.

### RuntimeConfig

This determines the dtype and the amount of KV cache to allocate. If `maxRuntimeContext` is omitted, it uses `maxTrainedContext`; specifying a value allocates only that amount to save memory.

| Field | Default |
|---|---|
| `kvCacheDtype` | `Dtype::Fp16` |
| `maxRuntimeContext` | `uint32_t(-1)`: up to `maxTrainedContext` |

## vAi graph

In vAi, a model is a DAG built by connecting nodes with `-`. For details, see [Node](../03-usage/node.md) and [NodeGroup](../03-usage/node-group.md).

*`Tq`: number of input tokens, `D`: embedding dimension, `H`: number of heads.*

### Model architecture

```cpp
input("inTokenIds") - embedding - decoder - lmHead - topkSampling - output("nextTokenId");
```

`decoder` is the shared [`Transformer`](../../../vai/groups/transformer.h) group. It connects the blocks in order and ends with `finalNorm`.

```cpp
for (size_t i = 1; i < blocks.size(); ++i)
    blocks[i - 1] - blocks[i];
blocks.back() - finalNorm;
```

### NodeGroup

GPT-2 uses these [node groups](../03-usage/node-group.md). All of them are [shared groups](../../../vai/groups/).
`Transformer` stacks `TransformerBlock` groups, and each `TransformerBlock` contains nested `MultiHeadAttention` and `FeedForward` groups.

#### TransformerBlock

```cpp
input - norm1 -    mhAttn - lastRow - gate/"then" - norm2 -    ffn;
input - "residual"/mhAttn - lastRow - gate/"then" - "residual"/ffn;
```

The final projection in each sublayer accepts the residual through `acceptResidual` and adds it within the kernel, without a separate `Add`.
`lastRow` and `gate` are `optional` members created by the configuration, and an absent one drops out of the connection. GPT-2 creates `lastRow` only in the last block, so only the last row, which determines the next token, reaches the FFN.

#### MultiHeadAttention

Q, K, and V are computed with a single `{D, 3D}` projection, then split into three slices along the head axis.

```cpp
projQKV - splitHeads - perQKV - sliceQ - normQ - ropeQ - "Q"/attn;
perQKV - sliceK - normK - ropeK - "K"/attn;
perQKV - sliceV                 - "V"/attn - mergeOut - projOut;
```

`normQ`, `normK`, `ropeQ`, and `ropeK` are `optional` members too. GPT-2 uses neither QK normalization nor RoPE, so none of the four is created.

With `acceptResidual = true`, `projOut` receives the block's residual through `residual` and adds it to the projection result.

#### FeedForward

This group connects two `Linear` nodes. GELU is attached as an epilogue to the first `Linear` and runs within GEMM, without a separate node.

```cpp
fc1 - fc2;   // 768 -> 3072 -> 768
```

`fc2` also sets `acceptResidual = true` and receives the block's residual through `residual`.

### Nodes used

[`TokenEmbedding`](../06-nodes.md#tokenembedding),
[`LayerNorm`](../06-nodes.md#layernorm),
[`Linear`](../06-nodes.md#linear),
[`Reshape`](../06-nodes.md#reshape),
[`Permute`](../06-nodes.md#permute),
[`Attention`](../06-nodes.md#attention),
[`Identity`](../06-nodes.md#identity),
[`Slice`](../06-nodes.md#slice),
[`TopKSampling`](../06-nodes.md#topksampling)

## Checkpoints

Place checkpoints in size-specific folders (`ModelConfig::variant`) under `data/ckpts/gpt2/openai/`, with the tokenizer in `data/tokenizers/gpt2/`.
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

Download the files from the corresponding HuggingFace repository and use them without conversion. The weights are licensed under MIT.

```bash
mkdir -p data/ckpts/gpt2/openai/124M data/tokenizers/gpt2
curl -L -o data/tokenizers/gpt2/tokenizer.json https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json
curl -L -o data/ckpts/gpt2/openai/124M/model.safetensors \
  https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
```

Download `model.safetensors` and `tokenizer.json` directly from the HuggingFace repository's **Files and versions** tab or use the commands above.

### Loading weights

The code that maps checkpoint tensors to graph slots is in [`models/production/gpt2/openai-weights.h`](../../../models/production/gpt2/openai-weights.h).
Loading takes one line at the call site. The specialization constructs the path (`VAI_DATA_DIR "/ckpts/gpt2/openai/" + modelCfg.variant`),
so only the size needs to be selected. For distribution tags, see [`distribution-weights.h` in Writing custom models](../contributing/models.md#distribution-weightsh).

```cpp
loadWeights<"openai">(net);
```

## Running

### Directory structure

```
models/production/gpt2/
├── net.h                         Model and NodeGroup definitions
├── openai-weights.h              OpenAI (HuggingFace `openai-community/*`) checkpoint mapping
├── demo.cpp                      demo-gpt2: complete one prompt and measure timing
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

The demo generates text that continues the text in `prompt`. It measures `prefill` and `decode` times separately and prints them with the generated text.
