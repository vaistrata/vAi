# PaliGemma

PaliGemma (Beyer et al., 2024) is a vision-language model developed by Google. A SigLIP So400m/14 vision encoder turns the image into tokens, which a linear projection places ahead of the prompt of a Gemma 2B decoder. The prefix of image tokens and prompt attends to itself in full, and only the generated tokens attend causally. The model predicts the next token by computing a probability distribution over a vocabulary of 257,152 tokens (256,000 text, 1,024 `<loc>` and 128 `<seg>`).

**References**: [Paper](https://arxiv.org/abs/2407.07726), [Checkpoint (HuggingFace)](https://huggingface.co/google/paligemma-3b-mix-224), [Repository](https://github.com/google-research/big_vision)

## Configuration

```cpp
namespace paligemma = model::production::paligemma;

auto modelCfg = paligemma::ModelConfig::_3B_224();

paligemma::Net net({
    .wDefaultDtype = Dtype::Fp16,
}, modelCfg, {
    .maxRuntimeContext = 2048,
});
```

### NeuralNetConfig

PaliGemma uses the defaults; the loader reads only each norm's parameters at fp32.
For details, see [NeuralNet](../03-usage/neural-net.md).

### ModelConfig

The two factories differ only in input resolution.

| Factory | `variant` | `visionTrf.imageSize` | Image tokens |
|---|---|---|---|
| `_3B_224()` | `"3B-224"` | 224 | 256 (16 × 16) |
| `_3B_448()` | `"3B-448"` | 448 | 1024 (32 × 32) |

`visionTrf` is the `VisionTransformerConfig` built by `so400m14(224)` or `so400m14(448)` in [`models/common.h`](../../../models/common.h); `languageTrf` is the `TransformerConfig` built by `gemma("2B")`.

| Member | `nBlocks` | `embedDim` | `headDim` | `numQHeads` | `numKVHeads` | `ffnHiddenDim` |
|---|---|---|---|---|---|---|
| `visionTrf` (SigLIP So400m/14) | 27 | 1152 | 72 | 16 | 16 | 4304 |
| `languageTrf` (Gemma 2B) | 18 | 2048 | 256 | 8 | 1 | 16384 |

`vocabSize` 257,152 and `maxTrainedContext` 8,192 are common.

### RuntimeConfig

This determines the dtype and the amount of KV cache to allocate. If `maxRuntimeContext` is omitted, it uses `maxTrainedContext`; specifying a smaller value allocates only that amount to save memory. A value above `maxTrainedContext` is clamped to it. The demo reduces it to 2,048.

| Field | Default |
|---|---|
| `kvCacheDtype` | `Dtype::Fp16` |
| `maxRuntimeContext` | `uint32_t(-1)`: up to `maxTrainedContext` |

Image tokens take KV cache positions too. With `_3B_448()` one image takes 1,024 positions, so `maxRuntimeContext` must cover the image tokens, the prompt and the tokens to generate.

## vAi graph

In vAi, a model is a DAG built by connecting nodes with `-`. For details, see [Node](../03-usage/node.md) and [NodeGroup](../03-usage/node-group.md).

*`N`: number of image tokens, `T`: number of prompt tokens, `Dv`: vision embedding dimension (1152), `D`: language embedding dimension (2048), `S`: input resolution.*

### Model architecture

```cpp
input("inImage")  - patchEmbed - encoder - projector - "in0"/catTokens;
input("inTokenIds")                      - embedding - "in1"/catTokens;
catTokens - decoder - lmHead - topkSampling - output("nextTokenId");
```

`patchEmbed` turns a `[3, S, S]` image into `[N, Dv]` patch tokens, `encoder` (SigLIP) processes them, and `projector` maps them to `[N, D]`. `catTokens` ([`Concat`](../06-nodes.md#concat)) appends the prompt embeddings after the image tokens to form `[N + T, D]`. Its `allowPartialInputs` is on, so a decode step, which feeds only the previous token and no image, brings only `in1` and the vision nodes do not run.

`embedding` scales its output by `sqrt(D)`, and `lmHead` shares the embedding's weights.

`encoder` and `decoder` are the shared [`Transformer`](../../../vai/groups/transformer.h) group. It connects the blocks in order and ends with `finalNorm`.

```cpp
for (size_t i = 1; i < blocks.size(); ++i)
    blocks[i - 1] - blocks[i];
blocks.back() - finalNorm;
```

- **`encoder`**: `Transformer<Classic>`, with `LayerNorm` and `FeedForward`. It has no KV cache and attends only among the patches of one image.
- **`decoder`**: `Transformer<Default>`, with `RmsNorm` and `GatedFeedForward`, and a KV cache of `maxRuntimeContext` positions. It sets no causal mask, so in the prefill the image tokens and the prompt attend to each other in full (prefix-LM).

### NodeGroup

PaliGemma uses these [node groups](../03-usage/node-group.md), all of them [shared groups](../../../vai/groups/).
`Transformer` stacks `TransformerBlock` groups, and each `TransformerBlock` contains nested `MultiHeadAttention` and `FeedForward` (vision) or `GatedFeedForward` (language) groups.

#### PatchEmbedding

It cuts the image into non-overlapping patches of `patchSize` (14), flattens each patch into a row, and projects the rows with a single `Linear`.

```cpp
cropRows - cropCols - splitPatches - perPatches - mergePatches - project;
// [3, S, S] -> [3, G*14, S] -> [3, G*14, G*14] -> [3, G, 14, G, 14]
//           -> [G, G, 3, 14, 14] -> [N, 588] -> [N, 1152]    (G = S / 14, N = G * G)
```

`S` is a multiple of 14, so `cropRows` and `cropCols` pass the input through. The loader adds SigLIP's position embedding into `project`'s bias as an `[N, Dv]` matrix, so no node adds the positions.

#### TransformerBlock

```cpp
input - norm1 -    mhAttn - lastRow - gate/"then" - norm2 -    ffn;
input - "residual"/mhAttn - lastRow - gate/"then" - "residual"/ffn;
```

Instead of a separate `Add`, the last projection of each sublayer accepts the residual through `acceptResidual` and adds it within the kernel.
`lastRow` and `gate` are `optional` members created by configuration; when absent, they drop out of the connection. PaliGemma places `lastRow` only in the last block of `decoder`, sending only the last row, which determines the next token, to the FFN. It does not use `gate`.

#### MultiHeadAttention

```cpp
input - projQ - splitQ - normQ - perQ - ropeQ - "Q"/attn;
input - projK - splitK - normK - perK - ropeK - "K"/attn;
input - projV - splitV         - perV         - "V"/attn - mergeOut - projOut;
```

- **`encoder`**: Q, K and V all have 16 heads. It has no RoPE, so `ropeQ` and `ropeK` are not created.
- **`decoder`**: with GQA, 8 query heads share 1 K/V head. RoPE (base 10,000) applies only to Q and K.

Neither uses QK normalization, so `normQ` and `normK` are not created. Because `projOut` has `acceptResidual = true`, it receives the block residual through `residual` and adds it to the projection result.

`Rope` shares the KV cache token counter, so rotation uses the cumulative absolute position even during decode steps, rather than the offset within the current forward pass.

#### FeedForward

The FFN of `encoder`. It connects two `Linear` nodes, and GELU (tanh approximation) is attached as the first `Linear`'s epilogue and handled within the GEMM.

```cpp
fc1 - fc2;   // 1152 -> 4304 -> 1152
```

`fc2` has `acceptResidual = true` and receives the block residual through `residual`.

#### GatedFeedForward

The FFN of `decoder`. Since `gate` and `value` read the same input and differ only in their weights, a single `Linear` computes both at twice the width. `GatedMul` applies GELU (tanh approximation) to the first half and multiplies it by the second half (GeGLU).

```cpp
gateValue - gatedMul - out;   // 2048 -> 32768 -> 16384 -> 2048
```

`out` also has `acceptResidual = true` and receives the block residual through `residual`.

### Nodes used

[`TokenEmbedding`](../06-nodes.md#tokenembedding),
[`LayerNorm`](../06-nodes.md#layernorm),
[`RmsNorm`](../06-nodes.md#rmsnorm),
[`Linear`](../06-nodes.md#linear),
[`Reshape`](../06-nodes.md#reshape),
[`Permute`](../06-nodes.md#permute),
[`Slice`](../06-nodes.md#slice),
[`Concat`](../06-nodes.md#concat),
[`Rope`](../06-nodes.md#rope),
[`Attention`](../06-nodes.md#attention),
[`GatedMul`](../06-nodes.md#gatedmul),
[`Identity`](../06-nodes.md#identity),
[`TopKSampling`](../06-nodes.md#topksampling)

## Checkpoints

Place checkpoints in resolution-specific folders (`ModelConfig::variant`) under `data/ckpts/paligemma/google/`, with the tokenizer in `data/tokenizers/paligemma/`.
Both resolutions share `tokenizer.json`. The loader reads every `*.safetensors` file in the folder, so a checkpoint split into shards is used as it is.

```
data/
├── ckpts/paligemma/google/
│   ├── 3B-224/
│   │   ├── model-00001-of-00003.safetensors
│   │   ├── model-00002-of-00003.safetensors
│   │   └── model-00003-of-00003.safetensors
│   └── 3B-448/
│       └── ...
└── tokenizers/paligemma/
    └── tokenizer.json
```

Download the files from the mix repository for each resolution and use them without conversion. The weights are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

The demo uses 224. The `google/paligemma-3b-mix-*` repositories are gated: accept the license on the repository page, then pass a HuggingFace access token (`HF_TOKEN`).

```bash
mkdir -p data/ckpts/paligemma/google/3B-224 data/tokenizers/paligemma
curl -L -H "Authorization: Bearer $HF_TOKEN" -o data/tokenizers/paligemma/tokenizer.json \
  https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/tokenizer.json
for f in model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors; do
  curl -L -H "Authorization: Bearer $HF_TOKEN" -o data/ckpts/paligemma/google/3B-224/$f \
    https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/$f
done
```

Download the files directly from the HuggingFace repository's **Files and versions** tab or use the commands above.

> 448 is downloaded the same way from [`google/paligemma-3b-mix-448`](https://huggingface.co/google/paligemma-3b-mix-448) into `3B-448/`. Its files are the same as 224's.

### Loading weights

The code that maps checkpoint tensors to graph slots is in [`models/production/paligemma/google-weights.h`](../../../models/production/paligemma/google-weights.h).
Loading takes one line at the call site. The specialization constructs the path (`VAI_DATA_DIR "/ckpts/paligemma/google/" + modelCfg.variant`), so only the resolution needs to be selected. For distribution tags, see [Writing the weight-loading function in Writing custom models](../contributing/models.md#33-writing-the-weight-loading-function).

```cpp
loadWeights<"google">(net);
```

## Running

### Directory structure

```
models/production/paligemma/
├── net.h                         Model definition
├── google-weights.h              Google (HuggingFace `google/paligemma-3b-mix-*`) checkpoint mapping
├── demo.cpp                      demo-paligemma, complete one prompt about one image and measure timing
└── CMakeLists.txt

data/
├── ckpts/paligemma/google/
│   └── <variant>/                ModelConfig::variant (3B-224, 3B-448)
│       └── *.safetensors
├── tokenizers/paligemma/
│   └── tokenizer.json
└── images/                       Input images
```

### Build and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo
```

*`demo-paligemma` is built by default.*

### demo-paligemma

```bash
./build/bin/demo-paligemma
# Windows (Visual Studio, Ninja Multi-Config): build\bin\RelWithDebInfo\demo-paligemma.exe
```

The demo runs 224 on `data/images/dog.jpg` together with the `prompt` (`"caption en"`) in [`demo.cpp`](../../../models/production/paligemma/demo.cpp). It measures `prefill` and `decode` times separately and prints them with the generated answer.
