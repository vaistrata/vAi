# Gemma 3

Gemma 3 (Gemma Team, 2025) is a decoder-only transformer language model developed by Google DeepMind. It stacks blocks with GQA self-attention and GeGLU feed-forward layers, and applies RMSNorm to both the input and the output of each sublayer. Five local blocks with sliding-window attention alternate with one global block that attends the whole context. The 4B and 12B sizes add a SigLIP vision encoder and accept images in the prompt. The model predicts the next token by computing a probability distribution over a vocabulary of 262,144 tokens (262,208 for 4B and 12B).

**References**: [Paper](https://arxiv.org/abs/2503.19786), [Checkpoint (HuggingFace)](https://huggingface.co/google/gemma-3-4b-it), [Repository](https://github.com/google-deepmind/gemma)

## Configuration

```cpp
namespace gemma3 = model::production::gemma3;

auto modelCfg = gemma3::ModelConfig::get("4B");

gemma3::Net net({
    .wDefaultDtype = Dtype::Fp16,
}, modelCfg, {
    .maxRuntimeContext = 4096,
});
```

### NeuralNetConfig

Gemma 3 uses the defaults; the loader reads each norm's parameters at fp32.
For details, see [NeuralNet](../03-usage/neural-net.md).

### ModelConfig

`ModelConfig::get()` takes the size as a string. The values in the table are fields of its `trf` member (`TransformerConfig`), read as `modelCfg.trf.embedDim`.

| `get()` | `embedDim` | `ffnHiddenDim` | `numQHeads` | `numKVHeads` | `nBlocks` | `slidingWindow` |
|---|---|---|---|---|---|---|
| `"270M"` | 640 | 2048 | 4 | 1 | 18 | 512 |
| `"1B"` | 1152 | 6912 | 4 | 1 | 26 | 512 |
| `"4B"` | 2560 | 10240 | 8 | 4 | 34 | 1024 |
| `"12B"` | 3840 | 15360 | 16 | 8 | 48 | 1024 |

All sizes use `headDim` 256. 270M and 1B are text-only, with `maxTrainedContext` 32,768. 4B and 12B set `vision` to accept images, use `maxTrainedContext` 131,072, and scale the global blocks' RoPE by `globalRopeFactor` 8.

### RuntimeConfig

This determines the dtype and the amount of KV cache to allocate, and the unit in which a prompt is fed. If `maxRuntimeContext` is omitted, it uses `maxTrainedContext`; specifying a value allocates only that amount to save memory. The demo and the chat example reduce it to 4,096.

| Field | Default |
|---|---|
| `kvCacheDtype` | `Dtype::Fp16` |
| `maxRuntimeContext` | `uint32_t(-1)`: up to `maxTrainedContext` |
| `prefillChunk` | `512` (`0`: `slidingWindow`) |

A prompt is fed in chunks of `prefillChunk` tokens. A chunk never splits an image, so with `vision` set, `prefillChunk` must be at least 256, the number of tokens one image takes.

## vAi graph

In vAi, a model is a DAG built by connecting nodes with `-`. For details, see [Node](../03-usage/node.md) and [NodeGroup](../03-usage/node-group.md).

*`D`: embedding dimension, `H`: number of query heads, `Hkv`: number of KV heads, `B`: number of images.*

### Model architecture

```cpp
input("inTokenIds") - embedding -                           "in0"/imageCopy;
input("inImages") - visionEncoder - projector - imageRows - "in1"/imageCopy;
imageCopy - decoder - lmHead - topkSampling - output("nextTokenId");
```

`visionEncoder` and `projector` are `optional` members created only for 4B and 12B. `imageRows` ([`Reshape`](../06-nodes.md#reshape)) flattens the image tokens from `[B, 256, D]` to `[B * 256, D]`, and `imageCopy` ([`IndexCopy`](../06-nodes.md#indexcopy)) writes them over the rows of the `<image_soft_token>` placeholders in the prompt; a step without images passes the embeddings through. `embedding` scales its output by `sqrt(D)`, and `lmHead` shares the embedding's weights.

`decoder` is Gemma 3's own [`Transformer`](../../../models/production/gemma3/groups/transformer.h) group. It connects the blocks in order and ends with `finalNorm`.

```cpp
for (size_t i = 1; i < blocks.size(); ++i)
    blocks[i - 1] - blocks[i];
blocks.back() - finalNorm;
```

Every sixth block is global and the rest are local.

- **Local**: attends the last `slidingWindow` tokens with RoPE base 10,000. Its KV cache is a ring that keeps only the most recent positions; its size is set by `slidingWindow` and `prefillChunk` and does not grow with `maxRuntimeContext`.
- **Global**: attends the whole prefix with RoPE base 1,000,000. Its KV cache holds `maxRuntimeContext` positions.

### NodeGroup

Gemma 3 uses these [node groups](../03-usage/node-group.md). `Transformer`, `TransformerBlock` and `VisionEncoder` are in [`models/production/gemma3/groups/`](../../../models/production/gemma3/groups/); the rest are [shared groups](../../../vai/groups/).
`Transformer` stacks `TransformerBlock` groups, and each `TransformerBlock` contains nested `MultiHeadAttention` and `GatedFeedForward` groups.

#### TransformerBlock

```cpp
input - norm1 - mhAttn - norm2 - lastRow - gate/"then" - norm3 - ffn - norm4;
input -       "residual"/norm2 - lastRow - gate/"then" -    "residual"/norm4;
```

Each sublayer is normalized on both its input (`norm1`, `norm3`) and its output (`norm2`, `norm4`). The output norms accept the residual through `acceptResidual` and add it after normalizing, within the kernel and without a separate `Add`.
`lastRow` and `gate` exist only in the last block. `lastRow` sends only the last row, which determines the next token, to the FFN. `gate` closes for a chunk that produces no token and skips the computation from `norm3` on.

#### MultiHeadAttention

With GQA, K and V are projected into fewer heads than the queries (`H == 8`, `Hkv == 4` in 4B). RoPE applies only to Q and K; V passes through unchanged.

```cpp
input - projQ - splitQ - normQ - perQ - ropeQ - "Q"/attn;
input - projK - splitK - normK - perK - ropeK - "K"/attn;
input - projV - splitV         - perV         - "V"/attn - mergeOut - projOut;
```

Gemma 3 creates `normQ` and `normK`, which RMS-normalize Q and K per head before RoPE. `projOut` does not take the residual; `norm2` adds it.

`Rope` shares the KV cache token counter, so rotation uses the cumulative absolute position even during decode steps, rather than the offset within the current forward pass.

#### GatedFeedForward

Since `gate` and `value` read the same input and differ only in their weights, a single `Linear` computes both at twice the width. `GatedMul` applies GELU (tanh approximation) to the first half and multiplies it by the second half (GeGLU).

```cpp
gateValue - gatedMul - out;   // 2560 -> 20480 -> 10240 -> 2560
```

`out` does not take the residual; `norm4` adds it.

#### VisionEncoder

Only 4B and 12B create it. The SigLIP So400m/14 encoder cuts each 896 × 896 image into 64 × 64 patches, and a 4×4 average pool reduces them to 256 tokens.

```cpp
patchEmbed - encoder - toGrid - pool - toTokens - imageNorm;
// [B, 3, 896, 896] -> [B, 4096, 1152] -> [B, 4096, 1152] -> [B, 64, 64, 1152]
//                  -> [B, 16, 16, 1152] -> [B, 256, 1152] -> [B, 256, 1152]
```

The net's `projector` then maps the image tokens to `D`.

### Nodes used

[`TokenEmbedding`](../06-nodes.md#tokenembedding),
[`RmsNorm`](../06-nodes.md#rmsnorm),
[`LayerNorm`](../06-nodes.md#layernorm),
[`Linear`](../06-nodes.md#linear),
[`Reshape`](../06-nodes.md#reshape),
[`Permute`](../06-nodes.md#permute),
[`Rope`](../06-nodes.md#rope),
[`Attention`](../06-nodes.md#attention),
[`GatedMul`](../06-nodes.md#gatedmul),
[`Identity`](../06-nodes.md#identity),
[`Slice`](../06-nodes.md#slice),
[`Gate`](../06-nodes.md#gate),
[`IndexCopy`](../06-nodes.md#indexcopy),
[`Pool2d`](../06-nodes.md#pool2d),
[`TopKSampling`](../06-nodes.md#topksampling)

## Checkpoints

Place checkpoints in size-specific folders (`ModelConfig::variant`) under `data/ckpts/gemma3/google/`, with the tokenizer in `data/tokenizers/gemma3/`.
All sizes share `tokenizer.json`. The loader reads every `*.safetensors` file in the folder, so a checkpoint split into shards, such as 4B, is used as it is.

```
data/
├── ckpts/gemma3/google/
│   ├── 1B/
│   │   └── model.safetensors
│   ├── 4B/
│   │   ├── model-00001-of-00002.safetensors
│   │   └── model-00002-of-00002.safetensors
│   └── ...
└── tokenizers/gemma3/
    └── tokenizer.json
```

Download the files from the Instruct repository for each size and use them without conversion. The weights are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

The demo and the chat example use 4B. The `google/gemma-3-*-it` repositories are gated: accept the license on the repository page, then pass a HuggingFace access token (`HF_TOKEN`).

```bash
mkdir -p data/ckpts/gemma3/google/4B data/tokenizers/gemma3
curl -L -H "Authorization: Bearer $HF_TOKEN" -o data/tokenizers/gemma3/tokenizer.json \
  https://huggingface.co/google/gemma-3-4b-it/resolve/main/tokenizer.json
for f in model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do
  curl -L -H "Authorization: Bearer $HF_TOKEN" -o data/ckpts/gemma3/google/4B/$f \
    https://huggingface.co/google/gemma-3-4b-it/resolve/main/$f
done
```

Download the files directly from the HuggingFace repository's **Files and versions** tab or use the commands above.

> The other sizes are downloaded the same way from [`google/gemma-3-270m-it`](https://huggingface.co/google/gemma-3-270m-it), [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) and [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it). 270M and 1B come as a single `model.safetensors`, and 12B as five shards. **An Instruct checkpoint trained on chat tokens is required**.

### Loading weights

The code that maps checkpoint tensors to graph slots is in [`models/production/gemma3/google-weights.h`](../../../models/production/gemma3/google-weights.h).
Loading takes one line at the call site. The specialization constructs the path (`VAI_DATA_DIR "/ckpts/gemma3/google/" + modelCfg.variant`), so only the size needs to be selected. For distribution tags, see [Writing the weight-loading function in Writing custom models](../contributing/models.md#33-writing-the-weight-loading-function).

```cpp
loadWeights<"google">(net);
```

## Running

### Directory structure

```
models/production/gemma3/
├── net.h                         Model definition
├── groups/                       NodeGroup definitions (Transformer, VisionEncoder)
├── processing.h                  Image preprocessing and image placeholder expansion
├── google-weights.h              Google (HuggingFace `google/gemma-3-*-it`) checkpoint mapping
├── demo.cpp                      demo-gemma3, complete one prompt over an image and measure timing
└── CMakeLists.txt

examples/chat-gemma3/
├── main.cpp                      chat-gemma3: multi-turn conversation, with images
└── CMakeLists.txt

data/
├── ckpts/gemma3/google/
│   └── <variant>/                ModelConfig::variant (270M, 1B, 4B, 12B)
│       └── *.safetensors
├── tokenizers/gemma3/
│   └── tokenizer.json
└── images/                       Input images
```

### Build and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVAI_BUILD_EXAMPLES=ON
cmake --build build --config RelWithDebInfo
```

*`demo-gemma3` is built by default; `chat-gemma3` is an example and needs `-DVAI_BUILD_EXAMPLES=ON`.*

### demo-gemma3

```bash
./build/bin/demo-gemma3
# Windows (Visual Studio, Ninja Multi-Config): build\bin\RelWithDebInfo\demo-gemma3.exe
```

The demo runs 4B on the `prompt` in [`demo.cpp`](../../../models/production/gemma3/demo.cpp) together with `data/images/paper_captured0.jpg`. It measures `prefill` and `decode` times separately and prints them with the generated text. Because it always passes an image, it needs 4B or 12B.

### chat-gemma3

```bash
./build/bin/chat-gemma3
# Windows (Visual Studio, Ninja Multi-Config): build\bin\RelWithDebInfo\chat-gemma3.exe
```

This example runs a multi-turn conversation with 4B from [`main.cpp`](../../../examples/chat-gemma3/main.cpp). Writing `<image:file>` in a message adds `data/images/file` to that turn, up to 2 images per turn. Conversation history stays in the KV cache, and the session ends on an empty line or when the cache is full.
