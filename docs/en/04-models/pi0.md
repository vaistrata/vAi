# π0

π0 (Black et al., 2024) is a vision-language-action (VLA) model developed by Physical Intelligence. PaliGemma 3B reads the camera images and the task instruction to fill a KV cache, and an action expert the size of Gemma 300M reads that cache to produce a chunk of robot actions. The actions come from flow matching: starting from noise, the model integrates a velocity field over a fixed number of steps. The model predicts 50 actions of 32 dimensions at a time.

**References**: [Paper](https://arxiv.org/abs/2410.24164), [Checkpoint (HuggingFace)](https://huggingface.co/lerobot/pi0_base), [Repository](https://github.com/Physical-Intelligence/openpi)

## Configuration

```cpp
namespace pi0 = model::production::pi0;

auto modelCfg = pi0::ModelConfig::_3B();

pi0::Net net({
    .wDefaultDtype = Dtype::Fp16,
}, modelCfg);
```

### NeuralNetConfig

π0 uses the defaults; the loader reads only each norm's parameters and the per-step time table (`stepTimeBias`) at fp32. The action output (`velocityHead`) is pinned to fp32 regardless of `oDefaultDtype`.
For details, see [NeuralNet](../03-usage/neural-net.md).

### ModelConfig

There is one factory, `_3B()`, with `variant` `"3B"`.

| Field | Value | Description |
|---|---|---|
| `maxCameras` | 3 | Cameras the policy was trained with; the most images one run takes |
| `maxPromptTokens` | 48 | Maximum tokens of the task instruction |
| `vocabSize` | 257,152 | 256,000 text, 1,024 `<loc>` and 128 `<seg>` |
| `actionDim` | 32 | Width of one action; the state is padded to the same width |
| `actionChunk` | 50 | Actions one chunk carries |

`visionTrf` is the `VisionTransformerConfig` built by `so400m14(224)` in [`models/common.h`](../../../models/common.h); `languageTrf` and `actionTrf` are the `TransformerConfig`s built by `gemma("2B")` and `gemma("300M")`.

| Member | `nBlocks` | `embedDim` | `headDim` | `numQHeads` | `numKVHeads` | `ffnHiddenDim` |
|---|---|---|---|---|---|---|
| `visionTrf` (SigLIP So400m/14) | 27 | 1152 | 72 | 16 | 16 | 4304 |
| `languageTrf` (Gemma 2B) | 18 | 2048 | 256 | 8 | 1 | 16384 |
| `actionTrf` (Gemma 300M) | 18 | 1024 | 256 | 8 | 1 | 4096 |

`languageTrf` and `actionTrf` differ only in width; their block count, `headDim` and number of KV heads are the same, which lets the two stacks share one KV cache block for block. `maxContext()` is the most positions the KV cache holds: the image tokens of three cameras (3 × 256), 48 instruction tokens, 1 state token and 50 actions, 867 in all.

### RuntimeConfig

This determines the number of denoising steps and the KV cache dtype.

| Field | Default |
|---|---|
| `denoiseSteps` | `10` |
| `kvCacheDtype` | `Dtype::Fp16` |

`denoiseSteps` is the number of Euler steps that produce an action chunk. The constructor and the loader set the step size and the time table from it, so the `RuntimeConfig` has to be passed as the constructor's third argument and must not be changed after construction.

## vAi graph

In vAi, a model is a DAG built by connecting nodes with `-`. For details, see [Node](../03-usage/node.md) and [NodeGroup](../03-usage/node-group.md).

*`B`: number of cameras, `T`: number of instruction tokens, `D`: language embedding dimension (2048), `Da`: action embedding dimension (1024), `A`: action dimension (32), `C`: chunk length (50).*

### Model architecture

```cpp
input("inImages") - patchEmbed - visionEncoder - projector - mergeCameras - "in0"/catCamsPrompt;
input("inPromptTokens")                                     - promptEmbed - "in1"/catCamsPrompt;
catCamsPrompt - languageEncoder;

input("inState") - stateProj - output("outStateToken");

input("inStateToken")                                                       - "in0"/catStateActs;
input("inActions") - actionProj              - actionTimeIn - actionTimeOut - "in1"/catStateActs;
input("inActions") - stepTimeBias - "residual"/actionTimeIn;
input("inActions")               - "residual"/velocityHead;
catStateActs - actionDecoder - sliceActions - velocityHead - output("outActions");
```

The one graph holds two phases, each fed different inputs.

**Prefill** (`Net::prefill`) takes the images, the instruction and the state, and runs once.

- `patchEmbed` and `visionEncoder` process the `B` cameras as a batch into `[B, 256, 1152]`; `projector` maps them to `[B, 256, D]`, and `mergeCameras` flattens them to `[B * 256, D]`.
- `catCamsPrompt` appends the instruction embeddings to form `[B * 256 + T, D]`, and `languageEncoder` fills the KV cache with this prefix. The prefix attends to itself in full.
- The `gate` of `languageEncoder`'s last block is always closed, so that block's `norm2` and FFN and `finalNorm` do not run. Only the KV cache is needed.
- `stateProj` projects the state `[1, A]` into a state token `[1, Da]` and keeps it.

**Decode** (`Net::decode`) takes the actions `[C, A]`, initialized with noise, runs `denoiseSteps` times, and writes the result back into the same tensor.

- `actionProj` projects the actions to `[C, Da]`. `actionTimeIn` adds this step's time term (`stepTimeBias`) as its `residual` and applies SiLU, and `actionTimeOut` projects again.
- `catStateActs` puts the state token in front to form `[1 + C, Da]`, and `actionDecoder` computes over it with the prefix KV cache.
- `sliceActions` drops the state token's row to leave `[C, Da]`, and `velocityHead` projects the velocity `v` to `[C, A]`.

`velocityHead` takes the input actions as its `residual` and has `scalarScale` `-1 / denoiseSteps`, so its output is one Euler step, `x - v / denoiseSteps`. With no separate add node, it integrates from `t = 1` (noise) to `t = 0` (actions).

`visionEncoder`, `languageEncoder` and `actionDecoder` are the shared [`Transformer`](../../../vai/groups/transformer.h) group. It connects the blocks in order and ends with `finalNorm`.

```cpp
for (size_t i = 1; i < blocks.size(); ++i)
    blocks[i - 1] - blocks[i];
blocks.back() - finalNorm;
```

- **`visionEncoder`**: SigLIP So400m/14. As a `Transformer<Classic>` it uses `LayerNorm` and `FeedForward`, and `allowBatch` keeps the camera axis, so each camera attends only among its own patches. It has no KV cache.
- **`languageEncoder`**: Gemma 2B. As a `Transformer<Default>` it uses `RmsNorm` and `GatedFeedForward`, with a KV cache of `maxContext()` positions.
- **`actionDecoder`**: Gemma 300M. A `Transformer<Default>` that shares the KV cache with `languageEncoder`.

#### Time embedding

The checkpoint's `action_time_mlp_in` is a `[Da, 2Da]` projection over `[action embedding, time embedding]`. The loader splits it along its input.

- The action half becomes `actionTimeIn`'s `[Da, Da]` weight.
- The time half and the bias are precomputed for each step `k` as `W_t · emb(t_k) + b` (`t_k = 1 - k / denoiseSteps`) and stored in the `[denoiseSteps, 1, Da]` table of `stepTimeBias` ([`CyclicTableReader`](../06-nodes.md#cyclictablereader)).

`stepTimeBias` emits the next row each time it runs, so each step gets its time term without the decode loop writing a time value from the host.

#### KV cache sharing and the mask

`languageEncoder` and `actionDecoder` use the same `K_cache` and `V_cache` in each block, and share the KV cache token counter.

- The prefill sets the counter to 0 and writes the prefix at the front of the cache.
- The decode holds the counter at the prefix length across its steps. Each step writes the K/V of the state and action tokens right after the prefix, overwriting what the previous step wrote.

`actionDecoder` has a block mask. It splits the state token `[0, 1)` and the action tokens `[1, 1 + C)` into two blocks, neither causal, so the state token sees the prefix and itself, and the action tokens see the prefix, the state token and every action token.

### NodeGroup

π0 uses these [node groups](../03-usage/node-group.md), all of them [shared groups](../../../vai/groups/).
`Transformer` stacks `TransformerBlock` groups, and each `TransformerBlock` contains nested `MultiHeadAttention` and `FeedForward` (vision) or `GatedFeedForward` (language, action) groups.

The vision and language encoders have the same structure as PaliGemma 3B ([PaliGemma](paligemma.md)).

#### PatchEmbedding

It cuts each camera image into non-overlapping patches of `patchSize` (14), flattens each patch into a row, and projects the rows with a single `Linear`. `allowBatch` is on, so the camera axis `B` is kept the whole way.

```cpp
cropRows - cropCols - splitPatches - perPatches - mergePatches - project;
// [B, 3, 224, 224] -> [B, 3, 16, 14, 16, 14] -> [B, 16, 16, 3, 14, 14]
//                  -> [B, 256, 588] -> [B, 256, 1152]
```

224 is a multiple of 14, so `cropRows` and `cropCols` pass the input through. The loader adds SigLIP's position embedding into `project`'s bias as a `[256, 1152]` matrix, so no node adds the positions.

#### TransformerBlock

```cpp
input - norm1 -    mhAttn - lastRow - gate/"then" - norm2 -    ffn;
input - "residual"/mhAttn - lastRow - gate/"then" - "residual"/ffn;
```

Instead of a separate `Add`, the last projection of each sublayer accepts the residual through `acceptResidual` and adds it within the kernel.
`lastRow` and `gate` are `optional` members created by configuration; when absent, they drop out of the connection. π0 does not use `lastRow`, and places `gate` only in `languageEncoder`'s last block.

#### MultiHeadAttention

```cpp
input - projQ - splitQ - normQ - perQ - ropeQ - "Q"/attn;
input - projK - splitK - normK - perK - ropeK - "K"/attn;
input - projV - splitV         - perV         - "V"/attn - mergeOut - projOut;
```

- **`visionEncoder`**: Q, K and V all have 16 heads. It has no RoPE, so `ropeQ` and `ropeK` are not created.
- **`languageEncoder`, `actionDecoder`**: with GQA, 8 query heads share 1 K/V head, and RoPE applies only to Q and K. With an input width of 1024, `actionDecoder`'s `projQ` maps 1024 → 2048, `projK` and `projV` map 1024 → 256, and `projOut` maps 2048 back to 1024.

None of the three uses QK normalization, so `normQ` and `normK` are not created. Because `projOut` has `acceptResidual = true`, it receives the block residual through `residual` and adds it to the projection result.

`Rope` reads the KV cache token counter, so in a decode the state and action tokens rotate by their absolute positions after the prefix.

#### FeedForward

The FFN of `visionEncoder`. It connects two `Linear` nodes, and GELU (tanh approximation) is attached as the first `Linear`'s epilogue and handled within the GEMM.

```cpp
fc1 - fc2;   // 1152 -> 4304 -> 1152
```

`fc2` has `acceptResidual = true` and receives the block residual through `residual`.

#### GatedFeedForward

The FFN of `languageEncoder` and `actionDecoder`. Since `gate` and `value` read the same input and differ only in their weights, a single `Linear` computes both at twice the width. `GatedMul` applies GELU (tanh approximation) to the first half and multiplies it by the second half (GeGLU).

```cpp
gateValue - gatedMul - out;   // languageEncoder: 2048 -> 32768 -> 16384 -> 2048
                              // actionDecoder:   1024 -> 8192  -> 4096  -> 1024
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
[`Gate`](../06-nodes.md#gate),
[`CyclicTableReader`](../06-nodes.md#cyclictablereader)

## Checkpoints

Place the checkpoint in `data/ckpts/pi0/lerobot/3B/` (`ModelConfig::variant`), with the tokenizer in `data/tokenizers/paligemma/`. π0 uses PaliGemma's tokenizer as it is.

```
data/
├── ckpts/pi0/lerobot/
│   └── 3B/
│       └── model.safetensors
└── tokenizers/paligemma/
    └── tokenizer.json
```

Download the checkpoint from [`lerobot/pi0_base`](https://huggingface.co/lerobot/pi0_base), LeRobot's port, and use it without conversion. The weights are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

`lerobot/pi0_base` has no tokenizer, so download `tokenizer.json` from [`google/paligemma-3b-mix-224`](https://huggingface.co/google/paligemma-3b-mix-224). That repository is gated: accept the license on the repository page, then pass a HuggingFace access token (`HF_TOKEN`).

```bash
mkdir -p data/ckpts/pi0/lerobot/3B data/tokenizers/paligemma
curl -L -o data/ckpts/pi0/lerobot/3B/model.safetensors \
  https://huggingface.co/lerobot/pi0_base/resolve/main/model.safetensors
curl -L -H "Authorization: Bearer $HF_TOKEN" -o data/tokenizers/paligemma/tokenizer.json \
  https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/tokenizer.json
```

Download the files directly from the HuggingFace repository's **Files and versions** tab or use the commands above.

### Loading weights

The code that maps checkpoint tensors to graph slots is in [`models/production/pi0/lerobot-weights.h`](../../../models/production/pi0/lerobot-weights.h).
Loading takes one line at the call site. The specialization constructs the path (`VAI_DATA_DIR "/ckpts/pi0/lerobot/" + modelCfg.variant`). For distribution tags, see [Writing the weight-loading function in Writing custom models](../contributing/models.md#33-writing-the-weight-loading-function).

```cpp
loadWeights<"lerobot">(net);
```

## Running

### Directory structure

```
models/production/pi0/
├── net.h                         Model definition
├── lerobot-weights.h             LeRobot (HuggingFace `lerobot/pi0_base`) checkpoint mapping
├── processing.h                  Flow-matching initial noise (sampleNoise)
├── demo.cpp                      demo-pi0, one action chunk from three cameras, an instruction and a state, with timing
└── CMakeLists.txt

data/
├── ckpts/pi0/lerobot/
│   └── 3B/                       ModelConfig::variant
│       └── model.safetensors
├── tokenizers/paligemma/
│   └── tokenizer.json
└── images/                       Input images
```

### Build and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo
```

*`demo-pi0` is built by default.*

### demo-pi0

```bash
./build/bin/demo-pi0
# Windows (Visual Studio, Ninja Multi-Config): build\bin\RelWithDebInfo\demo-pi0.exe
```

[`demo.cpp`](../../../models/production/pi0/demo.cpp) runs on the three camera images in `data/images/`, the instruction `prompt` and an arbitrary joint state. It measures the prefill and denoise times separately and prints them with the first 5 steps × 8 dimensions of the action chunk.
