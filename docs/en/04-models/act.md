# ACT

ACT (Action Chunking with Transformers, Zhao et al., 2023) is a robot imitation-learning policy. A ResNet-18 extracts a feature map from each camera, and a DETR-style transformer encoder-decoder takes them together with the robot state to predict several upcoming actions (a chunk) in one forward pass. Training learns a latent variable for the actions with a CVAE, but inference sets that latent to zero. vAi reads ACT policies trained by [LeRobot](https://github.com/huggingface/lerobot).

**References**: [Paper](https://arxiv.org/abs/2304.13705), [Checkpoint (HuggingFace)](https://huggingface.co/mair-robosyn/ACT_sim_item_assembly_official), [Repository](https://github.com/tonyzhaozh/act)

## Configuration

```cpp
namespace act = model::production::act;

auto modelCfg = act::ModelConfig::get({
    .variant = "mair-robosyn/item-assembly",
    .cameras = { "cam_high", "cam_right_wrist", "cam_left_wrist" },
    .imageSize = { 480, 640 },
    .stateDim = 14,
    .actionDim = 14,
    .actionChunk = 50,
});

act::Net net({
    .wDefaultDtype = Dtype::Fp16,
}, modelCfg);
```

### NeuralNetConfig

ACT uses the defaults; the loader reads only each `LayerNorm`'s coefficients at fp32. The action output (`actionHead`) is pinned to fp32 regardless of `oDefaultDtype`.
For details, see [NeuralNet](../03-usage/neural-net.md).

### ModelConfig

ACT has no per-size factories. `ModelConfig::get()` takes a `CheckpointConfig` holding what the checkpoint's `config.json` fixes.

| Field | `config.json` | Demo value |
|---|---|---|
| `variant` | (none) folder under `data/ckpts/act/` | `"mair-robosyn/item-assembly"` |
| `cameras` | `<name>` of each `observation.images.<name>`, in order | `{ "cam_high", "cam_right_wrist", "cam_left_wrist" }` |
| `imageSize` | `{H, W}` of the cameras' `shape` | `{ 480, 640 }` |
| `stateDim` | `shape` of `observation.state` | 14 |
| `actionDim` | `shape` of `action` | 14 |
| `actionChunk` | `chunk_size` | 50 |

A policy with the single unnamed camera `observation.image` leaves `cameras` as `{}`.

The rest of the architecture is fixed to LeRobot ACT's defaults.

| Field | `config.json` | Value |
|---|---|---|
| `embedDim` | `dim_model` | 512 |
| `numHeads` | `n_heads` | 8 |
| `ffnHiddenDim` | `dim_feedforward` | 3200 |
| `numEncoderBlocks` | `n_encoder_layers` | 4 |
| `normEps` | (none) | 1e-5 |

There is one decoder block (`n_decoder_layers`), and the FFN activation is ReLU. ResNet-18 strides the image by 32, so one camera's feature map is `ceil(H / 32) × ceil(W / 32)` (15 × 20 at 480 × 640), and the number of encoder tokens (`encoderTokens()`) is the latent and state tokens, 2, plus the cells of every camera's feature map.

### RuntimeConfig

ACT has no `RuntimeConfig`. One forward pass produces the whole chunk, with no KV cache and no repeated steps.

### Input and output normalization

LeRobot ACT is trained on normalized inputs and outputs (`MEAN_STD`). The loader fills `Net` members with the checkpoint's statistics; the caller normalizes the inputs and un-normalizes the output with them.

| Member | Contents | Used by |
|---|---|---|
| `imageMean`, `imageStd` | Per camera name, mean and std of the 3 channels (keyed `"image"` when `cameras` is `{}`) | Standardizing the image after scaling it to `[0, 1]` |
| `stateMean`, `stateStd` | `[stateDim]` | `normalizeState()` |
| `actionMean`, `actionStd` | `[actionDim]` | `denormalizeActions()` |

The output of `predict()` is in normalized units; `denormalizeActions()` turns it into joint units.

## vAi graph

In vAi, a model is a DAG built by connecting nodes with `-`. For details, see [Node](../03-usage/node.md) and [NodeGroup](../03-usage/node-group.md).

*`B`: number of cameras, `H × W`: input resolution, `h × w`: feature map size, `D`: embedding dimension (512), `Dh`: head dimension (64), `T`, `Tk`: number of encoder tokens (`encoderTokens()`), `C`: chunk length, `S`: state dimension, `A`: action dimension.*

### Model architecture

```cpp
input("inState") - latentStateProj - splitLatentState -                             "in0"/catTokens;
input("inImages") - imageEncoder - flatten - transpose - imageProj - mergeCameras - "in1"/catTokens;

catTokens - encoder[0];
for (uint32_t i = 1; i < modelCfg.numEncoderBlocks; ++i)
    encoder[i - 1] - encoder[i];

encoder.back() -         "memory"/decoder;
input("inDecoderQuery") - "query"/decoder;
decoder - decoderNorm - actionHead - output("outActions");
```

- **State**: `latentStateProj` projects the state `[1, S]` to `[1, 2D]`, and `splitLatentState` splits it into `[2, D]`. The first row is the latent token and the second the state token. The latent is zero at inference, so the latent token is the bias the loader put in.
- **Images**: `imageEncoder` (ResNet-18) processes the `B` cameras as a batch, turning `[B, 3, H, W]` into `[B, 512, h, w]`. `flatten`, `transpose` and `imageProj` turn each cell into a token, `[B, h*w, D]`, and `mergeCameras` joins the cameras in order into `[B*h*w, D]`.
- **Encoder**: `catTokens` puts the image tokens after the latent and state tokens to form `[2 + B*h*w, D]`, and 4 `EncoderBlock`s process it in order.
- **Decoder**: `DecoderBlock` cross-attends to the encoder output (`memory`), and `decoderNorm` and `actionHead` produce the action chunk `[C, A]`. `query` is the `decoderQuery` `[8, C, Dh]` the loader precomputes, which `predict()` feeds on every call.

### NodeGroup

ACT uses these [node groups](../03-usage/node-group.md). `EncoderBlock` and `DecoderBlock` are in [`models/production/act/groups/`](../../../models/production/act/groups/); the rest are [shared groups](../../../vai/groups/).
`ResNet18` stacks `BasicBlock` groups, `EncoderBlock` contains `MultiHeadAttention` and `FeedForward`, and `DecoderBlock` contains `FeedForward`.

#### ResNet18

torchvision's ResNet-18 without its classifier head. The stem is followed by 8 `BasicBlock`s, 2 per layer.

```cpp
conv1 - maxPool - blocks.front();
for (size_t i = 1; i < blocks.size(); ++i)
    blocks[i - 1] - blocks[i];
// [B, 3, 480, 640] -> [B, 64, 240, 320] -> [B, 64, 120, 160]
//   -> layer1 [B, 64, 120, 160] -> layer2 [B, 128, 60, 80]
//   -> layer3 [B, 256, 30, 40]  -> layer4 [B, 512, 15, 20]
```

`BasicBlock` connects two 3x3 convs; the second takes the input (or the 1x1 `shortcut`) as its `residual`, adds it, and applies ReLU.

```cpp
input - conv1 -               conv2;
input - shortcut - "residual"/conv2;
```

BatchNorm is not a node. The loader folds the BN statistics into each conv's `scale` and `bias`.

#### EncoderBlock

DETR's post-norm block: the residual is added after each sublayer, then normalized.

```cpp
input -            mhAttn - norm1 -            ffn - norm2;
input - "residual"/mhAttn;  norm1 - "residual"/ffn;
```

Instead of a separate `Add`, the last projections of `mhAttn` and `ffn` accept the residual through `acceptResidual` and add it, and `norm1` and `norm2` normalize the sums.

#### MultiHeadAttention

`EncoderBlock`'s `mhAttn` is `MultiHeadAttention<Fused>`. It computes Q, K and V with a single projection whose weight is `[3D, D]`, then cuts them into three pieces along the head axis.

```cpp
projQKV - splitHeads - perQKV - sliceQ - normQ - ropeQ - "Q"/attn;
perQKV - sliceK - normK - ropeK - "K"/attn;
perQKV - sliceV                 - "V"/attn - mergeOut - projOut;
```

ACT uses neither QK normalization nor RoPE, so `normQ`, `normK`, `ropeQ` and `ropeK` are not created.

DETR adds the position to Q and K only, which read `x + pos`, while V reads `x`. Since `(x + pos)Wᵀ + b = xWᵀ + (b + pos·Wᵀ)`, the loader makes `projQKV`'s bias a `[T, 3D]` matrix of per-token rows with `pos·Wᵀ` already added to the Q and K parts. One projection then reads `x` alone and produces all three. There is no KV cache and no mask, so every token attends to every other.

#### FeedForward

The FFN of `EncoderBlock` and `DecoderBlock`. It connects two `Linear` nodes, and ReLU is attached as the first `Linear`'s epilogue and handled within the GEMM.

```cpp
fc1 - fc2;   // 512 -> 3200 -> 512
```

#### DecoderBlock

Of ACT's one decoder block, only the cross-attention and the FFN are in the graph.

```cpp
query                                      - "Q"/attn;
memory - projKV - splitKV - perKV - sliceK - "K"/attn;
perKV - sliceV - "V"/attn - mergeOut - projOut - norm1 -            ffn - norm2;
                                                 norm1 - "residual"/ffn;
// projKV [Tk, 2D] -> splitKV [Tk, 16, 64] -> perKV [16, Tk, 64] -> sliceK/V [8, Tk, 64]
// attn [C, 8, 64] -> mergeOut [C, D]
```

- `projKV` projects K and V at once. As in the encoder, the position is already added to the bias of its K half.
- The decoder's input is zeros, so the self-attention and the norm ahead of the cross-attention depend on the weights alone. The loader computes their output `x1` once, projects `x1` plus the query positions into `query`, and adds `x1` (the cross-attention's residual) into `projOut`'s bias.

### Nodes used

[`Convolution`](../06-nodes.md#convolution),
[`Pool2d`](../06-nodes.md#pool2d),
[`Reshape`](../06-nodes.md#reshape),
[`Permute`](../06-nodes.md#permute),
[`Linear`](../06-nodes.md#linear),
[`Concat`](../06-nodes.md#concat),
[`Slice`](../06-nodes.md#slice),
[`Attention`](../06-nodes.md#attention),
[`LayerNorm`](../06-nodes.md#layernorm),
[`Identity`](../06-nodes.md#identity)

## Checkpoints

Place each checkpoint in an `<org>/<name>` folder (`ModelConfig::variant`) under `data/ckpts/act/`. The loader reads every `*.safetensors` file in the folder.

```
data/
└── ckpts/act/
    └── mair-robosyn/item-assembly/
        └── model.safetensors
```

Download the files from the policy's HuggingFace repository and use them without conversion. Licenses differ by repository.

The demo uses `mair-robosyn/ACT_sim_item_assembly_official`, which states no license.

```bash
mkdir -p data/ckpts/act/mair-robosyn/item-assembly
curl -L -o data/ckpts/act/mair-robosyn/item-assembly/model.safetensors \
  https://huggingface.co/mair-robosyn/ACT_sim_item_assembly_official/resolve/main/model.safetensors
```

Download the files directly from the HuggingFace repository's **Files and versions** tab or use the command above.

> A policy that ships its normalization statistics in a separate `policy_preprocessor_step_3_normalizer_processor.safetensors` (newer LeRobot exports) needs that file in the same folder too. Older exports keep the statistics inside `model.safetensors`.

The loader supports the following.

| Supported | Not supported |
|---|---|
| Any number of cameras, all one resolution | Cameras of different resolutions |
| Any state width, action width and chunk length | An architecture other than `ModelConfig`'s (not checked) |
| `MEAN_STD` normalization (in `model.safetensors` or the step 3 normalizer file) | `observation.environment_state` |
| | `MIN_MAX` or `IDENTITY` normalization |
| | A preprocessing step ahead of the normalizer (e.g., relative actions) |

### Loading weights

The code that maps checkpoint tensors to graph slots is in [`models/production/act/lerobot-weights.h`](../../../models/production/act/lerobot-weights.h).
Loading takes one line at the call site. The specialization constructs the path (`VAI_DATA_DIR "/ckpts/act/" + modelCfg.variant`). For distribution tags, see [Writing the weight-loading function in Writing custom models](../contributing/models.md#33-writing-the-weight-loading-function).

```cpp
loadWeights<"lerobot">(net);
```

## Running

### Directory structure

```
models/production/act/
├── net.h                         Model definition
├── groups/                       NodeGroup definitions (EncoderBlock, DecoderBlock)
├── lerobot-weights.h             LeRobot ACT policy checkpoint mapping
├── demo.cpp                      demo-act, one action chunk from three cameras and a state, with timing
└── CMakeLists.txt

data/
├── ckpts/act/
│   └── <variant>/                ModelConfig::variant (e.g., mair-robosyn/item-assembly)
│       └── *.safetensors
└── images/                       Input images
```

### Build and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo
```

*`demo-act` is built by default.*

### demo-act

```bash
./build/bin/demo-act
# Windows (Visual Studio, Ninja Multi-Config): build\bin\RelWithDebInfo\demo-act.exe
```

[`demo.cpp`](../../../models/production/act/demo.cpp) runs the `mair-robosyn/item-assembly` policy on the three camera images in `data/images/` and a joint state of all zeros. It measures the forward time and prints it with the first 5 steps × 8 dimensions of the action chunk, in joint units.
