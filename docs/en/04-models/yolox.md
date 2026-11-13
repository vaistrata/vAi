# YOLOX

YOLOX (Megvii, 2021) is a single-pass object detection model. It consists of a CSPDarknet backbone with Focus at the input,
a YOLOPAFPN neck, and an anchor-free detection head with separate regression and classification branches. At three scales (strides 8, 16, and 32),
it produces boxes, objectness scores, and class probabilities. It uses checkpoints for the 80 COCO classes.

The supported sizes are Small, Medium, Large, and XLarge. YOLOX-Nano uses depthwise convolution (`groups = in_channels`), so it is currently excluded; support is planned.

**References**: [Repository](https://github.com/Megvii-BaseDetection/YOLOX),
[Paper](https://arxiv.org/abs/2107.08430),
[Official checkpoints](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0)

## Configuration

```cpp
namespace yolox = model::experimental::yolox;

auto cfg = yolox::getConfigCOCO(yolox::Tier::LARGE);

yolox::Net net({
    .device       = device,
    .defaultDtype = Dtype::Fp16,
}, cfg, {
    .imageSize     = 640,
    .confThreshold = 0.25f,
    .iouThreshold  = 0.45f,
});
```

### NeuralNetConfig

YOLOX uses the defaults. Both conv weights and BN parameters are stored in `defaultDtype` (fp16).
The weight loader computes BN `scale` and `bias` on the CPU, so there is no separate precision setting.
For details, see [Using vAi > NeuralNet](../03-usage/neural-net.md).

### ModelConfig

`Tier` determines the base channel width (`baseChannels`) and repeat count (`baseDepth`),
and `getConfigCOCO(tier)` adds 80 classes to create the `ModelConfig`.

| `Tier` | `baseChannels` | `baseDepth` | Checkpoint |
|---|---:|---:|---|
| `SMALL` | 32 | 1 | `yolox_s.pth` |
| `MEDIUM` | 48 | 2 | `yolox_m.pth` |
| `LARGE` | 64 | 3 | `yolox_l.pth` |
| `XLARGE` | 80 | 4 | `yolox_x.pth` |

Starting at the stem, the backbone channel widths are 1, 2, 4, 8, and 16 times `baseChannels`.
The CSP repeat counts in `dark2` through `dark5` are 1, 3, 3, and 1 times `baseDepth`, respectively,
and the neck's CSP repeat count is `baseDepth`. All COCO checkpoints use `numClasses` 80.

### RuntimeConfig

These settings specify the input resolution and postprocessing thresholds. The training resolution of 640 is the best match, but other values are accepted.

| Field | Default |
|---|---|
| `imageSize` | `640` |
| `confThreshold` | `0.25` |
| `iouThreshold` | `0.45` |

## vAi graph

In vAi, a model is a DAG built by connecting nodes with `-`. For details, see [Using vAi > Node](../03-usage/node.md) and [NodeGroup](../03-usage/node-group.md).

The dedicated `node::FocusConv` handles the Focus stem; the built-in
[`Convolution`](../06-nodes.md#convolution) handles the remaining convolutions. For Conv-BN-SiLU blocks,
the weight loader computes BN `scale` and `bias` on the CPU and fills those slots,
and the node uses the filled slots and SiLU. The detection head uses biased 1x1 convolutions for prediction.

*`H` and `W`: input resolution, `N`: number of detection positions across all scales (8,400 at 640x640).*

### Model architecture

```cpp
input("in0") - backbone;
backbone/"out_p3" - "in_p3"/neck/"out_p3" - "in_p3"/head;
backbone/"out_p4" - "in_p4"/neck/"out_p4" - "in_p4"/head;
backbone/"out_p5" - "in_p5"/neck/"out_p5" - "in_p5"/head;

for (const char* scale : {"p3", "p4", "p5"})
    for (const char* output : {"regobj", "cls"})
    {
        const std::string slot = std::string(output) + "_" + scale;
        head/slot - slot/detector;
    }
detector - output("out0");  // [N, 5 + numClasses]
```

The output has shape `[8400, 85]`. Each row contains `cx, cy, width, height, objectness, class[0..79]`.
Box coordinates are in pixels after grid and stride decoding; objectness and class values are probabilities after sigmoid.
NMS runs outside the graph in postprocessing ([`models/experimental/yolox/processing.h`](../../../models/experimental/yolox/processing.h)).
The output contains the same values as the comparison frameworks' batch-1 output, `[1, 8400, 85]`.

### NodeGroup

These [node groups](../03-usage/node-group.md) are defined for the YOLOX implementation. The four top-level groups are `Backbone`, `Neck`, `Head`,
and `DetectionDecoder`, with `FocusModule`, `Bottleneck`, `CSPLayer`,
`SPPBottleneck`, and stage-specific groups nested inside them.

#### FocusModule

The official Focus splits even and odd rows and columns, stacks them along the channel axis, and applies a 3x3 convolution.
vAi performs this in one `node::FocusConv`. It computes Conv-BN-SiLU directly from the original input without creating
an intermediate Focus tensor, mapping `[C, H, W]` to `[Co, H/2, W/2]`.

#### Bottleneck

This block connects a 1x1 convolution to a 3x3 convolution. If `shortcut` is enabled and the input and output channel counts match,
the second convolution adds the input as its `residual`.

```cpp
input - conv1 - conv2;
input - "residual"/conv2;  // shortcut with matching channels only
```

#### CSPLayer

The input splits into two 1x1 conv branches. The first passes through a chain of `Bottleneck` groups,
then concatenates with the second along the channel axis before a 1x1 conv sets the output channel count.

```cpp
input - conv1 - bottlenecks[0];
for (uint32_t i = 1; i < repeats; ++i)
    bottlenecks[i - 1] - bottlenecks[i];
bottlenecks.back() - "in0"/concat;
input - conv2     - "in1"/concat;
concat - conv3;
```

#### SPPBottleneck

Three consecutive 5x5 pooling operations compute the official SPP's parallel 5x5, 9x9, and 13x13 max pooling.
The dedicated `node::SPPF` handles pooling and concatenates the four tensors, including the input, in one kernel.

```cpp
conv1 - sppf - conv2;
```

#### Backbone, Neck, Head

| Group | Description | Slots |
|---|---|---|
| `Backbone` | Focus stem followed by four stride-2 conv + CSP stages (`dark2` through `dark5`), with SPP added in the last stage | `in0` -> `out_p3`, `out_p4`, `out_p5` |
| `Neck` | YOLOPAFPN: top-down FPN (upsample + concat + CSP), then bottom-up PAN (stride-2 conv + concat + CSP) | `in_p3..5` -> `out_p3..5` |
| `Head` | A 1x1 stem per scale followed by two branches, regression/objectness and classification (two 3x3 convs + 1x1 pred) | `in_p3..5` -> `regobj_p3..5` `[5, Hₚ, Wₚ]`, `cls_p3..5` `[numClasses, Hₚ, Wₚ]` |
| `DetectionDecoder` | Reads all three scales' outputs directly and applies grid/stride decoding and sigmoid | `regobj_p3..5`, `cls_p3..5` -> `out0` `[N, 5 + numClasses]` |

A single five-channel convolution computes the four regression channels and one objectness channel.
`node::Decode` reads `regobj` and `cls` directly, avoiding a `[85, Hₚ, Wₚ]` concatenation buffer for each scale.

### Model-specific nodes

These YOLOX-specific operations live in [`models/experimental/yolox/nodes/`](../../../models/experimental/yolox/nodes/) and
[`shaders/`](../../../models/experimental/yolox/shaders/).

| Node | Description | Kernel |
|---|---|---|
| `node::FocusConv` | Fused Focus and stem Conv-BN-SiLU | `focus-conv.comp`, `focus-conv-cm.comp` |
| `node::SPPF` | Three-stage max pooling and concatenation in one kernel | `sppf.comp` |
| `node::Decode` | Decodes regression/objectness and class outputs to `[N, 5 + numClasses]` | `decode.comp` |

The FP16 cooperative-matrix path in `node::FocusConv` transforms the weights once at initialization
using `focus-conv-transform.comp`.

### Nodes used

[`Convolution`](../06-nodes.md#convolution),
[`NearestUpsample`](../06-nodes.md#nearestupsample),
[`Concat`](../06-nodes.md#concat),
[`Add`](../06-nodes.md#add),
[`Identity`](../06-nodes.md#identity)

## Checkpoints

Each tier has its own folder under `data/ckpts/yolox/megvii/`.
The demo input image is in `data/images/`.

```text
data/
├── ckpts/yolox/megvii/
│   ├── large/
│   │   ├── model.pth             Megvii original
│   │   └── model.safetensors     conversion result: file read by the loader
│   └── ...
└── images/
    └── dog.jpg                   demo input
```

Download the `.pth` files distributed by Megvii, then convert them to `.safetensors`.

**1.** Download the required checkpoint from the [official release](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0)
and save it as `data/ckpts/yolox/megvii/<small|medium|large|xlarge>/model.pth`. For Large:

```bash
mkdir -p data/ckpts/yolox/megvii/large
curl -L -o data/ckpts/yolox/megvii/large/model.pth \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth
```

**2.** Convert the file. The converter reads the checkpoint's `model` state dict, so it does not require the YOLOX package.

```bash
pip install torch safetensors
python models/experimental/yolox/megvii-weights-convertor.py        # all available tiers
python models/experimental/yolox/megvii-weights-convertor.py small large    # selected tiers
```

This creates `model.safetensors` next to each `model.pth`.

### Load Weight

The mapping from checkpoint keys to graph slots is in
[`models/experimental/yolox/megvii-weights.h`](../../../models/experimental/yolox/megvii-weights.h).
The caller needs one line:

```cpp
loadWeights<"megvii">(net);
```

The specialization constructs the path from `model.cfg.tier` (`VAI_DATA_DIR "/yolox/yolox_<tier letter>.safetensors"`),
so the caller only selects the tier. For distribution tags, see [Supported models > Loading weights](README.md#loading-weights).

The official state dict's `backbone.backbone`, `backbone`, and `head` keys map to the backbone, neck, and head members.
Convolution weights and BN parameters are read at the net's `wDefaultDtype`. The regression and objectness weights and biases
are combined in `regObjPred`.

## Running

### Directory structure

```text
models/experimental/yolox/
├── net.h                       Model and NodeGroup definitions
├── nodes/                      dedicated nodes (FocusConv, SPPF, Decode)
├── shaders/                    kernels for dedicated nodes
├── megvii-weights.h             Megvii checkpoint mapping
├── megvii-weights-convertor.py  .pth -> .safetensors conversion
├── demo.cpp                    demo-yolox: detection and timing for one image
├── processing.h                preprocessing and postprocessing (letterbox, NMS)
└── CMakeLists.txt

data/
├── ckpts/yolox/megvii/
│   └── <tier>/model.safetensors      Tier: small, medium, large, xlarge
└── images/
    └── dog.jpg
```

### Build and run

```bash
cmake -B build
cmake --build build --config RelWithDebInfo --target demo-yolox

# Run
./build/bin/demo-yolox
```

The demo detects objects in `data/images/dog.jpg` using the LARGE tier and fp16, and prints profiles by kernel type.
Change the tier and inference settings at the top of [`demo.cpp`](../../../models/experimental/yolox/demo.cpp).

Preprocessing converts the RGB image loader's output to BGR CHW, places it from the top-left corner while preserving its aspect ratio,
and fills the remaining area with 114. Input values are floats in the range 0-255, without `1/255` normalization.

## Benchmark summary

These are mean inference latencies for YOLOX Large FP16 at 640x640. For measurement environments and conditions (including whether preprocessing and NMS are included),
and reproduction instructions, see [`benchmarks/yolox/README.md`](../../../benchmarks/yolox/README.md).

| | 890M (Strix Point) | 8060S (Strix Halo) | RTX 5070 | RTX 3060 Laptop |
|---|---|---|---|---|
| vAi (ms) | TBD | TBD | TBD | TBD |
| ncnn (ms) | TBD | TBD | TBD | TBD |
| OpenVINO (ms) | TBD | TBD | TBD | TBD |


For a summary across all models, see [Supported models](README.md#combined-benchmarks).
