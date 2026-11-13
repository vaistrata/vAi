# YOLOv8

YOLOv8 (Ultralytics, 2023) is a single-pass object detection model. It consists of a CSPDarknet backbone, a PANet neck, and an anchor-free detection head based on DFL (Distribution Focal Loss). It produces boxes and class probabilities at three scales (strides 8, 16, and 32). It uses checkpoints for the 80 COCO classes.

**References**: [Repository](https://github.com/ultralytics/ultralytics), [Checkpoint (HuggingFace)](https://huggingface.co/Ultralytics/YOLOv8), [Documentation](https://docs.ultralytics.com/models/production/yolov8/)

## Configuration

```cpp
namespace yolov8 = model::internal::yolov8;

auto cfg = yolov8::getConfigCOCO(yolov8::Tier::LARGE);

yolov8::Net net({
    .device       = device,
    .defaultDtype = Dtype::Fp16,
}, cfg, {
    .imageSize     = 640,
    .confThreshold = 0.25f,
    .iouThreshold  = 0.45f,
});
```

### NeuralNetConfig

YOLOv8 uses the defaults. Both `conv` weights and `BN` parameters are stored in `defaultDtype` (`fp16`).
The weight loader computes BN `scale` and `bias` on the CPU, so there is no separate precision setting.
For details, see [NeuralNet](../03-usage/neural-net.md).

### ModelConfig

`Tier` determines the channel widths (`p1` through `p5`) and the number of `C2f` repeats. `getConfigCOCO(tier)`
adds 80 classes and `regMax` 16 to create the `ModelConfig`.

| `Tier` | `p1` | `p2` | `p3` | `p4` | `p5` | `C2f` repeats | Checkpoint |
|---|---|---|---|---|---|---|---|
| `NANO` | 16 | 32 | 64 | 128 | 256 | 1 | `yolov8n` |
| `SMALL` | 32 | 64 | 128 | 256 | 512 | 1 | `yolov8s` |
| `MEDIUM` | 48 | 96 | 192 | 384 | 576 | 2 | `yolov8m` |
| `LARGE` | 64 | 128 | 256 | 512 | 512 | 3 | `yolov8l` |
| `XLARGE` | 80 | 160 | 320 | 640 | 640 | 3 | `yolov8x` |

All COCO checkpoints use `numClasses` 80 and `regMax` 16.

### RuntimeConfig

These settings specify the input resolution and postprocessing thresholds. The training resolution of 640 is the best match, but other values are accepted.

| Field | Default |
|---|---|
| `imageSize` | `640` |
| `confThreshold` | `0.25` |
| `iouThreshold` | `0.45` |

## vAi graph

In vAi, a model is a DAG built by connecting nodes with `-`. For details, see [Node](../03-usage/node.md) and [NodeGroup](../03-usage/node-group.md).

The built-in [`Convolution`](../06-nodes.md#convolution) handles every `conv` in the detection backbone.
The weight loader computes BN `scale` and `bias` on the CPU and fills those slots. The node uses the filled slots and SiLU and selects the Winograd, implicit GEMM, or GEMM
path according to the kernel shape. The only dedicated nodes left in the model folder are the set of operations specific to YOLO.

*`H` and `W`: input resolution, `N`: number of anchors across all scales (8,400 at 640x640).*

### Model architecture

```cpp
input("in0") - backbone;
backbone/"out_p3" - "in_p3"/neck/"out_p3" - "in_p3"/head;
backbone/"out_p4" - "in_p4"/neck/"out_p4" - "in_p4"/head;
backbone/"out_p5" - "in_p5"/neck/"out_p5" - "in_p5"/head;

head/"reg_p3" - "reg_p3"/detector;   // six slots: reg/cls × p3/p4/p5
head/"cls_p3" - "cls_p3"/detector;   // (same for p4 and p5)
detector - output("out0");           // [4 + numClasses, N]
```

The output is a raw tensor of shape `[84, 8400]` (4 rows of boxes + 80 rows of per-class probabilities). NMS runs outside the GPU
during post-processing ([`models/internal/yolov8/processing.h`](../../../models/internal/yolov8/processing.h)).
It has the same shape as the YOLOv8 output from the comparison frameworks (ncnn, ONNX Runtime, and the rest), so the benches share one post-processing path.

### NodeGroup

These [node groups](../03-usage/node-group.md) are defined for the YOLOv8 implementation. The four top-level groups are `Backbone`, `Neck`, `Head`,
and `DetectionDecoder`, with `Bottleneck`, `C2fModule`, and `SPPFModule` nested inside them.

#### Bottleneck

This is a residual block that connects two 3x3 `conv` operations. With `shortcut` enabled, `conv2` adds the input as its `residual`; when disabled, data passes through without a residual connection.

```cpp
input - conv1 - conv2;
input - "residual"/conv2;   // shortcut only
```

#### C2fModule

This CSP block expands the channels with a 1x1 `conv`, sends only the second half of the channels (`Slice`) through a chain of `Bottleneck` groups,
then concatenates the input and every stage's output before projecting back with a 1x1 `conv`.

```cpp
convIn                                        - "in0"/concat;
convIn - lastHalfChannels - bottlenecks[0]    - "in1"/concat;
bottlenecks[i-2] - bottlenecks[i-1]           - "in<i>"/concat;   // i = 2..n
concat - convOut;
```

#### SPPFModule

This is Spatial Pyramid Pooling Fast. Three-stage max pooling and concat are processed by a single kernel in the dedicated `node::SPPF`.

```cpp
convIn - sppf - convOut;
```

#### Backbone, Neck, Head

| Group | Description | Slots |
|---|---|---|
| `Backbone` | CSPDarknet: one stride-2 `conv` (`p1`), followed by four stages of (stride-2 `conv` + `C2f`) (`p2` through `p5`), ending with SPPF | `in0` -> `out_p3`, `out_p4`, `out_p5` |
| `Neck` | PANet: a top-down FPN ([`NearestUpsample`](../06-nodes.md#nearestupsample) + concat + `C2f`), followed by a bottom-up PAN (stride-2 `conv` + concat + `C2f`) | `in_p3..5` -> `out_p3..5` |
| `Head` | Two branches, `reg` and `cls`, per scale (two 3x3 `conv` operations + 1x1 `pred`), flattened per scale and output unchanged | `in_p3..5` -> `reg_p3..5` `[4·regMax, Nₚ]`, `cls_p3..5` `[numClasses, Nₚ]` |
| `DetectionDecoder` | Decodes the regression and class outputs from three scales with a single `node::Decode` kernel (DFL -> pixel boxes, `cls` -> `sigmoid`) | `reg_p3..5`, `cls_p3..5` -> `out0` `[4 + numClasses, N]` |

### Model-specific nodes

These operations are specific to YOLO. They are located in [`models/internal/yolov8/nodes/`](../../../models/internal/yolov8/nodes/) and
[`shaders/`](../../../models/internal/yolov8/shaders/).

| Node | Description | Kernel |
|---|---|---|
| `node::SPPF` | 3-stage max pooling and concat in a single kernel | `sppf.comp` |
| `node::Decode` | Decodes the DFL regression and class output to `[4 + numClasses, N]` | `decode.comp` |

### Nodes used

[`Convolution`](../06-nodes.md#convolution),
[`NearestUpsample`](../06-nodes.md#nearestupsample),
[`Concat`](../06-nodes.md#concat),
[`Reshape`](../06-nodes.md#reshape),
[`Slice`](../06-nodes.md#slice),
[`Add`](../06-nodes.md#add),
[`Identity`](../06-nodes.md#identity)

## Checkpoints

Each `tier` has its own folder under `data/ckpts/yolov8/ultralytics/`. There are no shared files such as a tokenizer,
and the demo input image is in `data/images/`.

```
data/
├── ckpts/yolov8/ultralytics/
│   ├── large/
│   │   ├── model.pt              Ultralytics original (only the downloaded tier is required)
│   │   └── model.safetensors     conversion result: file read by the loader
│   └── ...
└── images/
    └── dog.jpg                   demo input
```

Ultralytics distributes `.pt` files rather than safetensors, so these checkpoints require a one-time conversion
after downloading, unlike the other models. The weights are subject to [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

1. Download the file from [huggingface.co/Ultralytics/YOLOv8](https://huggingface.co/Ultralytics/YOLOv8) and place it at
`data/ckpts/yolov8/ultralytics/<nano|small|medium|large|xlarge>/model.pt`.

2. Convert the files. A `.pt` file stores the entire module rather than a state dict, so unpickling requires
the `ultralytics` package.

```bash
pip install ultralytics safetensors
python models/internal/yolov8/ultralytics-weights-convertor.py        # all files present
python models/internal/yolov8/ultralytics-weights-convertor.py nano large    # specify tiers
```

This creates `model.safetensors` next to each `model.pt`.

### Loading weights

The code that maps checkpoints to graph slots is in [`models/internal/yolov8/ultralytics-weights.h`](../../../models/internal/yolov8/ultralytics-weights.h).
Only one line is needed at the call site.

```cpp
loadWeights<"ultralytics">(net);
```

The path is assembled by the specialization from `model.cfg.tier` (`VAI_DATA_DIR "/yolov8/yolov8" + <tier character>`), so the caller
only needs to set the `tier`. For how distribution-name tags work, see [`distribution-name-weights.h` in Writing custom models](../contributing/models.md#distribution-weightsh). Mapping Ultralytics `model.<n>` index keys to the `backbone`, `neck`, and `head` members is the core of the loader.
Both the `conv` weights and `BN` parameters (`gamma`, `beta`, `running mean`/`var`) are read at the net's `wDefaultDtype`.
The weight loader computes BN `scale` and `bias` before passing them to `Convolution`.

## Running

### Directory structure

```
models/internal/yolov8/
├── net.h                             model and NodeGroup definitions
├── nodes/                            dedicated nodes (SPPF, Decode)
├── shaders/                          kernels for the dedicated nodes
├── ultralytics-weights.h             Ultralytics checkpoint mapping
├── ultralytics-weights-convertor.py  .pt -> .safetensors conversion
├── demo.cpp                          demo-yolov8: detection on one image + timing
├── processing.h                      pre-processing, post-processing (letterbox, NMS)
└── CMakeLists.txt

data/
├── ckpts/yolov8/ultralytics/
│   └── <tier>/model.safetensors      Tier: nano, small, medium, large, xlarge
└── images/
    └── dog.jpg
```

### Build and run

```bash
cmake -B build
cmake --build build --config RelWithDebInfo

# Run
./build/bin/demo-yolov8
```

This detects objects in `data/images/dog.jpg` using the `LARGE` `tier` and `fp16`, then prints profiles by kernel shape.
The `tier` and inference settings can be changed at the top of [`demo.cpp`](../../../models/internal/yolov8/demo.cpp).

## Benchmark summary

These results compare frameworks using **YOLOv8 Nano, 640x640, FP16**. For other model sizes and FP32 results, measurement environments and conditions (including whether preprocessing and NMS are included), and reproduction instructions,
see [`benchmarks/yolov8/README.md`](../../../benchmarks/yolov8/README.md).

| framework | 890M (Strix Point) | 8060S (Strix Halo) | RTX 5070 Ti | RX 9070 XT |
|---|---:|---:|---:|---:|
| vAi (Vulkan) | 3.238 | 0.759 | 0.616 | 0.657 |
| PyTorch (CUDA, ROCm) | 5.254 | 5.220 | 1.336 | 4.081 |
| ncnn (Vulkan) | 12.035 | 2.223 | 1.763 | 1.707 |
| ExecuTorch (Vulkan) | 27.517 | 6.151 | 3.926 | 5.729 |
| MIGraphX (HIP) | 3.527 | 0.806 | - | 0.683 |
| TensorRT (CUDA) | - | - | 0.356 | - |
| ONNX Runtime (CUDA, ROCm) | 3.433 | 0.883 | 1.484 | 0.704 |

Mean latency (ms). `-` indicates a device unsupported by that framework or OS.

For a summary across all models, see [Combined benchmarks in Supported models](README.md#combined-benchmarks).
