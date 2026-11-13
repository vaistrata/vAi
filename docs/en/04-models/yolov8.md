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

All convolutions in the detection backbone use the built-in [`Convolution`](../06-nodes.md#convolution).
The weight loader computes BN `scale` and `bias` on the CPU and passes them to the node. Given
`hasScale`, `hasBias`, and the SiLU option, the node selects the Winograd, implicit GEMM, or GEMM
path according to the kernel shape. The model folder contains only the node implementations for YOLO-specific operations.

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

The output is a raw tensor of shape `[84, 8400]` (4 box rows + 80 class probability rows). NMS runs outside the GPU
in postprocessing ([`examples/yolov8/processing.h`](../../../examples/yolov8/processing.h)).
This matches the YOLOv8 output format of comparison frameworks such as ncnn and ONNX Runtime, so the benchmarks share the same postprocessing.

### NodeGroup

These [node groups](../03-usage/node-group.md) are defined for the YOLOv8 implementation. The four top-level groups are `Backbone`, `Neck`, `Head`,
and `DetectionDecoder`, with `Bottleneck`, `C2fModule`, and `SPPFModule` nested inside them.

#### Bottleneck

This residual block connects two 3x3 `conv` nodes. When `shortcut` is disabled, it passes through without a residual.

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

This is Spatial Pyramid Pooling Fast. The model-specific `SPPFNode` handles three stages of max pooling and concatenation in a single kernel.

```cpp
convIn - sppf - convOut;
```

#### Backbone, Neck, Head

| Group | Description | Slots |
|---|---|---|
| `Backbone` | CSPDarknet: one stride-2 `conv` (`p1`), then four stages of stride-2 `conv` + `C2f` (`p2` through `p5`), ending with SPPF | `in0` -> `out_p3`, `out_p4`, `out_p5` |
| `Neck` | PANet: top-down FPN ([`NearestUpsample`](../06-nodes.md#nearestupsample) + concat + `C2f`), then bottom-up PAN (stride-2 `conv` + concat + `C2f`) | `in_p3..5` -> `out_p3..5` |
| `Head` | Two branches, `reg` and `cls`, per scale (two 3x3 `conv` nodes + 1x1 `pred`); flattens each scale and outputs it directly | `in_p3..5` -> `reg_p3..5` `[4*regMax, Nₚ]`, `cls_p3..5` `[numClasses, Nₚ]` |
| `DetectionDecoder` | Decodes regression and class outputs from all three scales in one `DecodeNode` kernel (DFL -> pixel boxes, `cls` -> `sigmoid`) | `reg_p3..5`, `cls_p3..5` -> `out0` `[4 + numClasses, N]` |

### Model-specific nodes

These operations are specific to YOLO. Their implementations are in [`models/yolov8/nodes/`](../../../models/yolov8/nodes/) and
[`shaders/`](../../../models/yolov8/shaders/).

| Node | Description | Kernel |
|---|---|---|
| `SPPFNode` | Three stages of max pooling and concatenation in one kernel | `sppf.comp` |
| `DecodeNode` | Decodes DFL regression and class outputs into `[4 + numClasses, N]` | `decode.comp` |

### Nodes used

[`Convolution`](../06-nodes.md#convolution),
[`NearestUpsample`](../06-nodes.md#nearestupsample),
[`Concat`](../06-nodes.md#concat),
[`Reshape`](../06-nodes.md#reshape),
[`Slice`](../06-nodes.md#slice),
[`Add`](../06-nodes.md#add),
[`Identity`](../06-nodes.md#identity)

## Checkpoints

Place checkpoints under `data/yolov8/`. The `yolov8<tier>` portion of the filename identifies the model size.
All tiers share the demo input image.

```
data/yolov8/
├── dog.jpg                     Shared demo input
├── yolov8l.pt                  Ultralytics original
├── yolov8l.safetensors         Conversion output: the file read by the loader
└── ...
```

Ultralytics distributes `.pt` files rather than safetensors, so these checkpoints require a one-time conversion
after downloading, unlike the other models. The weights are subject to [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

1. Download from [huggingface.co/Ultralytics/YOLOv8](https://huggingface.co/Ultralytics/YOLOv8) and save as
`data/yolov8/yolov8<n|s|m|l|x>.pt`.

2. Convert the files. A `.pt` file stores the entire module rather than a state dict, so unpickling requires
the `ultralytics` package.

```bash
pip install ultralytics safetensors
python models/yolov8/ultralytics-weights-convertor.py        # all available files
python models/yolov8/ultralytics-weights-convertor.py n l    # select tiers
```

This creates `yolov8<tier>.safetensors` in the same directory.

### Loading weights

The code that maps checkpoint tensors to graph slots is in
[`models/yolov8/ultralytics-weights.h`](../../../models/yolov8/ultralytics-weights.h). Loading takes one line at the call site.

```cpp
loadWeights<"ultralytics">(net);
```

The specialization constructs the path from `model.cfg.tier` (`VAI_DATA_DIR "/yolov8/yolov8" + <tier letter>`),
so the caller only selects `tier`. For distribution tags, see [`distribution-weights.h` in Writing custom models](../contributing/models.md#distribution-weightsh). The loader's main task is mapping Ultralytics `model.<n>` index keys to the `backbone`, `neck`, and `head` members.
It reads both `conv` weights and `BN` parameters (`gamma`, `beta`, `running mean`/`var`) in the node's `wDtype`.
The weight loader computes BN `scale` and `bias` and passes them to `Convolution`.

## Running

### Directory structure

```
models/yolov8/
├── model.h                           Model and NodeGroup definitions
├── nodes/                            Model-specific nodes (SPPF, Decode)
├── shaders/                          Kernels for model-specific nodes
├── ultralytics-weights.h              Ultralytics checkpoint mapping
└── ultralytics-weights-convertor.py    .pt -> .safetensors conversion

examples/yolov8/
├── demo.cpp                          demo-yolov8: detect objects in one image and measure timing
├── processing.h  processing.cpp      Preprocessing and postprocessing (letterbox, NMS)
└── CMakeLists.txt

data/yolov8/
├── dog.jpg
├── yolov8n.safetensors                NANO
├── yolov8s.safetensors                SMALL
├── yolov8m.safetensors                MEDIUM
├── yolov8l.safetensors                LARGE
└── yolov8x.safetensors                XLARGE
```

### Build and run

```bash
cmake -B build
cmake --build build --config RelWithDebInfo

# Run
./build/bin/demo-yolov8
```

The demo runs detection on `data/yolov8/dog.jpg` with the `LARGE` `tier` and `fp16`, then prints a profile by kernel type.
Change `tier` and inference settings at the top of [`demo.cpp`](../../../examples/yolov8/demo.cpp).

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
