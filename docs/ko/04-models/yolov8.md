# YOLOv8

YOLOv8(Ultralytics, 2023)은 단일 패스 객체 검출 모델입니다. CSPDarknet 백본, PANet 넥, DFL(Distribution Focal Loss) 기반 anchor-free 검출 헤드로 이루어지며 세 스케일(stride 8, 16, 32)에서 박스와 클래스 확률을 뽑습니다. COCO 80클래스 체크포인트를 씁니다.

**참고자료**: [저장소](https://github.com/ultralytics/ultralytics), [체크포인트(HuggingFace)](https://huggingface.co/Ultralytics/YOLOv8), [문서](https://docs.ultralytics.com/models/production/yolov8/)

## 설정

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

YOLOv8는 기본값을 그대로 씁니다. `conv` 가중치와 `BN` 파라미터 모두 `defaultDtype`(`fp16`)으로
저장됩니다. BN의 `scale`·`bias`를 가중치 로더가 CPU에서 계산하므로 별도 정밀도 설정이 없습니다.
세부사항은 [신경망](../03-usage/neural-net.md) 참조.

### ModelConfig

`Tier`가 채널 폭(`p1`~`p5`)과 `C2f` 반복 수를 정하고 `getConfigCOCO(tier)`가 여기에 클래스 80개와
`regMax` 16을 얹어 `ModelConfig`를 만듭니다.

| `Tier` | `p1` | `p2` | `p3` | `p4` | `p5` | `C2f` 반복 | 체크포인트 |
|---|---|---|---|---|---|---|---|
| `NANO` | 16 | 32 | 64 | 128 | 256 | 1 | `yolov8n` |
| `SMALL` | 32 | 64 | 128 | 256 | 512 | 1 | `yolov8s` |
| `MEDIUM` | 48 | 96 | 192 | 384 | 576 | 2 | `yolov8m` |
| `LARGE` | 64 | 128 | 256 | 512 | 512 | 3 | `yolov8l` |
| `XLARGE` | 80 | 160 | 320 | 640 | 640 | 3 | `yolov8x` |

`numClasses` 80, `regMax` 16은 COCO 체크포인트 공통입니다.

### RuntimeConfig

입력 해상도와 후처리 임계값입니다. 640은 학습 해상도라 가장 잘 맞지만 다른 값도 넣을 수 있습니다.

| 필드 | 기본값 |
|---|---|
| `imageSize` | `640` |
| `confThreshold` | `0.25` |
| `iouThreshold` | `0.45` |

## vAi 그래프

vAi에서 모델은 노드를 `-`로 이어 만든 DAG입니다. 세부사항은 [노드](../03-usage/node.md)와 [노드그룹](../03-usage/node-group.md) 문서를 참조.

검출 백본의 conv는 전부 빌트인 [`Convolution`](../06-nodes.md#convolution)가 맡습니다.
가중치 로더가 CPU에서 BN의 `scale`·`bias`를 합성해 슬롯에 넣습니다. 노드는
채워진 슬롯과 SiLU 옵션으로 커널 형태에 따라 Winograd, implicit GEMM, GEMM
경로를 스스로 고릅니다. 모델 폴더에 남은 전용 노드는 YOLO에만 있는 연산 셋뿐입니다.

*`H`와 `W`: 입력 해상도, `N`: 전 스케일 앵커 수 (640x640이면 8,400).*

### 모델 아키텍처

```cpp
input("in0") - backbone;
backbone/"out_p3" - "in_p3"/neck/"out_p3" - "in_p3"/head;
backbone/"out_p4" - "in_p4"/neck/"out_p4" - "in_p4"/head;
backbone/"out_p5" - "in_p5"/neck/"out_p5" - "in_p5"/head;

head/"reg_p3" - "reg_p3"/detector;   // reg/cls × p3/p4/p5 여섯 슬롯
head/"cls_p3" - "cls_p3"/detector;   // (p4, p5 동일)
detector - output("out0");           // [4 + numClasses, N]
```

출력은 `[84, 8400]` 형태의 raw 텐서(박스 4행 + 클래스별 확률 80행)입니다. NMS는 GPU 밖
후처리([`models/internal/yolov8/processing.h`](../../../models/internal/yolov8/processing.h))에서 합니다.
ncnn, ONNX Runtime 등 비교 프레임워크의 YOLOv8 출력과 같은 형태라 벤치에서 후처리를 공유합니다.

### NodeGroup

YOLOv8를 구현하며 정의한 [노드그룹](../03-usage/node-group.md)입니다. `Backbone`, `Neck`, `Head`,
`DetectionDecoder` 네 개가 최상위이고 그 안에 `Bottleneck`, `C2fModule`, `SPPFModule`이 중첩됩니다.

#### Bottleneck

3x3 `conv` 둘을 잇는 잔차 블록입니다. `shortcut`이 켜지면 `conv2`가 입력을 `residual`로 받아 더하고, 꺼지면 잔차 없이 통과합니다.

```cpp
input - conv1 - conv2;
input - "residual"/conv2;   // shortcut일 때만
```

#### C2fModule

CSP 블록입니다. 1x1 `conv`로 넓힌 뒤 채널 뒷절반만(`Slice`) `Bottleneck` 사슬에 태우고
입력과 각 단계 출력을 전부 쌓아 1x1 `conv`로 되돌립니다.

```cpp
convIn                                        - "in0"/concat;
convIn - lastHalfChannels - bottlenecks[0]    - "in1"/concat;
bottlenecks[i-2] - bottlenecks[i-1]           - "in<i>"/concat;   // i = 2..n
concat - convOut;
```

#### SPPFModule

Spatial Pyramid Pooling Fast입니다. 3단 최대 풀링과 concat을 전용 노드 `node::SPPF` 한 커널로 처리합니다.

```cpp
convIn - sppf - convOut;
```

#### Backbone, Neck, Head

| 그룹 | 내용 | 슬롯 |
|---|---|---|
| `Backbone` | CSPDarknet: stride-2 `conv` 하나(`p1`) 뒤 (stride-2 `conv` + `C2f`) 네 단(`p2`~`p5`), 끝에 SPPF | `in0` → `out_p3`, `out_p4`, `out_p5` |
| `Neck` | PANet: top-down FPN([`NearestUpsample`](../06-nodes.md#nearestupsample) + concat + `C2f`) 뒤 bottom-up PAN(stride-2 `conv` + concat + `C2f`) | `in_p3..5` → `out_p3..5` |
| `Head` | 스케일마다 `reg`와 `cls` 두 갈래(3x3 `conv` 둘 + 1x1 `pred`), 스케일별로 평탄화해 그대로 내보냄 | `in_p3..5` → `reg_p3..5` `[4·regMax, Nₚ]`, `cls_p3..5` `[numClasses, Nₚ]` |
| `DetectionDecoder` | 세 스케일의 회귀와 클래스 출력을 `node::Decode` 한 커널로 디코드 (DFL → 픽셀 박스, `cls` → `sigmoid`) | `reg_p3..5`, `cls_p3..5` → `out0` `[4 + numClasses, N]` |

### 전용 노드

YOLO에만 있는 연산입니다. [`models/internal/yolov8/nodes/`](../../../models/internal/yolov8/nodes/)와
[`shaders/`](../../../models/internal/yolov8/shaders/)에 있습니다.

| 노드 | 내용 | 커널 |
|---|---|---|
| `node::SPPF` | 3단 최대 풀링과 concat을 한 커널로 | `sppf.comp` |
| `node::Decode` | DFL 회귀와 클래스 출력을 `[4 + numClasses, N]`로 디코드 | `decode.comp` |

### 사용 노드

[`Convolution`](../06-nodes.md#convolution),
[`NearestUpsample`](../06-nodes.md#nearestupsample),
[`Concat`](../06-nodes.md#concat),
[`Reshape`](../06-nodes.md#reshape),
[`Slice`](../06-nodes.md#slice),
[`Add`](../06-nodes.md#add),
[`Identity`](../06-nodes.md#identity)

## 체크포인트

체크포인트는 `data/ckpts/yolov8/ultralytics/` 아래 tier별 폴더에 둡니다.
데모 입력 이미지는 `data/images/`에 있습니다.

```
data/
├── ckpts/yolov8/ultralytics/
│   ├── large/
│   │   ├── model.pt              Ultralytics 원본
│   │   └── model.safetensors     변환 결과: 로더가 읽는 파일
│   └── ...
└── images/
    └── dog.jpg                   데모 입력
```

Ultralytics가 배포하는 것은 safetensors가 아니라 `.pt`라 다른 모델과 달리 받은 뒤 한 번 변환해야
합니다. 가중치는 [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)을 따릅니다.

1. [huggingface.co/Ultralytics/YOLOv8](https://huggingface.co/Ultralytics/YOLOv8)에서 받아
`data/ckpts/yolov8/ultralytics/<nano|small|medium|large|xlarge>/model.pt`로 둡니다.

2. 변환합니다. `.pt`는 state dict가 아니라 모듈째 저장된 형식이라 언피클에 `ultralytics` 패키지가
필요합니다.

```bash
pip install ultralytics safetensors
python models/internal/yolov8/ultralytics-weights-convertor.py        # 있는 것 전부
python models/internal/yolov8/ultralytics-weights-convertor.py nano large    # tier 지정
```

각 `model.pt` 옆에 `model.safetensors`가 생깁니다.

### 가중치 로딩

체크포인트를 그래프 슬롯에 매핑하는 코드는 [`models/internal/yolov8/ultralytics-weights.h`](../../../models/internal/yolov8/ultralytics-weights.h)에
있습니다. 실행하는 쪽은 한 줄이면 됩니다.

```cpp
loadWeights<"ultralytics">(net);
```

경로는 `model.cfg.tier`로 특수화가 조립하므로(`VAI_DATA_DIR "/yolov8/yolov8" + <tier 글자>`) 부르는
쪽은 `tier`만 정하면 됩니다. 배포명 태그의 원리는 [커스텀 모델 작성의 `배포명-weights.h`](../05-contributing-models.md#배포명-weightsh) 참조. Ultralytics의 `model.<n>` 인덱스 키를 `backbone`, `neck`, `head` 멤버로 옮기는 매핑이 로더의
본체입니다. `conv` 가중치와 `BN` 파라미터(`gamma`, `beta`, `running mean`/`var`)를 모두 넷의 `wDefaultDtype`으로
읽습니다. 가중치 로더가 BN의 `scale`·`bias`를 계산해 `Convolution`에 전달합니다.

## 실행

### 폴더 구조

```
models/internal/yolov8/
├── net.h                             모델 정의 및 NodeGroup 정의
├── nodes/                            전용 노드 (SPPF, Decode)
├── shaders/                          전용 노드의 커널
├── ultralytics-weights.h             Ultralytics 체크포인트 매핑
├── ultralytics-weights-convertor.py  .pt -> .safetensors 변환
├── demo.cpp                          demo-yolov8: 이미지 한 장 검출 + 시간 측정
├── processing.h                      전처리, 후처리 (letterbox, NMS)
└── CMakeLists.txt

data/
├── ckpts/yolov8/ultralytics/
│   └── <tier>/model.safetensors      Tier: nano, small, medium, large, xlarge
└── images/
    └── dog.jpg
```

### 빌드 및 실행

```bash
cmake -B build
cmake --build build --config RelWithDebInfo

# Run
./build/bin/demo-yolov8
```

`data/images/dog.jpg`를 `LARGE` `tier`, `fp16`으로 검출하고 커널 형태별 프로파일을 찍습니다.
`tier`와 추론 설정은 [`demo.cpp`](../../../models/internal/yolov8/demo.cpp) 상단에서 바꿉니다.

## 벤치마크 요약

비교 프레임워크 대비 **YOLOv8 Nano · 640x640 · FP16** 기준입니다. 다른 모델 크기와 FP32 결과, 측정 환경과 조건(전처리, NMS 포함 여부), 재현 방법은
[`benchmarks/yolov8/README.md`](../../../benchmarks/yolov8/README.md) 참조.

| framework | 890M (Strix Point) | 8060S (Strix Halo) | RTX 5070 Ti | RX 9070 XT |
|---|---:|---:|---:|---:|
| vAi (Vulkan) | 3.238 | 0.759 | 0.616 | 0.657 |
| PyTorch (CUDA, ROCm) | 5.254 | 5.220 | 1.336 | 4.081 |
| ncnn (Vulkan) | 12.035 | 2.223 | 1.763 | 1.707 |
| ExecuTorch (Vulkan) | 27.517 | 6.151 | 3.926 | 5.729 |
| MIGraphX (HIP) | 3.527 | 0.806 | - | 0.683 |
| TensorRT (CUDA) | - | - | 0.356 | - |
| ONNX Runtime (CUDA, ROCm) | 3.433 | 0.883 | 1.484 | 0.704 |

평균 지연(ms). `-`는 해당 프레임워크나 OS가 지원하지 않는 장치입니다.

전체 모델 요약은 [제공 모델의 통합 벤치마크](README.md#통합-벤치마크) 참조.
