# YOLOX

YOLOX(Megvii, 2021)은 단일 패스 객체 검출 모델입니다. Focus를 입력단에 둔 CSPDarknet 백본,
YOLOPAFPN 넥, 회귀와 분류를 분리한 anchor-free 검출 헤드로 이루어지며 세 스케일(stride 8, 16, 32)에서
박스, objectness, 클래스 확률을 뽑습니다. COCO 80클래스 체크포인트를 씁니다.

지원 범위는 Small, Medium, Large, XLarge입니다. YOLOX-Nano는 Depthwise Conv(`groups = in_channels`)를 사용하므로 현재 지원에서 제외하며, 추후 지원 예정입니다.

**참고자료**: [Repository](https://github.com/Megvii-BaseDetection/YOLOX),
[Paper](https://arxiv.org/abs/2107.08430),
[Official checkpoints](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0)

## 설정

```cpp
namespace yolox = model::experimental::yolox;

auto cfg = yolox::getConfigCOCO(yolox::Tier::LARGE);

yolox::Net net({
    .device        = device,
    .wDefaultDtype = Dtype::Fp16,
}, cfg, {
    .imageSize     = 640,
    .confThreshold = 0.25f,
    .iouThreshold  = 0.45f,
});
```

### NeuralNetConfig

YOLOX는 기본값을 그대로 씁니다. conv 가중치와 BN 파라미터 모두 `wDefaultDtype`(fp16)으로
저장됩니다. BN의 `scale`·`bias`를 가중치 로더가 CPU에서 계산하므로 별도 정밀도 설정이 없습니다.
세부사항은 [vAi 사용법 > NeuralNet](../03-usage/neural-net.md) 참조.

### ModelConfig

`Tier`가 기본 채널 폭(`baseChannels`)과 반복 수(`baseDepth`)를 정하고,
`getConfigCOCO(tier)`가 여기에 클래스 80개를 더해 `ModelConfig`를 만듭니다.

| `Tier` | `baseChannels` | `baseDepth` | 체크포인트 |
|---|---:|---:|---|
| `SMALL` | 32 | 1 | `yolox_s.pth` |
| `MEDIUM` | 48 | 2 | `yolox_m.pth` |
| `LARGE` | 64 | 3 | `yolox_l.pth` |
| `XLARGE` | 80 | 4 | `yolox_x.pth` |

백본의 채널 폭은 stem부터 `baseChannels`의 1, 2, 4, 8, 16배입니다.
`dark2`~`dark5`의 CSP 반복 수는 각각 `baseDepth`의 1, 3, 3, 1배이며,
넥의 CSP 반복 수는 `baseDepth`입니다. `numClasses` 80은 COCO 체크포인트 공통입니다.

### RuntimeConfig

입력 해상도와 후처리 임계값입니다. 640은 학습 해상도라 가장 잘 맞지만 다른 값도 넣을 수 있습니다.

| 필드 | 기본값 |
|---|---|
| `imageSize` | `640` |
| `confThreshold` | `0.25` |
| `iouThreshold` | `0.45` |

## vAi 그래프

vAi에서 모델은 노드를 `-`로 이어 만든 DAG입니다. 세부사항은 [vAi 사용법 > 노드](../03-usage/node.md)와 [노드그룹](../03-usage/node-group.md) 문서를 참조.

Focus를 처리하는 stem은 전용 `node::FocusConv`가, 나머지 conv는 빌트인
[`Convolution`](../06-nodes.md#convolution)가 맡습니다. Conv-BN-SiLU 블록은
가중치 로더가 CPU에서 BN의 `scale`·`bias`를 합성해 슬롯에 넣고, 노드는
채워진 슬롯과 SiLU 옵션을 사용합니다. 검출 헤드의 prediction은 bias가 있는 1x1 conv입니다.

*`H`와 `W`: 입력 해상도, `N`: 전 스케일 검출 위치 수 (640x640이면 8,400).*

### 모델 아키텍처

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

출력은 `[8400, 85]` 형태입니다. 각 행은 `cx, cy, width, height, objectness, class[0..79]`이며,
박스 좌표는 grid와 stride가 적용된 픽셀 단위입니다. objectness와 클래스 값은 sigmoid를 거친 확률입니다.
NMS는 그래프 밖 후처리([`models/experimental/yolox/processing.h`](../../../models/experimental/yolox/processing.h))에서 합니다.
비교 프레임워크의 배치 1 출력 `[1, 8400, 85]`와 같은 내용을 담습니다.

### NodeGroup

YOLOX를 구현하며 정의한 [노드그룹](../03-usage/node-group.md)입니다. `Backbone`, `Neck`, `Head`,
`DetectionDecoder` 네 개가 최상위이고 그 안에 `FocusModule`, `Bottleneck`, `CSPLayer`,
`SPPBottleneck`과 단계별 그룹이 중첩됩니다.

#### FocusModule

공식 Focus는 입력의 짝수·홀수 행과 열을 나눠 채널 방향으로 쌓은 뒤 3x3 conv를 수행합니다.
vAi는 이를 `node::FocusConv` 하나로 처리합니다. 중간 Focus 텐서를 만들지 않고 원본 입력에서
바로 Conv-BN-SiLU를 계산하며, `[C, H, W]`를 `[Co, H/2, W/2]`로 바꿉니다.

#### Bottleneck

1x1 conv와 3x3 conv를 잇는 블록입니다. `shortcut`이 켜져 있고 입력·출력 채널 수가 같으면
두 번째 conv가 입력을 `residual`로 받아 더합니다.

```cpp
input - conv1 - conv2;
input - "residual"/conv2;  // shortcut이고 채널 수가 같을 때만
```

#### CSPLayer

입력을 두 개의 1x1 conv로 나눕니다. 첫 번째 가지는 `Bottleneck` 사슬을 통과하고,
두 번째 가지와 채널 방향으로 합친 뒤 1x1 conv로 출력 채널 수를 맞춥니다.

```cpp
input - conv1 - bottlenecks[0];
for (uint32_t i = 1; i < repeats; ++i)
    bottlenecks[i - 1] - bottlenecks[i];
bottlenecks.back() - "in0"/concat;
input - conv2     - "in1"/concat;
concat - conv3;
```

#### SPPBottleneck

공식 SPP의 병렬 5x5, 9x9, 13x13 max pooling을 연속 5x5 pooling 세 번으로 계산합니다.
전용 `node::SPPF`가 pooling과 입력을 포함한 네 텐서의 concat을 한 커널로 처리합니다.

```cpp
conv1 - sppf - conv2;
```

#### Backbone, Neck, Head

| 그룹 | 내용 | 슬롯 |
|---|---|---|
| `Backbone` | Focus stem 뒤 stride-2 conv + CSP 네 단(`dark2`~`dark5`), 마지막 단에 SPP 추가 | `in0` → `out_p3`, `out_p4`, `out_p5` |
| `Neck` | YOLOPAFPN: top-down FPN(upsample + concat + CSP) 뒤 bottom-up PAN(stride-2 conv + concat + CSP) | `in_p3..5` → `out_p3..5` |
| `Head` | 스케일마다 1x1 stem 뒤 회귀·objectness와 분류 두 갈래(3x3 conv 둘 + 1x1 pred) | `in_p3..5` → `regobj_p3..5` `[5, Hₚ, Wₚ]`, `cls_p3..5` `[numClasses, Hₚ, Wₚ]` |
| `DetectionDecoder` | 세 스케일의 출력을 직접 읽어 grid/stride decode와 sigmoid 수행 | `regobj_p3..5`, `cls_p3..5` → `out0` `[N, 5 + numClasses]` |

회귀 4채널과 objectness 1채널 prediction은 하나의 5채널 conv로 계산합니다.
`node::Decode`는 `regobj`와 `cls`를 직접 읽으므로 스케일별 `[85, Hₚ, Wₚ]` concat 버퍼가 필요 없습니다.

### 전용 노드

YOLOX에 필요한 연산입니다. [`models/experimental/yolox/nodes/`](../../../models/experimental/yolox/nodes/)와
[`shaders/`](../../../models/experimental/yolox/shaders/)에 있습니다.

| 노드 | 내용 | 커널 |
|---|---|---|
| `node::FocusConv` | Focus와 stem Conv-BN-SiLU 융합 | `focus-conv.comp`, `focus-conv-cm.comp` |
| `node::SPPF` | 3단 max pooling과 concat을 한 커널로 | `sppf.comp` |
| `node::Decode` | 회귀·objectness와 클래스 출력을 `[N, 5 + numClasses]`로 디코드 | `decode.comp` |

`node::FocusConv`의 FP16 cooperative-matrix 경로는 초기 1회
`focus-conv-transform.comp`로 가중치를 변환합니다.

### 사용 노드

[`Convolution`](../06-nodes.md#convolution),
[`NearestUpsample`](../06-nodes.md#nearestupsample),
[`Concat`](../06-nodes.md#concat),
[`Identity`](../06-nodes.md#identity)

## 체크포인트

체크포인트는 `data/ckpts/yolox/megvii/` 아래 tier별 폴더에 둡니다.
데모 입력 이미지는 `data/images/`에 있습니다.

```text
data/
├── ckpts/yolox/megvii/
│   ├── large/
│   │   ├── model.pth             Megvii 원본
│   │   └── model.safetensors     변환 결과: 로더가 읽는 파일
│   └── ...
└── images/
    └── dog.jpg                   데모 입력
```

Megvii가 배포하는 `.pth`를 받은 뒤 `.safetensors`로 변환합니다.

**1.** [공식 release](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0)에서
필요한 체크포인트를 받아 `data/ckpts/yolox/megvii/<small|medium|large|xlarge>/model.pth`로 둡니다. Large 기준:

```bash
mkdir -p data/ckpts/yolox/megvii/large
curl -L -o data/ckpts/yolox/megvii/large/model.pth \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth
```

**2.** 변환합니다. 체크포인트의 `model` state dict를 읽으므로 이 변환에는 YOLOX 패키지가 필요 없습니다.

```bash
pip install torch safetensors
python models/experimental/yolox/megvii-weights-convertor.py        # 있는 것 전부
python models/experimental/yolox/megvii-weights-convertor.py small large    # tier 지정
```

각 `model.pth` 옆에 `model.safetensors`가 생깁니다.

### 가중치 로딩

체크포인트를 그래프 슬롯에 매핑하는 코드는
[`models/experimental/yolox/megvii-weights.h`](../../../models/experimental/yolox/megvii-weights.h)에 있습니다.
실행하는 쪽은 한 줄이면 됩니다.

```cpp
loadWeights<"megvii">(net);
```

경로는 `net.cfg.tier`로 특수화가 조립하므로(`VAI_DATA_DIR "/ckpts/yolox/megvii/" + tierName(net.cfg.tier)`)
부르는 쪽은 tier만 정하면 됩니다. 배포명 태그의 원리는 [제공 모델 > 가중치 로딩](README.md#가중치-로딩) 참조.

공식 state dict의 `backbone.backbone`, `backbone`, `head` 키를 각각 백본, 넥, 헤드의 멤버로 매핑합니다.
conv 가중치와 BN 파라미터는 넷의 `wDefaultDtype`으로 읽으며, 헤드의 회귀와 objectness 가중치·bias는
합쳐서 `regObjPred`에 넣습니다.

## 실행

### 폴더 구조

```text
models/experimental/yolox/
├── net.h                       모델 정의 및 NodeGroup 정의
├── nodes/                      전용 노드 (FocusConv, SPPF, Decode)
├── shaders/                    전용 노드의 커널
├── megvii-weights.h             Megvii 체크포인트 매핑
├── megvii-weights-convertor.py  .pth → .safetensors 변환
├── demo.cpp                    demo-yolox: 이미지 한 장 검출 + 시간 측정
├── processing.h                전처리, 후처리 (letterbox, NMS)
└── CMakeLists.txt

data/
├── ckpts/yolox/megvii/
│   └── <tier>/model.safetensors      Tier: small, medium, large, xlarge
└── images/
    └── dog.jpg
```

### 빌드 및 실행

```bash
cmake -B build
cmake --build build --config RelWithDebInfo --target demo-yolox

# Run
./build/bin/demo-yolox
```

`data/images/dog.jpg`를 LARGE tier, fp16으로 검출하고 커널 형태별 프로파일을 찍습니다.
tier와 추론 설정은 [`demo.cpp`](../../../models/experimental/yolox/demo.cpp)의 상단에서 바꿉니다.

전처리는 RGB 이미지 로더의 결과를 BGR CHW로 바꾸고, 비율을 유지해 좌상단부터 배치하며
남는 영역은 114로 채웁니다. 입력은 0–255 float이고 `1/255` 정규화를 하지 않습니다.

## 벤치마크 요약

YOLOX Large FP16, 640x640 기준 평균 추론 지연시간입니다. 측정 환경과 조건(전처리, NMS 포함
여부), 재현 방법은 [`benchmarks/yolox/README.md`](../../../benchmarks/yolox/README.md) 참조.

| | 890M (Strix Point) | 8060S (Strix Halo) | RTX 5070 | RTX 3060 Laptop |
|---|---|---|---|---|
| vAi (ms) | TBD | TBD | TBD | TBD |
| ncnn (ms) | TBD | TBD | TBD | TBD |
| OpenVINO (ms) | TBD | TBD | TBD | TBD |


전체 모델 요약은 [제공 모델](README.md#통합-벤치마크) 참조.
