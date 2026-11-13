# 제공 모델

vAi로 구현한 모델들입니다.

```
models/<impl>/<model>/
├── net.h                        모델 정의 및 NodeGroup 정의
├── <배포명>-weights.h             배포 체크포인트 → 그래프 슬롯 매핑 (loadWeights<"배포명">)
├── nodes/                        모델 전용 노드 (있는 경우)
├── shaders/                      전용 노드의 커널 (있는 경우)
├── demo.cpp                      demo-<model> - 모델 추론 및 추론 시간 측정
└── CMakeLists.txt

data/ckpts/<model>/<배포명>/      체크포인트
data/tokenizers/<model>/          토크나이저
data/images/                      입력 이미지
```

## 가중치 로딩

```cpp
// GPT-2 예시
#include "net.h"
#include "openai-weights.h"

namespace gpt2 = model::production::gpt2;

gpt2::Net net({ .device = device, .defaultDtype = Dtype::Fp16 },
              gpt2::ModelConfig::_124M());

loadWeights<"openai">(net);
```

[`vai/engine/weights.h`](../../../vai/engine/weights.h)가 `loadWeights<Name, Model>`를 선언만 해두고 각 모델이 배포명별 특수화를 제공합니다.

새로운 모델 작성은 [커스텀 모델 작성](../contributing/models.md) 참조.

## 목록

### 비전

| 모델 | 모델 크기 |
|---|---|
| [YOLOv8](yolov8.md) | NANO(n), SMALL(s), MEDIUM(m), LARGE(l), XLARGE(x) |
| [YOLOX](yolox.md) | SMALL(s), MEDIUM(m), LARGE(l), XLARGE(x) |

### LLM

| 모델 | 지원 모델 크기 |
|---|---|
| [GPT-2](gpt2.md) | 124M, 355M, 774M, 1.5B |
| [Llama 3](llama3.md) | 3.2-1B, 3.2-3B |

## 통합 벤치마크

### 비전

vAi 측정값, **640x640 · FP16** 기준.

| 모델 | 지표 | 890M (Strix Point) | 8060S (Strix Halo) | RTX 5070 Ti | RX 9070 XT |
|---|---|---:|---:|---:|---:|
| YOLOv8n | 지연 (ms) | 3.238 | 0.759 | 0.616 | 0.657 |

### LLM

vAi 측정값, **prompt 512 토큰** 기준.

| 모델 | 지표 | 890M (Strix Point) | 8060S (Strix Halo) | UHD Graphics (TGL GT1) | RTX 3060 Laptop | RTX 5070 | RTX 4090 | RTX 5090 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 124M | prefill (t/s) | 10,899 | 50,777 | 1,201 | 61,236 | 154,785 | 269,911 | 308,638 |
|  | decode (t/s) | 130.5 | 500.9 | 94.06 | 743.2 | 1,319 | 1,631 | 1,756 |
| Llama 3 3.2-1B | prefill (t/s) | 1,653 | 4,932 | 123.1 | 8,761 | 25,313 | 51,332 | 56,603 |
|  | decode (t/s) | 16.46 | 81.66 | 9.63 | 115.3 | 213.2 | 311.0 | 455.4 |
| Llama 3 3.2-3B | prefill (t/s) | - | 1,726 | - | - | 9,091 | 17,219 | 21,013 |
|  | decode (t/s) | - | 33.87 | - | - | 87.74 | 128.0 | 194.0 |

다른 프레임워크 대비 결과는 [YOLOv8의 벤치마크 요약](yolov8.md#벤치마크-요약), [YOLOX의 벤치마크 요약](yolox.md#벤치마크-요약), [GPT-2의 벤치마크 요약](gpt2.md#벤치마크-요약), [Llama 3의 벤치마크 요약](llama3.md#벤치마크-요약)에 있습니다. 측정 환경과 조건, 재현 방법은 [`benchmarks/`](../../../benchmarks/) 참조.
