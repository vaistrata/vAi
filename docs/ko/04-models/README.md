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

gpt2::Net net({ .device = device, .wDefaultDtype = Dtype::Fp16 },
              gpt2::ModelConfig::_124M());

loadWeights<"openai">(net);
```

[`vai/engine/weights.h`](../../../vai/engine/weights.h)가 `loadWeights<Name, Model>`를 선언만 해두고 각 모델이 배포명별 특수화를 제공합니다.

새로운 모델 작성은 [커스텀 모델 작성](../contributing/models.md) 참조.

## 목록

### 비전

| 모델 | 모델 크기 |
|---|---|
| [YOLOX](yolox.md) | SMALL(s), MEDIUM(m), LARGE(l), XLARGE(x) |

### LLM

| 모델 | 지원 모델 크기 |
|---|---|
| [GPT-2](gpt2.md) | 124M, 355M, 774M, 1.5B |
| [Llama 3](llama3.md) | 3.2-1B, 3.2-3B |
