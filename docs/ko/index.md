# 개요

vAi는 **그래프 DSL 기반 Vulkan 추론 엔진**입니다. 모델 그래프를 C++ DSL(연산 노드를 잇는 전용 문법)로 조립하고 Vulkan의 Compute Shader로 실행합니다.

## 특징

### 1. 이식성

단 하나의 코드로 여러 GPU와 SoC에서 바로 실행하고 테스트할 수 있습니다. 업계 표준 Vulkan 하나만 쓰기 때문에 NVIDIA, AMD, 인텔, 퀄컴마다 전용 SDK에 맞춰 코드를 처음부터 다시 짤 필요가 없습니다. 장비를 바꿔가며 새로 구현하던 일이 vAi에서는 빌드 한 번으로 끝납니다.

### 2. 경량성

vAi로 LLM을 배포하면 런타임 바이너리가 2MB 남짓입니다. 처음부터 엣지 디바이스를 겨냥해 의존성 체인을 세밀하게 다듬었습니다. 그래서 실제로 쓰는 로직과 연산자 커널만 빌드에 들어가는 셀렉티브 빌드가 별도 설정 없이 기본으로 동작합니다.

### 3. 고성능

추론 한 번에 CPU가 하는 일이 거의 없습니다. 보통의 추론 엔진은 매 추론마다 연산 그래프를 다시 훑으며 커널 실행 명령을 그때그때 GPU에 제출하지만 vAi는 명령서를 처음 한 번 만들어 두고 계속 재사용합니다. 또한 GPU 커널 튜닝으로 실제로 도는 커널은 장비마다 달라 이식성을 얻는 대가로 성능을 내주지 않아도 됩니다.

### 4. 유연성

비전 모델과 LLM을 한 런타임 내에서 함께 다룰 수 있습니다. 보통 비전 모델은 ONNX Runtime, ncnn, ExecuTorch 같은 범용 추론 엔진으로 처리하고, LLM처럼 별도의 효율화가 필요한 모델은 llama.cpp, vLLM, TensorRT-LLM 같은 전용 프레임워크를 따로 얹습니다. vAi는 다형성을 활용해 KV 캐시처럼 매번 갱신되는 상태와 그 전이 방식을 연산자마다 스스로 정의합니다. 그러므로 런타임을 하나 더 쌓지 않고도 비전, LLM, VLM, VLA를 모두 소화할 수 있습니다.

### 5. 직관성

그래프 DSL 기반으로 모델의 구조가 코드에 그대로 드러납니다. vAi는 전용 그래프 문법과 객체지향 설계를 사용하여 레고를 맞추듯 노드를 이어 붙이면 원하는 모델을 쉽게 완성할 수 있습니다. `Llama3`의 트랜스포머 블록은 다음과 같이 적을 수 있습니다.

```cpp
struct TransformerBlock : public vai::NodeGroup {
    vai::node::Identity input;
    vai::node::RmsNorm norm1;
    MultiHeadAttention mhAttn;
    vai::node::Add add1;
    vai::node::RmsNorm norm2;
    GatedFeedForward ffn;
    vai::node::Add add2;

    TransformerBlock(...)
    {
        input - norm1 - mhAttn - "in0"/add1 - norm2 - ffn - "in0"/add2;
        input                  - "in1"/add1               - "in1"/add2;
    }
};
```

멤버 목록이 곧 블록의 구성이고 배선이 곧 데이터 흐름입니다. 생성자의 첫 줄인 주 경로와 두 번째 줄인 잔차 경로가 실제 설계도처럼 그려집니다. 코드가 곧 모델의 구조도라 가독성이 좋고 수정하기 쉽습니다.

### 6. 확장성

새로운 연산자나 나만의 최적화가 들어간 레이어를 직접 만들어 넣을 수 있습니다. vAi는 코어 코드를 **단 한 줄도 수정하지 않고** 노드 클래스를 상속해 원하는 커널만 작성하면 됩니다. 원본 프로젝트에 병합되기를 기다리지 않고도 새로 만든 커널을 필요한 사람에게 그대로 넘길 수 있습니다. 포크를 유지할 필요도 플러그인 규격에 맞출 필요도 없습니다.

## 프로젝트 구조

```
vai/                        엔진 본체 (라이브러리 vAi)
  engine/
    neural-net.h
    tensor.h
    safetensors-parser.h
    weights.h
    buffer-pool.h
    pipeline-pool.h
    error.h
  nodes/                    기본 제공 노드들
  shaders/                  기본 제공 커널들 (.comp)
  utils/

data/                       모델별 체크포인트·입력 데이터 등

models/                     제공 모델들
  <impl>/<모델명>/
    net.h                     모델 정의
    <배포명>-weights.h         체크포인트 로더
    demo.cpp                  demo-<모델명>

examples/                   모델 데모 밖의 프로그램들 (예: mnist/, chat-llama3/)
benchmarks/
tests/
external/eva/               Vulkan 래퍼 (서브모듈)
docs/
```

## 공통 코드 컨벤션

**1. CamelCase**: 타입은 대문자로 시작하고, 함수와 변수는 소문자로 시작합니다.

```cpp
class Linear;        // 타입
void addSlot(...);       // 함수
uint32_t numHeads;       // 변수
```

**2. 여는 중괄호**: 대상에 따라 다음을 따릅니다.

| 대상 | 여는 중괄호 |
|---|---|
| `if` · `for` 블록 | Allman |
| 일반 함수 · 멤버 함수 | Allman |
| 람다 함수 | 자유 |
| 구조체 · 클래스 선언 | 자유 |

```cpp
// Allman 스타일
void foo()
{
    ...
}
```
