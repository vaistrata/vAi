# 개요

vAi는 **그래프 DSL 기반 Vulkan 추론 엔진**입니다. 모델 그래프를 C++ DSL(연산 노드를 잇는 전용 문법)로 조립하고 Vulkan의 Compute Shader로 실행합니다.

## 1. 특징

### 1.1 이식성

단 하나의 코드로 여러 GPU에서 바로 실행하고 테스트할 수 있습니다. 업계 표준 GPU API인 Vulkan 하나만 쓰기 때문에 NVIDIA, AMD, INTEL, Qualcomm 등 여러 GPU 벤더마다 전용 SDK에 맞춰 코드를 재작성할 필요가 없습니다.

### 1.2 경량성

vAi로 LLM을 배포하면 런타임 바이너리가 2MB 남짓입니다. on-device 환경을 고려해 의존성 체인을 설계하였습니다.실제로 사용하는 모델 코드와 연산자 커널만 빌드에 들어가는 셀렉티브 빌드가 별도 설정 없이 동작합니다.

### 1.3 고성능

입력 형상과 버퍼 바인딩 등 조건이 같으면 GPU 커맨드 버퍼를 재사용합니다. 이는 추론에 CPU의 자원을 거의 사용하지 않습니다. 보통의 추론 엔진들은 매 추론마다 연산 그래프를 다시 순회하며 커널 실행 명령을 반복적으로 GPU에 제출하지만 vAi는 커맨드 버퍼를 처음 한번 만들어 두고 재사용합니다. 또한 GPU 커널 튜닝으로 실제로 도는 커널은 장비마다 달라 이식성을 얻는 대가로 성능을 내주지 않아도 됩니다.

### 1.4 직관성

그래프 DSL 기반으로 모델의 구조가 코드에 그대로 드러납니다. vAi는 전용 그래프 문법과 객체지향 설계를 사용하여 레고를 맞추듯 `Node`, `NodeGroup`을 이어 붙이면 원하는 모델을 쉽게 완성할 수 있습니다.

```cpp
template<TransformerType layout, QkvLayout qkv>
struct TransformerLayer : public vai::NodeGroup
{
    vai::node::Identity input;
    typename BranchNorm<layout>::Type norm1;
    MultiHeadAttention<qkv> mhAttn;
    std::optional<vai::node::Slice> lastRow;
    std::optional<vai::node::Gate> gate;
    typename BranchNorm<layout>::Type norm2;
    typename BranchFfn<layout>::Type ffn;

    TransformerLayer(...)
    {
        input - norm1 -    mhAttn - lastRow - gate/"then" - norm2 -    ffn;
        input - "residual"/mhAttn - lastRow - gate/"then" - "residual"/ffn;
    }
};
```

멤버 목록이 곧 블록의 구성이고 배선이 곧 데이터 흐름입니다. 생성자의 첫 줄인 주 경로와 두 번째 줄인 잔차 경로가 실제 설계도처럼 그려집니다. 코드가 곧 모델의 구조도라 가독성이 좋고 수정하기 쉽습니다.

### 1.5 확장성

새로운 연산자나 나만의 최적화가 들어간 레이어를 직접 만들어 넣을 수 있습니다. vAi는 코어 코드를 **단 한 줄도 수정하지 않고** `Node` 클래스를 상속해 원하는 커널만 작성하면 됩니다. 원본 프로젝트에 병합되기를 기다리지 않고도 새로 만든 커널을 필요한 사람에게 그대로 넘길 수 있습니다. 포크를 유지할 필요도 플러그인 규격에 맞출 필요도 없습니다.

## 2. 프로젝트 구조

```
vai/                        엔진 본체 (라이브러리 vAi)
  engine/
    neural-net.h
    neural-net.cpp           그래프 정렬·버퍼 준비·커맨드 기록
    graph-flow.h             연결 DSL과 NodeGroup
    graph-controller.h       입력·조건 분기에 따른 실행 경로
    tensor.h
    safetensors-parser.h
    weights.h
    buffer-pool.h
    pipeline-pool.h
    error.h
  nodes/                    기본 제공 노드들
  groups/                   기본 제공 그룹들
  shaders/                  기본 제공 커널들 (.comp)
  utils/

data/                       모델별 체크포인트·입력 데이터 등

models/                     제공 모델들
  <impl>/<모델명>/
    net.h                     모델 정의
    <배포명>-weights.h         체크포인트 로더
    demo.cpp                  demo-<모델명>

examples/                   모델 데모 밖의 프로그램들 (예: mnist/, chat-llama3/)
python/                     Python 바인딩 (pyvai)
benchmarks/
external/eva/               Vulkan 래퍼 (서브모듈)
docs/
```

처음에는 [빠른 실행](02-quickstart.md)으로 MNIST를 돌린 뒤 [노드](03-usage/node.md)와 [노드그룹](03-usage/node-group.md)을 읽으세요. 그래프 실행과 데이터 표현은 이어지는 [NeuralNet](03-usage/neural-net.md)과 [Tensor](03-usage/tensor.md)에서 다룹니다.


| 수정 대상 | 문서 및 코드 위치 |
|---|---|
| 모델의 구성·배선 | [MNIST 예제](../../examples/mnist/main.cpp), [제공 모델](../../models/), [커스텀 모델 작성](contributing/models.md) |
| 여러 모델이 공유하는 블록 | [공통 그룹](../../vai/groups/), [노드그룹](03-usage/node-group.md) , [커스텀 그룹 작성](contributing/groups.md) |
| 연산의 형상 처리·GPU 계산 | [노드](../../vai/nodes/)와 [셰이더](../../vai/shaders/), [커스텀 노드와 커널 작성](contributing/nodes.md) |
| 그래프 실행·버퍼 관리 | [neural-net.cpp](../../vai/engine/neural-net.cpp), [실행 파이프라인](03-usage/execution-pipeline.md) |

## 3. 공통 코드 컨벤션

**3.1 CamelCase**: 타입은 대문자로 시작하고, 함수와 변수는 소문자로 시작합니다.

```cpp
class Linear;            // 타입
void addSlot(...);       // 함수
uint32_t numQHeads;       // 변수
```

대상별 규칙은 다음과 같습니다.

| 대상 | 규칙 | 예 |
|---|---|---|
| 클래스 · 구조체 · `enum` · 타입 별칭 · `concept` | `PascalCase` | `NeuralNet`, `PipeParams` |
| 함수 · 멤버 함수 | `camelCase` | `loadWeights()`, `setProfileTag()` |
| 변수 · 매개변수 | `camelCase` | `outShape`, `targetDtype` |
| `public` 멤버 변수 | `camelCase` | `acceptResidual`, `iDtype` |
| `private` · `protected` 멤버 변수 | `camelCase` 뒤에 `_` | `device_`, `cmdBuffer_` |
| `enum`의 값 | `PascalCase` | `Fp16`, `Bilinear` |
| 상수 | `camelCase`. 앞에 `k`를 붙이지 않습니다. | `tokensPerSide` |
| 네임스페이스 | 소문자 | `vai`, `node`, `detail` |
| 매크로 | `UPPER_CASE` | `VAI_ASSERT` |

다음 이름은 위 규칙의 예외입니다.

- 텐서의 차원은 수식의 문자를 그대로 씁니다. (예: `N`, `K`, `Co`, `Dh`, `Tmax`.)
- 모델의 크기별 설정 함수는 `_`로 시작합니다. 이름이 숫자로 시작할 수 없기 때문입니다. (예: `_124M()`, `_3B_224()`.)
- `Node`의 `public` 멤버 `config_`는 `_`를 붙입니다. 생성자의 매개변수 `config`와 구분하기 위한 이름입니다.

**3.2 여는 중괄호**: 대상에 따라 다음을 따릅니다.

| 대상 | 여는 중괄호 |
|---|---|
| 일반 함수 · 멤버 함수 | 다음 줄 (Allman) |
| `if` · `for` · `while` · `switch` 블록 | 다음 줄 (Allman) |
| 구조체 · 클래스 선언 | 다음 줄 (Allman) |
| 람다 함수 · `namespace` · `enum` | 같은 줄 |

```cpp
namespace vai::node {

enum class Mode { Fast, Exact };

class Linear : public vai::Node
{
public:
    void prepare()
    {
        if (config_.acceptResidual)
            addResidual();
        auto scaled = [&](uint32_t x) { return x * factor_; };
    }
};

}   // namespace vai::node
```

**3.3 namespace**: 엔진은 `vai`를 네임스페이스로 사용하고, 공통 노드와 노드그룹은 `vai::node`, `vai::group`을 사용합니다.
모델과 모델 전용 노드, 노드그룹은 `model::<impl>::<모델명>` 아래에 둡니다. `<impl>`은 `production` 또는 `experimental`입니다.

```cpp
namespace vai
namespace vai::node
namespace vai::group

// YOLOX의 경우
namespace model::experimental::yolox
namespace model::experimental::yolox::node
namespace model::experimental::yolox::group
```

**3.4 자동 검사**: 3.1과 3.2는 저장소 루트의 `.clang-format`과 `.clang-tidy`에 설정되어 있고, 커밋할 때와 PR을 올릴 때 자동으로 검사합니다. 줄 길이와 들여쓰기 같은 나머지 형식, 검사를 통과하는 방법은 [기여 절차와 자동 검사](contributing/README.md)에서 다룹니다.
