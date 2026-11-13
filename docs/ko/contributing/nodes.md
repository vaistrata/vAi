# 커스텀 노드와 커널 작성

일반적으로 `Node`는 그래프의 계산 단위입니다. `Node`는 필요한 `Tensor`를 준비하고 GPU 명령을 기록합니다.

GPU 명령에는 실행할 커널, 커널이 읽고 쓸 버퍼, 처리할 데이터 범위 등이 포함됩니다. 이 명령은 커맨드버퍼(`CommandBuffer`)라는 객체에 기록됩니다.  vAi 엔진은 이 커맨드버퍼를 GPU의 실행 큐에 전달합니다. GPU는 전달받은 명령에 따라 커널을 실행합니다.

새 GPU 연산을 추가할 때는 C++ `Node` 코드와 GLSL 커널 코드를 작성합니다. `Node` 코드는 `Tensor`를 준비하고 커널 실행에 필요한 설정과 명령을 기록합니다. 커널 코드는 GPU가 버퍼의 값을 읽고, 계산한 뒤, 결과를 쓰는 방법을 구현합니다. `Node`가 지정하는 `Tensor`의 형상과 dtype, 버퍼의 연결 번호, 처리할 데이터 범위는 커널의 선언과 계산 방식에 맞춰야 합니다.

구현하기에 앞서 먼저 [제공 노드](../06-nodes.md)를 확인합니다. 기존 `Node`의 조합으로 만들 수 있는 연산은 [그룹](../03-usage/node-group.md)으로 구성하면 됩니다. 새 GPU 계산이 필요하면 이 문서의 순서로 `Node`와 커널을 작성합니다. 물론 `Node` 하나에서 여러 커널을 실행할 수도 있습니다. 이번 장에서는 [`Activation` `Node`](../../../vai/nodes/activation.h)와 [activation.comp 커널](../../../vai/shaders/activation.comp)의 구현 내용을 따라 가며 커스텀 노드와 커널 작성법을 설명합니다. 활성화 함수는 ReLU를 선택하고, 별도 출력에 결과를 쓰는 경로를 기준으로 설명합니다.


## 1. 연산의 입력과 출력 정하기

먼저 계산할 값과 지원할 입력 조건을 정합니다. 이 조건은 `Node`의 입력 검사, 커널 구현, 결과 검증에 함께 사용합니다.

| 항목 | 정할 내용 | `Activation` 예시 |
|---|---|---|
| 계산 | 입력에서 출력을 구하는 방법 | ReLU: 각 원소에 `max(x, 0)` 적용 |
| 슬롯 이름 | 그래프에서 연결할 입력과 출력 | 입력 `in0`, 출력 `out0` |
| 형상 | 각 축의 의미와 지원 크기 | 입력과 출력의 형상이 같음 |
| dtype | 지원하는 입력과 출력의 저장 형식 | FP32와 FP16 |
| 설정 | `Node`를 만들 때 정할 값 | 활성화 함수 `epilogue`, 제자리 연산 `inPlace`, 출력 dtype `oDtype` |

축 수, 차원 크기, 정렬 조건에 제한이 있으면 함께 적습니다. 예를 들어 마지막 차원만 정규화하는 연산은 그 축과 가중치의 크기를 명시합니다.

설정은 `Node`를 만들 때 정합니다. 실제 입력의 형상과 `dtype`은 `prepare()`에서 읽습니다. 출력 `dtype`은 `Node`의 `oDtype` 설정으로 정합니다. `Activation`의 `oDtype` 기본값은 `None`으로 설정되어 있는데 이렇게 별도로 지정하지 않으면 `oDefaultDtype()`으로 엔진의 기본 출력 `dtype`을 가져와 사용합니다. 사용자가 `FP16` 등을 직접 지정한다면 그 값을 사용하게 됩니다.

다음 코드는 ReLU를 선택한 `Node`를 모델의 멤버로 선언하는 작성 예시입니다. `inPlace`는 기본값인 `false`를 사용하도록 했습니다.
```cpp
vai::node::Activation activation{{.epilogue = vai::Epilogue::Relu}};
```

## 2. 노드에서 텐서 준비하기

### 2.1 생성자의 슬롯 등록과 설정 보관

커스텀 노드 클래스는 `vai::Node`를 `public`으로 상속하고, `prepare()`와 `run()`을 구현합니다. `Node` 헤더는 `vai/nodes/node-common.h`를 포함합니다. `Node`를 모델의 멤버로 생성하고 그래프에 연결하는 방법은 [커스텀 모델 작성](models.md)을 참조하면 됩니다.

다음은 `Activation`의 선언과 생성자에서 발췌한 코드입니다. 생성자는 설정을 보관하고, 입력과 출력을 받을 슬롯을 등록합니다. 이 코드는 `vai::node` 네임스페이스 안에서 확인할 수 있습니다.

```cpp
class Activation : public vai::Node
{
    // ... 파이프라인과 내부 상태
public:
    struct Config {
        vai::Epilogue epilogue = vai::Epilogue::None;
        bool inPlace = false;
        vai::Dtype oDtype = vai::Dtype::None;
    } config_;

    Activation(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

    // ...
};

inline Activation::Activation(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0",  NodeSlot::input);

    config_.oDtype = (config_.oDtype == Dtype::None) ? oDefaultDtype() : config_.oDtype;
}
```

슬롯은 용도에 맞게 등록합니다.

| 종류 | 용도 |
|---|---|
| `input` | 앞 `Node`에서 받은 `Tensor` |
| `output` | 다음 `Node`에 전달할 `Tensor` |
| `temporary` | 이 `Node`를 계산하는 동안 사용할 임시 `Tensor` |
| `persistent` | 가중치나 캐시처럼 실행 사이에 유지할 `Tensor` |

용도가 `persistent`인 `Tensor`는 생성자나 가중치 로더에서 형상과 dtype을 정합니다. 엔진은 첫 `prepare()` 전에 GPU 버퍼를 연결하고, CPU에 있는 값을 업로드합니다. 초기값이 필요한 캐시는 사용 전에 그 값을 채우도록 합니다.

<a id="prepare"></a>

### 2.2 `prepare()`의 입력 검사와 출력 설정

`prepare()`는 슬롯에서 입력을 읽고, 1절에서 정한 조건을 검사합니다. 그 입력으로 출력과 임시 `Tensor`의 형상과 dtype을 정합니다. 파이프라인 선택에 필요한 입력 조건도 이때 확인합니다. 파이프라인은 4절에서 더 설명하겠습니다.

다음은 `Activation::prepare()`의 `Tensor` 준비 부분입니다. 다음과 같이 `prepare()`을 작성하면 됩니다.

```cpp
inline void Activation::prepare()
{
    const Tensor& inTen = (*this)["in0"];
    VAI_ASSERT(inTen.validShape());

    const Dtype iDtype = inTen.dtype();
    if (config_.inPlace)
        VAI_ASSERT(config_.oDtype == iDtype);

    // ... 입력 dtype에 맞는 파이프라인 준비

    (*this)["out0"] = config_.inPlace ? inTen : Tensor(inTen.shape(), config_.oDtype);
}
```

여기서 `(*this)["슬롯 이름"]`으로 슬롯의 `Tensor`를 읽거나 대입합니다. `Tensor(shape, dtype)`처럼 `Tensor`의 조건을 정하고 이후 엔진이 출력과 임시 `Tensor`에 필요한 GPU 버퍼를 연결한 뒤 `run()`을 호출합니다. 새 출력의 버퍼는 `run()`에서 사용합니다.

기본 경로는 입력과 같은 형상의 새 출력 `Tensor`를 만듭니다.

## 3. 커널에서 값 계산하기

커널은 `.comp` 파일에 작성하는 GLSL 컴퓨트 셰이더입니다. 먼저 GPU 스레드 하나가 처리할 일을 정합니다. 원소 하나, 행 하나, 타일 하나처럼 연산에 맞는 단위를 선택합니다. `Node`가 전달하는 실행 범위도 이 단위와 맞춰야 합니다. 예시로 `activation.comp`의 작업 크기와 계산 부분을 발췌했습니다. 이 커널은 스레드 하나가 원소 하나를 처리합니다.

```glsl
#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_16bit_storage : require
#include "common/activation.h"
#include "common/spec_constants.h"

layout(local_size_x = 256) in;

// ... 버퍼와 dtype 설정 선언
layout(push_constant) uniform PushConstants { uint N; };

// ... dtype에 맞게 읽고 쓰는 readI(), readO(), writeO()
void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= N) return;

    float x = (inPlace != 0u) ? readO(i) : readI(i);
    writeO(i, applyEpilogue(x));
}
```

이 커널은 스레드 하나가 원소 하나를 계산합니다. `local_size_x = 256`은 작업 그룹 하나에 스레드 256개를 배치합니다. 각 스레드는 `gl_GlobalInvocationID.x`를 처리할 원소의 번호 `i`로 사용합니다. `N`은 전체 원소 수입니다. 마지막 그룹에는 처리할 원소가 없는 스레드가 생길 수 있으므로, 다음 조건으로 계산 범위를 벗어난 스레드를 종료합니다.

```glsl
uint i = gl_GlobalInvocationID.x;
if (i >= N) return;
```

커널은 기본 경로인 `readI()`로 입력을 읽고, `applyEpilogue()`로 계산한 뒤 `writeO()`로 출력에 씁니다. ReLU를 선택하면 `applyEpilogue(x)`는 `max(x, 0.0)`을 계산합니다. 버퍼를 읽고 쓰는 함수는 dtype에 맞는 저장 형식을 사용하면 됩니다.

여러 스레드가 같은 출력 위치를 쓰는 연산은 쓰기 충돌을 처리해야 합니다. `Activation`은 각 스레드가 서로 다른 출력 원소를 계산합니다. 따라서 충돌을 처리할 필요가 없지만 `Argmax` 커널의 경우에는 여러 스레드가 하나의 출력값을 함께 계산합니다. 여러 스레드가 같은 출력 위치에 쓰면 결과가 덮어써질 수 있습니다. 이를 막으려면 각 스레드의 중간 결과를 모아 한 스레드가 최종 값을 쓰도록 구성해야 합니다. 같은 작업 그룹에서 중간 결과를 공유할 때는 필요한 값이 모두 준비된 후 읽도록 커널 안에서 동기화시킵니다. `Argmax`는 이를 피하기 위해 중간 결과를 합친 뒤, 작업 그룹의 0번 스레드만 해당 행의 결과 `outIdx[row]`에 씁니다.

`Argmax`의 각 스레드는 자신이 찾은 최댓값과 위치를 공유 메모리에 저장합니다. `barrier()`로 필요한 값이 모두 준비된 뒤 비교합니다. 비교가 끝나면 0번 스레드만 최종 결과를 출력에 씁니다.

```glsl
sMax[tid] = bestVal;
sIdx[tid] = bestIdx;
barrier();

// ... 중간 결과를 비교합니다. 각 비교 단계에서도 barrier()를 호출합니다.

if (tid == 0u) outIdx[row] = sIdx[0];
```
GLSL의 `barrier()`는 같은 작업 그룹의 스레드를 동기화합니다. 이후 설명할 5절의 `.barrier(...)`는 앞뒤 GPU 명령의 버퍼 접근을 동기화합니다. 여기서 `barrier()`는 중간 결과를 읽을 시점을 맞추고, 마지막 `if`문이 최종 결과를 쓰는 스레드를 하나로 정합니다.

## 4. 노드와 커널의 연결 맞추기

커널이 계산하려면 사용할 버퍼, 데이터 형식, 계산 크기를 알아야 합니다. 각 정보는 다음 방법으로 전달합니다.

| 전달할 정보 | 전달 방법 | `Node`와 커널에서 맞출 내용 |
|---|---|---|
| 입력·출력·가중치 버퍼 | 디스크립터: 버퍼를 커널의 연결 번호에 지정 | `set`, `binding`, 버퍼의 역할 |
| 파이프라인을 만들 때 정할 값 | 특수화 상수 | 상수 ID, 값의 타입과 의미 |
| 명령에 넣을 작은 값 | 푸시 상수 | 필드의 순서, 크기, 바이트 위치 |

푸시 상수에는 형상, 오프셋, 작은 스칼라 값 등을 넣습니다. 특수화 상수에는 dtype이나 알고리즘 선택처럼 커널을 구성하는 값을 넣습니다. 값을 정할 시점에 따라 전달 방법을 선택합니다.

### 4.1 입력 조건별 파이프라인 요청

파이프라인은 커널과 특수화 설정을 묶은 GPU 실행 객체입니다. `pipelinePool().request()`에 커널 이름과 특수화 상수를 전달합니다. 같은 요청의 파이프라인은 풀에서 재사용합니다. `pipelinePool().request()`에는 확장자를 제외한 커널 파일 이름을 전달합니다. 예를 들어 `activation.comp`는 `"activation"`으로 요청합니다.

다음은 `Activation::prepare()`에서 파이프라인을 준비하는 부분입니다. 입력 dtype은 실제 입력에서 읽습니다. 출력 dtype은 생성자에서 정한 값을 사용합니다.

```cpp
const PipeParams p = { .iDtype = iDtype };
if (!pipe || p != pipeParams)
{
    pipeParams = p;
    pipe = pipelinePool().request("activation", {
        eva::constant_id<0>(config_.inPlace ? 1u : 0u),
        eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),
        eva::constant_id<SPEC_ID_TYPE_I>(iDtype),
        eva::constant_id<SPEC_ID_EPILOGUE>((uint32_t) config_.epilogue)
    });
    if (!descSet)
        descSet = gVaiDestSetPool(pipe.descSetLayout(0));
}
```

`PipeParams`는 이 `Node` 안에서 정의한 구조체입니다. 파이프라인 선택에 사용한 입력 dtype을 보관합니다. 그 조건이 바뀌면 파이프라인을 다시 요쳥하게 됩니다. 새 `Node`도 파이프라인 선택에 사용하는 형상이나 dtype을 빠짐없이 비교해야 합니다.

`descSet`은 커널에 연결할 버퍼 목록을 담는 디스크립터 세트입니다. 기존 `Node`처럼 공용 풀 `gVaiDestSetPool`에서 확보합니다. 예제는 같은 커널의 버퍼 구성을 사용하므로 이 목록을 재사용합니다. 다른 커널로 바꿀 때 버퍼의 타입, 개수, 연결 번호가 다르면 그 구성으로 새로 확보해야 합니다.

### 4.2 C++·GLSL 버퍼와 상수 대응

다음은 `activation.comp`에서 제자리 연산 설정, dtype, 버퍼를 받는 선언을 나타냅니다.

```glsl
layout(constant_id = 0) const uint inPlace = 0;
layout(constant_id = SPEC_ID_TYPE_O) const uint typeO = 0;
layout(constant_id = SPEC_ID_TYPE_I) const uint typeI = 0;

layout(set = 0, binding = 0) buffer O_BUF_F32 { float bufferO_f32[]; };
layout(set = 0, binding = 0) buffer O_BUF_F16 { float16_t bufferO_f16[]; };
layout(set = 0, binding = 1) buffer I_BUF_F32 { float bufferI_f32[]; };
layout(set = 0, binding = 1) buffer I_BUF_F16 { float16_t bufferI_f16[]; };
```

`Node`에서 전달한 값은 다음 위치로 연결됩니다. 번호와 타입을 함께 확인합니다.

| C++에서 지정할 값 | 커널에서 받을 위치 |
|---|---|
| `descSet.write({outBuf, inBuf})`의 첫 버퍼 | `set = 0, binding = 0`: 출력 |
| 같은 목록의 두 번째 버퍼 | `set = 0, binding = 1`: 입력 |
| `constant_id<0>(...)` | `constant_id = 0`: 제자리 연산 여부 |
| `constant_id<SPEC_ID_TYPE_O>(...)` | `constant_id = SPEC_ID_TYPE_O`: 출력 dtype |
| `constant_id<SPEC_ID_TYPE_I>(...)` | `constant_id = SPEC_ID_TYPE_I`: 입력 dtype |
| `constant_id<SPEC_ID_EPILOGUE>(...)` | `common/activation.h`의 `epilogue`: 활성화 함수 |
| 푸시 상수의 `uint32_t N` | `PushConstants`의 `uint N` |

dtype 상수는 [common/spec_constants.h](../../../vai/shaders/common/spec_constants.h)의 공용 ID와 값을 사용합니다. `activation.comp`는 같은 연결 번호에 FP32와 FP16 버퍼를 선언하고, dtype 상수로 사용할 쪽을 선택합니다. 이 방식을 사용하면 C++에서 버퍼를 연결하는 순서를 유지할 수 있습니다. 활성화 함수를 선택하는 상수는 `common/activation.h`가 선언합니다. 커널에서는 같은 ID를 다시 선언하지 않습니다.

여러 필드를 푸시 상수로 전달하면 C++ 구조체와 GLSL 블록의 바이트 배치까지 맞춰야 합니다. 특히 배열과 벡터는 선언 순서만 같아도 정렬에 따라 위치가 다를 수 있습니다.

## 5. 실행 명령 기록하기

`run()`은 준비한 버퍼와 파이프라인으로 GPU 명령을 기록합니다. GPU 계산은 엔진이 이 명령을 제출한 뒤 수행됩니다.

다음은 `Activation::run()`에서 발췌한 코드입니다.

```cpp
inline void Activation::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf  = config_.inPlace ? eva::Buffer() : (*this)["in0"].buffer();

    descSet.write({ outBuf, inBuf });

    uint32_t N = (*this)["out0"].numElements();
    struct {
        uint32_t N;
    } constants = { N };

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch2(N)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}
```

위 코드의 `.buffer()`는 `Tensor`가 사용하는 GPU 버퍼 범위를 가져옵니다. 디스크립터와 배리어에 이 범위를 그대로 전달합니다. `Tensor`가 큰 버퍼의 일부를 참조할 때도 올바른 위치를 사용할 수 있습니다.

`setPushConstants()`의 인자는 바이트 오프셋, 데이터 크기, 데이터 주소 순서입니다. 예제는 오프셋 0에 원소 수 `N`을 기록합니다.

제자리 연산이 필요하면, 모델에서 `Node`를 생성할 때 `Config`의 `inPlace`를 `true`로 지정합니다. 이 경로는 입력과 출력에 같은 `Tensor`를 사용하고, `readO()`로 읽은 위치에 결과를 씁니다. 입력도 출력 버퍼에서 읽으므로 두 번째 버퍼는 비워 둡니다. 입력과 출력 dtype이 같아야 합니다. 또한 다른 `Node`가 원래 입력값을 사용하는지도 확인합니다.

### 5.1 작업 단위와 실행 범위

`dispatch2(x, y, z)`에는 각 방향의 스레드 수를 전달합니다. 이 함수가 파이프라인의 작업 그룹 크기로 올림 나눗셈해 그룹 수를 정합니다. `dispatch(x, y, z)`에는 작업 그룹 수를 직접 전달합니다.

예시에서는 원소마다 스레드 하나가 필요하므로 `dispatch2(N)`을 사용합니다. 행이나 타일 단위로 계산하는 커널은 그 단위에 맞게 실행 범위를 정합니다.

### 5.2 커널 간 버퍼 동기화

`Activation` 커널은 계산 결과를 `outBuf`에 씁니다. 다음 커널이 이 결과를 읽기 전에, 쓰기가 완료되고 그 결과를 읽을 수 있도록 배리어(`.barrier`)를 기록합니다.

```cpp
.barrier(
    eva::SYNC_SCOPE::COMPUTE_WRITE /  // 앞 커널이 값을 씁니다.
    outBuf /                         // 동기화할 버퍼 범위입니다.
    eva::SYNC_SCOPE::COMPUTE_READ);   // 다음 커널이 값을 읽습니다.
```
한 `Node`에서 여러 커널을 실행할 때도, 앞 커널의 결과를 다음 커널이 읽는 단계 사이에 필요한 배리어를 기록합니다. 다음 작업이 계산 대신 복사라면, `COMPUTE_READ` 대신 복사 작업의 읽기에 맞는 동기화 종류를 지정해야 합니다.

### 5.3 명령 재사용과 호출별 상태 갱신

엔진은 기록한 GPU 명령을 재사용합니다. `prepare()`와 `run()`은 명령을 기록하거나 다시 기록할 때 호출합니다. 따라서 푸시 상수의 값도 기록한 명령을 재사용하는 동안 유지됩니다.

매 호출 달라지는 카운터나 상태는 별도 버퍼로 전달할 수 있습니다. CPU에서 갱신한다면 이전 GPU 작업의 완료를 기다린 뒤 모델의 실행 메서드에서 값을 갱신합니다. 이 흐름은 [모델의 상태 관리](models.md#42-실행-메서드의-상태-관리)에서 더 다루고 있습니다.

생성 후 설정을 바꾸는 API가 필요하면 기록한 명령의 갱신도 처리합니다. 명령에 넣은 값이나 버퍼 연결이 바뀌면 소속 모델의 `invalidateCommandBuffer()`로 재기록을 요청합니다. 특수화 설정이 바뀌면 파이프라인도 다시 선택합니다.

## 6. 빌드하고 결과 확인하기

### 6.1 파일 배치와 커널 빌드

| 용도 | `Node` | 커널 |
|---|---|---|
| 여러 모델에서 쓰는 연산 | `vai/nodes/<이름>.h` | `vai/shaders/<이름>.comp` |
| 특정 모델에서 쓰는 연산 | `models/<모델명>/nodes/<이름>.h` | `models/<모델명>/shaders/<이름>.comp` |

공용 커널은 [루트 빌드 설정](../../../CMakeLists.txt)이 수집해 `vAi`와 함께 빌드합니다.
`Node`에서는 예제처럼 `pipelinePool().request("커널 이름", {...})`로 요청합니다.
`Node`를 사용하는 C++ 파일에서는 해당 `Node` 헤더를 포함합니다.

모델 전용 커널은 [모델 빌드 설정](../../../models/CMakeLists.txt)이 `<모델명>-shaders` 타겟으로 빌드합니다.
`Node`에서 `<MODEL>_SHADER_OUTPUT_DIR`을 함께 전달합니다.
다음은 [YOLOv8의 SPPF `Node`](../../../models/yolov8/nodes/sppf.h)에서 발췌한 요청 코드입니다.

```cpp
pipe = pipelinePool().request(YOLOV8_SHADER_OUTPUT_DIR, "sppf", {
    eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),
    eva::constant_id<SPEC_ID_TYPE_I>(p.iDtype)
});
```

커널을 사용하는 실행 타겟에도 빌드 의존성을 추가합니다.
다음은 [YOLOv8 데모의 빌드 설정](../../../examples/yolov8/CMakeLists.txt)에 있는 코드입니다.
여기서 `${PROJECT_NAME}`은 `demo-yolov8`입니다.

```cmake
add_dependencies(${PROJECT_NAME} yolov8-shaders)
```

여러 커널이 공유하는 GLSL 코드는 `vai/shaders/common/`에 둡니다.
이 경로의 헤더는 `#include "common/<이름>.h"`로 읽습니다.
공용 헤더의 버퍼 선언이나 상수 구성을 바꾸면 이를 사용하는 커널도 함께 확인합니다.

제출 전에 다음 규칙을 확인해주세요.

- 지원하는 형상과 dtype 조합으로 실행합니다.
- 최소 크기와 작업 그룹 크기의 배수가 아닌 입력을 확인합니다. 예제 커널은 원소 수 1, 256, 257로 경계를 확인할 수 있습니다.
- 선택한 계산식의 경계값을 확인합니다. ReLU는 음수, 0, 양수를 입력해 결과를 비교합니다.
- 같은 그래프를 반복 실행하고, 입력 형상을 바꾼 뒤에도 결과를 비교합니다.
- 출력 공유, 제자리 연산, 임시 `Tensor`가 있으면 해당 경로도 확인합니다.
- 비교한 입력, 기준 구현, dtype, 허용 오차, 실행 방법을 기록합니다.

`Node`를 추가하는 PR에는 Python 바인딩과 테스트를 함께 넣습니다. 둘 중 하나가 없으면 PR의 검사가 실패합니다. 작성 방법은 [노드를 추가할 때 함께 넣는 것](README.md#6-노드를-추가할-때-함께-넣는-것)을 참조하면 됩니다.

### 6.2 코드와 주석 작성 규칙

| 항목 | 규칙 |
|---|---|
| `Node` 선언 | `vai::Node`를 `public`으로 상속합니다. `class`를 사용할 수 있습니다. |
| 네임스페이스 | 기본 제공 `Node`는 `vai::node`에 둡니다. 모델 전용 `Node`는 모델의 네임스페이스에 둡니다. |
| C++ 이름과 형식 | 타입은 `PascalCase`, 함수와 변수는 `camelCase`로 씁니다. 함수의 여는 중괄호는 다음 줄에 둡니다. [공통 코드 컨벤션](../index.md#3-공통-코드-컨벤션)을 따릅니다. |
| 헤더 정의 | 헤더 가드를 둡니다. 클래스 밖에 정의한 일반 멤버 함수에는 `inline`을 붙입니다. |
| `Node` 주석 | 이름, 계산, 입력·출력·가중치의 슬롯과 형상, 지원 조건을 적습니다. |
| 커널 주석 | 계산식, 버퍼 역할, 특수화 상수의 의미, 실행 단위를 적습니다. 커널의 나머지 코드 형식은 구현에 맞게 선택합니다. |

`Node`의 설명 주석은 다음 형식을 사용합니다.
첫 줄은 실제 클래스 이름과 요약입니다. 가중치가 있으면 `Params:` 항목도 추가합니다.
다음은 `Activation`의 주석에서 입출력 부분을 발췌한 예시입니다.

```cpp
/*=============================================================================
  Activation - Applies an element-wise activation.
  Input: in0 [any shape]
  Output: epilogue(in0); optionally in-place

  // ... 지원 조건과 동작 설명
=============================================================================*/
```

커널도 같은 구분선의 블록 주석을 사용합니다.
다음은 `activation.comp`의 주석에서 발췌한 예시입니다.

```glsl
/*=============================================================================
 activation.comp:
     O[i] = epilogue(I[i])     (element-wise; epilogue via spec constant)

 // ... 옵션 설명
 - O (typeO) and I (typeI) are fp32 or fp16, accessed through same-binding
   buffer aliases. Dispatch (host): dispatch2(N).
=============================================================================*/
```
