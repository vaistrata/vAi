# 커스텀 노드와 커널 작성

일반적으로 노드​는 뉴럴 네트워크에서 하나의 연산을 나타내며, GPU 관점에서는 하나의 실행 단위에 해당합니다.

각 노드는 해당 연산을 수행하는 GPU 커널​을 실행하며, 커널 실행에 필요한 인자와 버퍼를 바인딩하고 실제 실행을 제어하는 역할을 담당합니다. 즉, 쉽게 말하면 노드는 하나의 연산과 그 연산을 GPU에서 실행하기 위한 제어를 담당하는 단위라고 볼 수 있습니다.

기본적으로 지원되지 않는 연산이 필요한 경우에는 본 섹션을 참고하여 직접 구현할 수 있습니다. 본 섹션은 간단한 `activation.comp`, `activation.h` 를 예시를 들어 설명하고 있습니다.

```
vai/nodes/vai-<name>.h        노드 (헤더 온리)
vai/shaders/vai-<name>.comp   커널
```

특정 모델에서만 쓰는 노드라면 모델 폴더에 두어도 됩니다.
[`models/internal/yolov8/nodes/`](../../models/internal/yolov8/nodes/)이 그 사례입니다.

## 코드 컨벤션

**컨벤션 통일이 최우선입니다**. 기존 네이밍, 패턴, 헬퍼에 맞추고 예외를
만들지 않습니다. 단, 커널은 헤드 스탬프만 지키면 되고, 나머지 코드 컨벤션은
강요되지 않습니다.

### 헤드 스탬프

노드와 커널 모두 `/* ... */` 블록 주석과 `=` 펜스로 여닫습니다. 노드 스탬프는
[제공 노드들](06-nodes.md)의 생성 원본이므로 형식을 지켜야 합니다.

```cpp
/*=============================================================================
  RmsNorm - RMS normalization (normalize over last dimension without mean centering)
  Input : in0 [..., D]
  Params: scale [D]
  Output: out0 [..., D] = in0 / rms(in0) * scale

=============================================================================*/
```

## 커스텀 커널 작성

### 커널

커널은 `.comp` 파일, 곧 Vulkan 컴퓨트 셰이더입니다.

### 기본 구조

```glsl
/*=============================================================================
 activation.comp:
     O[i] = epilogue(I[i])     (element-wise; epilogue via spec constant)

 - epilogue : 0 none, 1 relu, 2 gelu, 3 silu, 4 sigmoid
 - inPlace : when 1, reads from and writes to the output buffer (in0 unbound)
 - O (typeO) and I (typeI) are fp32 or fp16, accessed through same-binding
   buffer aliases. Dispatch (host): dispatch2(N).
=============================================================================*/
#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_16bit_storage : require
#include "common/activation.h"
#include "common/spec_constants.h"

layout(local_size_x = 256) in;

layout(constant_id = 0) const uint inPlace = 0;   // 0: separate out, 1: in-place
layout(constant_id = SPEC_ID_TYPE_O) const uint typeO = 0;  // O (output) dtype
layout(constant_id = SPEC_ID_TYPE_I) const uint typeI = 0;  // I (input) dtype

layout(set = 0, binding = 0) buffer O_BUF_F32 { float bufferO_f32[]; };
layout(set = 0, binding = 0) buffer O_BUF_F16 { float16_t bufferO_f16[]; };
layout(set = 0, binding = 1) buffer I_BUF_F32 { float bufferI_f32[]; };
layout(set = 0, binding = 1) buffer I_BUF_F16 { float16_t bufferI_f16[]; };

layout(push_constant) uniform PushConstants { uint N; };

float readI(uint i) { return (typeI == 0u) ? bufferI_f32[i] : float(bufferI_f16[i]); }
float readO(uint i) { return (typeO == 0u) ? bufferO_f32[i] : float(bufferO_f16[i]); }
void writeO(uint i, float v) { if (typeO == 0u) bufferO_f32[i] = v; else bufferO_f16[i] = float16_t(v); }

void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= N) return;

    float x = (inPlace != 0u) ? readO(i) : readI(i);
    writeO(i, applyEpilogue(x));
}
```

헤드 스탬프에는 수식, 워크그룹 배치, 특수화 상수의 의미 등 호스트가 어떻게 디스패치해야 하는지를
적습니다. 커널을 읽는 사람이 호스트 코드를 열지 않아도 되게 하는 것이 목적입니다.

`layout(local_size_x = 256) in;`은 하나의 워크그룹을 구성하는 local invocation의 개수와 차원별 배치를 지정합니다. 이 경우 워크그룹 크기는 (256, 1, 1)이며, 하나의 워크그룹은 총 256개의 invocation으로 구성됩니다.

### 공용 셰이더 헤더

여러 커널에서 함께 쓰는 GLSL 선언과 함수는 `vai/shaders/common/`에 둡니다.
공용 헤더는 독립된 커널이 아니라서 직접 컴파일되지 않고 각 `.comp` 파일에
텍스트로 들어갑니다.

```glsl
layout(constant_id = 0) const uint inPlace = 0;   // 0: separate out, 1: in-place
layout(constant_id = SPEC_ID_TYPE_O) const uint typeO = 0;  // O (output) dtype
layout(constant_id = SPEC_ID_TYPE_I) const uint typeI = 0;  // I (input) dtype
```

glslc의 `#include` 기준 경로는 `vai/shaders/`입니다. 따라서 커널의 실제 위치와
관계없이 `common/<파일>.h` 형식으로 참조합니다.

공용 헤더에는 실제로 여러 커널이 공유하는 코드만 둡니다. 디스크립터 바인딩,
푸시 상수, 특수화 상수를 선언하면 호스트와 맞물리는 셰이더 ABI가 됩니다. 공용 헤더를
바꾸면 그 헤더를 `#include`하는 모든 커널을 다시 검증해야 합니다.

### 특수화 상수 - 커널
Vulkan에서 커널은 일반적으로 GLSL → SPIR-V → GPU ISA 과정을 거쳐 컴파일됩니다.

```glsl
#extension GL_GOOGLE_include_directive : require // 이 extension이 있어야 #include가 가능합니다.
#include "common/spec_constants.h"
```

특수화 상수는 SPIR-V에 특수화 가능한 상수로 포함되며, 파이프라인 생성 시 실제 값이 결정됩니다. 따라서 드라이버가 최종 GPU 코드를 생성할 때 해당 값을 상수로 활용할 수 있으며, 데드 코드 제거, 상수 전파, 루프 전개 등의 최적화를 수행하기 쉬워집니다.

### 디스크립터 - 커널

동일한 역할을 하는 버퍼는 데이터 타입이 다르더라도 동일한 디스크립터 바인딩을 공유하도록 구성할 수 있습니다. 실제로 사용할 데이터 타입은 `activation` 커널에서는 특수화 상수를 통해 선택합니다.

위 예시에서 입력 데이터의 정밀도가 `float`인 경우 `typeI` 특수화 상수에 0을 지정하여 `bufferI_f32`에서 값을 읽고, `float16_t`인 경우 1을 지정하여 `bufferI_f16`에서 값을 읽도록 합니다.

### 푸시 상수 - 커널

푸시 상수는 형상, 오프셋, 작은 스칼라 파라미터처럼 실행 매개변수 값을
전달합니다.

빌드는 자동입니다. `vai/shaders/`에 두면 glslc가 컴파일합니다. `vai/shaders/`가
`#include` 경로에 있어 `common/*.h`를 같은 경로로 참조할 수 있습니다.


## 커스텀 노드 작성

`vai::Node`를 상속해 `prepare()`와 `run()`을 구현합니다.

```cpp
class Activation : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct Config {
        vai::Epilogue epilogue = vai::Epilogue::None;
        bool inPlace = false;
        vai::Dtype iDtype = vai::Dtype::None;
        vai::Dtype oDtype = vai::Dtype::None;
    } config_;

    Activation(Config options);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;
};
```

### 생성자

노드 그래프에서 사용할 출력 슬롯 `out0`과 입력 슬롯 `in0`을 `addSlot()`을 통해 등록합니다.

```cpp
inline Activation::Activation(Config options)
: config_(options)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0",  NodeSlot::input);

    config_.iDtype = (config_.iDtype == Dtype::None) ? ioDtypeHint() : config_.iDtype;
    config_.oDtype = (config_.oDtype == Dtype::None) ? ioDtypeHint() : config_.oDtype;
    if (config_.inPlace)   // in-place reads and writes the same buffer
        VAI_ASSERT(config_.iDtype == config_.oDtype);

    pipe = pipelinePool().request("activation", {
        eva::constant_id<0>(config_.inPlace ? 1u : 0u),      // inPlace
        eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),    // typeO
        eva::constant_id<SPEC_ID_TYPE_I>(config_.iDtype),    // typeI
        eva::constant_id<SPEC_ID_EPILOGUE>((uint32_t) config_.epilogue)   // epilogue
    });

    descSet = gVaiDestSetPool(pipe.descSetLayout(0));
}
```


#### 특수화 상수 - 노드
생성자에서 특수화 상수 값을 정의합니다.`pipelinePool().request()`의 두 번째 인자로 커널에게 값을
전달합니다. 이때 노드의 `constant_id` 값과 [특수화 상수 - 커널](#특수화-상수---커널)의 특수화 상수 `constant_id` 값이 일치해야 합니다.

### prepare()

```cpp
inline void Activation::prepare()
{
    const Tensor& inTen = (*this)["in0"];
    VAI_ASSERT(inTen.dtype() == config_.iDtype);
    VAI_ASSERT(inTen.validShape());

    (*this)["out0"] = config_.inPlace ? inTen : Tensor(inTen.shape(), config_.oDtype);
}
```

입력에 대해서 검증하고, 출력 및 중간 텐서의 형상과 데이터 타입을 결정합니다.

### run()

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

#### 디스크립터 셋 - 노드
`descSet.write()`를 통해 입력 및 출력 버퍼를 커널의 디스크립터에 바인딩합니다. 이때 각 버퍼가 [디스크립터 - 커널](#디스크립터---커널)에 선언된 바인딩과 일치하도록 연결해야 합니다.

#### 푸시 상수 - 노드
`setPushConstants()`를 사용하여 푸시 상수 값을 커맨드 버퍼에 기록합니다. 첫 번째 인자는 오프셋, 두 번째 인자는 기록할 데이터의 바이트 크기, 세 번째 인자는 데이터가 저장된 메모리의 주소입니다.

노드에서 전달하는 데이터의 크기, 순서 및 레이아웃은 [푸시 상수 - 커널](#푸시-상수---커널)에 선언된 푸시 상수 블록과 일치해야 합니다.

#### 디스패치
디스패치 크기는 셰이더가 처리하는 단위와 일치해야 합니다. `activation` 커널은 전체 원소 수를 `dispatch2(N)`에 전달합니다.

```glsl
void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= N) return;

    // output 계산
}
```

각 invocation은 `gl_GlobalInvocationID.x`가 전체 원소 수 `N` 이상인지 확인하고, 유효 범위를 벗어난 경우 즉시 종료합니다.

#### 동기화

디스패치 뒤의 배리어 커맨드는 현재 커널의 출력이 다음 노드의
입력에 보이도록 합니다.

Vulkan에서는 하나의 큐에 여러 커널을 제출했을때, CUDA 처럼 직렬적이지 않고 병렬적으로 실행되기 때문에 실행 순서와 메모리 가시성을 보장하기 위해서 명시적 동기화가 필요합니다. 이를 위해서 배리어 커맨드를 사용합니다.

### 전달 방식 비교

| 전달 방식 | 전달 시점 | 대표 값 |
|---|---|---|
| 특수화 상수 | `pipelinePool().request()` | 데이터 타입, 워크그룹 크기, 알고리즘 분기 |
| 푸시 상수 | `run()`의 `setPushConstants()` | 형상, 오프셋, `eps` 같은 스칼라 |
| 디스크립터 | `descSet.write()` | 입력, 출력, 가중치와 큰 파라미터 |
