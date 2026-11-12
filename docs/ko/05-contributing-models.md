# 커스텀 모델 작성

> **커스텀 모델은 원하는 대로 자유롭게 만들 수 있습니다**. 기여자분들이 만드신 모델이 공식 배포판에 포함되기 위해서는 아래 컨벤션과 규칙에 맞춰서 모델을 작성해주시고 PR을 보내주셔야 합니다.

모델은 `Node`를 연결한 그래프입니다. 체크포인트 로더는 각 `Node`의 가중치 슬롯을 채웁니다. 이번 장에서는 작성하고자 하는 모델의 그래프 구현부터 실행 확인까지의 과정에 대해서 설명드립니다. 이번 장에서는 예시로 GPT-2 구현을 따라가며 모델 구조, 가중치 로딩, 실행 확인 과정을 확인해보겠습니다.

## 1. 폴더/파일 생성하기

모델 구조, 가중치 로딩, 실행 예제를 분리합니다. 체크포인트의 키나 저장 형식이 바뀌면 체크포인트 로더에서 처리할 수 있도록 합니다.

| 파일 | 역할 |
|---|---|
| **`models/<모델명>/model.h`** | 모델 - 전체 그래프, 모델 설정, 입출력, 실행 상태 |
| **`models/<모델명>/<배포명>-weights.h`** | 체크포인트 로더 - 체크포인트를 읽고 가중치 슬롯에 할당 |
| **`examples/<모델명>/demo.cpp`** | 실행 예제 - 입력 준비, 모델 실행, 출력 확인 |
| `models/<모델명>/groups/*.h` | 모델 전용 블록을 `Node`와 그룹으로 조합할 때 추가 |
| `models/<모델명>/nodes/*.h` | 기존 `Node`로 필요한 동작을 구현할 수 없을 때 추가 |
| `models/<모델명>/shaders/*.comp` | 새 GPU 계산이 필요할 때 추가 |
| `data/<모델명>/` | 체크포인트와 입력 데이터 위치 |
| `data/<모델명>/ckpts/` | 체크포인트 기본 저장 경로 |
| `data/<모델명>/ckpts/<크기>/` | 크기별 체크포인트 경로 (예: `data/gpt2/ckpts/124M/`) |

GPT-2의 경우 모델 이름으로 `gpt2`, 체크포인트 배포명은 `openai`를 사용합니다. 같은 모델 구조에 다른 배포자의 체크포인트를 지원하려면 새로운 로더를 추가하면 됩니다.

커스텀 모델 구현에 앞서서 필요한 `Node`를 파악한 후 먼저 [제공 노드](06-nodes.md)와 [공통 그룹](../../vai/groups/)을 확인합니다. 지원하는 기본 제공 `Node`와 공통 그룹에 필요한 구현체가 없으면 새로 구현하시면 됩니다. 새로 `Node`와 그룹을 만드는 방법은 [vAi 사용법](03-usage/README.md)에서 다룹니다. 이 문서에서는 제작하신 `Node`와 그룹으로 커스텀 모델을 만드는 과정과 공식 저장소에 올릴 때 추가로 지켜야할 규칙을 위주로 설명드립니다.

## 2. 그래프 만들기

### 2.1 모델 구조와 실행 설정 분리

먼저 원본 모델의 연산 순서와 입출력을 확인합니다. 레이어 수, 차원, 활성화 함수, 정규화 방식은 체크포인트와 맞아야 합니다. 여러 `Config` 구조체가 있는데 다음 기준으로 구분하고 작성합니다.

| 설정 | 정하는 내용 | GPT-2 예시 |
|---|---|---|
| `ModelConfig` | 체크포인트가 정하는 모델 구조 | 레이어 수, 임베딩 차원, 어휘 크기, 학습된 문맥 길이 |
| `RuntimeConfig` | 사용자가 고르는 실행 설정 | KV 캐시 dtype, 실행할 최대 문맥 길이 |
| `vai::NeuralNetConfig` | 장치와 엔진 자원 | 장치, 기본 가중치·출력 dtype, 버퍼 풀 |

`Config` 구조체는 [`model.h`](../../models/gpt2/model.h)에서 정의합니다. 이후 모델 생성자가 설정을 받아서 각 구성 요소에 전달합니다.
```cpp

Model(const vai::NeuralNetConfig& netCfg,
      const ModelConfig& modelCfg,
      const RuntimeConfig& runtimeCfg = {})
: NeuralNet(netCfg)                       // 엔진 설정
, modelCfg(modelCfg)
, runtimeCfg(runtimeCfg)
, maxRuntimeContext(std::min(
    runtimeCfg.maxRuntimeContext,
    modelCfg.languageCfg.maxTrainedContext))
, embedding({                             // 모델 구조 설정
    .vocabSize = modelCfg.languageCfg.vocabSize,
    .embedDim = modelCfg.languageCfg.layerCfg.embedDim,
    .posTableSize = modelCfg.languageCfg.maxTrainedContext})
, decoder(modelCfg.languageCfg.with({      // 구조 설정에 실행 설정 적용
    .kvCacheDtype = runtimeCfg.kvCacheDtype,
    .kvCacheSize = maxRuntimeContext,
    .sliceLastRow = true}))
, lmHead({
    .inDim = modelCfg.languageCfg.layerCfg.embedDim,
    .outDim = modelCfg.languageCfg.vocabSize})
{
    // ...
}
```

모델 구조가 고정되어 있으면 `ModelConfig` 구조체를 생략할 수 있습니다. 외부에서 변경할 실행 설정도 하나뿐이면 별도 `RuntimeConfig` 구조체도 생략할 수 있습니다.

### 2.2 노드와 그룹 조립

모델의 입출력 이름을 정하고, 생성자에서 `Node`와 그룹을 연결합니다. 아래 코드는 [GPT-2 모델](../../models/gpt2/model.h)의 멤버와 생성자를 발췌한 코드입니다.
```cpp
namespace gpt2 {

using Transformer = vai::group::Transformer<
    vai::TransformerType::Classic, vai::QkvLayout::Fused>;

struct Model : public vai::NeuralNet {
    vai::node::TokenEmbedding embedding;
    Transformer decoder;
    vai::node::Linear lmHead;
    vai::node::TopKSampling topkSampling;

    // ... 모델 설정 멤버

    Model(...)
    // ... 클래스 및 각 멤버의 초기화 목록
    {
        input("inTokenIds") - embedding - decoder - lmHead - topkSampling
            - output("nextTokenId");

        // ... 실행 상태 초기화 생략
    }
};

} // namespace gpt2
```

모델은 `Node`와 그룹을 멤버로 가집니다. 생성자 안에서 입력 경계부터 출력 경계까지 연결합니다. 위 코드에서 GPT-2의 경우 임베딩, `Transformer` 디코더 그룹, 출력 투영과 토큰 샘플링 순서로 연결되는 것을 확인할 수 있습니다.

전체 모델의 경계는 `input()`과 `output()`으로 연결합니다. 주 경로 외에 잔차 경로와 여러 입력의 슬롯 이름도 마찬가지로 맞춰야 합니다.

연결한 `Node`와 그룹의 주소는 유지해야 합니다. 멤버는 값으로 선언하는 방식을 권장합니다. 또한, 벡터에는 `reserve()`로 공간을 확보한 뒤 `emplace_back()`으로 생성합니다. 다른 소유 방식은 [컨테이너 사용 시 주의할 점](03-usage/node-group.md#4-컨테이너-사용-시-주의할-점)을 참조합니다.

### 2.3 반복 실행 상태 관리

GPT-2는 프롬프트를 처리한 뒤, **모델을 반복 호출**해 토큰을 하나씩 생성합니다. 매번 이전 토큰을 다시 계산하지 않도록, 이전에 계산한 K/V를 KV캐시에 저장해 재사용합니다.  따라서 모델은 캐시와 지금까지 처리한 토큰 수를 유지합니다. 처리한 토큰 수는 다음 입력의 임베딩 위치와 캐시에 값을 쓸 위치를 정하는데 사용합니다.

이런 반복적인 실행이 필요한 모델을 구현할 때는 다음을 정합니다.

- **유지할 정보**: 다음 호출에서도 사용할 캐시와 위치 정보.
- **갱신 시점**: 새 입력을 처리하기 전에 필요한 값을 갱신합니다.
- **초기화 시점**: 독립된 프롬프트를 시작할 때 이전 캐시를 사용하지 않도록 합니다.

GPT-2는 `step()`에서 이전 GPU 작업의 완료를 기다린 뒤 위치와 샘플링 값을 갱신합니다. `resetCache()`는 캐시 위치를 0으로 되돌립니다.

## 3. 가중치 불러오기

그래프를 만들었으면 각 `Node`에 학습된 가중치를 넣습니다. 체크포인트 로더는 파일의 `Tensor`를 모델이 요구하는 슬롯, 형상, dtype, 값의 의미에 맞춥니다. 가중치는 해당 `Node`의 `persistent` 슬롯에 할당합니다. 캐시처럼 `Node`가 준비하는 상태는 2절에서 정한 초기화 규칙을 따릅니다.

### 3.1 체크포인트 키와 슬롯 대응

체크포인트의 `Tensor` 이름을 키라고 합니다. 체크포인트 로더는 이 키와 모델의 가중치 슬롯을 대응시킵니다. 키나 파일 구성이 바뀌어도 같은 모델 구조를 사용할 수 있도록, 이 대응은 체크포인트 로더에서 처리합니다. 다음은 [GPT-2 로더](../../models/gpt2/openai-weights.h)에서 가중치를 슬롯에 넣는 부분을 발췌한 코드입니다. `model`과 `ckpt`는 로딩 함수가 인자로 받습니다.

```cpp
vai::Dtype edt = model.embedding.config_.wDtype;
model.embedding["weight"] = vai::Tensor(ckpt["wte.weight"], edt);
model.embedding["posWeight"] = vai::Tensor(ckpt["wpe.weight"], edt);
```

`wte.weight`와 `wpe.weight`는 체크포인트의 키입니다. `weight`와 `posWeight`는 `Node`가 정의한 슬롯 이름입니다. `Tensor(ckpt[key], dtype)`를 통해서 해당 키의 `Tensor`를 지정한 dtype으로 읽습니다. 같은 종류의 `Node`가 반복되면 공통 로딩 함수로 묶습니다. 이 함수는 대상 `Node`와 키의 접두사를 받아 필요한 슬롯을 채우도록 작성합니다. 선택 파라미터는 `Node` 설정을 확인한 뒤 읽습니다. 예를 들어 `hasBias`가 켜진 `Node`에만 편향을 넣습니다.

### 3.2 가중치 형상과 값 변환

먼저 대상 슬롯이 요구하는 형상과 dtype을 확인합니다. 슬롯 이름이 같아도 축 순서나 저장된 값의 의미는 다를 수 있습니다.
슬롯의 조건은 [제공 노드](06-nodes.md)와 해당 `Node`의 구현에서 확인합니다. 아래 표는 형상과 값의 차이에 따라 다르게 구현된 체크포인터 로더의 예시입니다. 모두 vAi가 기본 제공하는 모델의 코드에서 확인할 수 있습니다.

| 차이 | 체크포인트 로더에서 처리할 일 | 실제 구현 |
|---|---|---|
| 축과 형상이 다름 | `permute()`로 축을 바꿉니다. `reshape()`로 형상을 바꾸거나 `vai::slice()`로 필요한 행을 읽습니다. | [PaliGemma 로더](../../models/paligemma/google-weights.h) |
| 가중치가 따로 저장됨 | 대상 `Node`가 읽는 순서로 `vai::concat()`을 사용해 합칩니다. | [Llama 3 로더](../../models/llama3/meta-weights.h) |
| 값의 의미가 다름 | `Node`가 사용하는 수식에 맞게 값을 계산합니다. | [Gemma 3 로더](../../models/gemma3/google-weights.h), [YOLOv8 로더](../../models/yolov8/ultralytics-weights.h) |
| 두 슬롯이 가중치를 공유함 | 두 슬롯에 동일한 `Tensor`를 대입합니다. | [GPT-2 로더](../../models/gpt2/openai-weights.h) |

변환은 첫 실행 전에 완료해야 합니다. 변환 뒤에는 대상 슬롯의 형상과 dtype이 맞는지 확인합니다.

원본이 `.safetensors`가 아니면 변환 스크립트를 함께 제공해야 합니다.  파일 이름은 `<배포명>-weights-convertor.py` 형식입니다.

<a id="배포명-weightsh"></a>

### 3.3 가중치 로딩 함수 작성

사용하는 코드에서 `loadWeights<"배포명">(model)`으로 가중치를 불러올 수 있도록 작성합니다.
이 함수는 체크포인트를 열고 모델의 가중치 슬롯을 채웁니다.

다음은 [GPT-2 로더](../../models/gpt2/openai-weights.h)에서 발췌한 코드입니다.

```cpp
template<>
inline void loadWeights<"openai", model::production::gpt2::Net>(model::production::gpt2::Net& net)
{
    std::string dir = std::string(VAI_DATA_DIR "/gpt2/ckpts/") + model.modelCfg.variant;
    gpt2::openai::loadWeights(model, vai::Checkpoint(dir));
}
```

이 예제에서 `loadWeights<"openai">()`는 체크포인트 폴더를 엽니다.
`gpt2::openai::loadWeights()`는 받은 체크포인트의 `Tensor`를 슬롯에 넣습니다.
파일을 고르는 함수와 슬롯을 채우는 함수를 나눠 작성하는 것을 권장합니다.
그러면 파일 경로가 바뀌어도 슬롯을 채우는 코드를 그대로 사용할 수 있습니다.

`loadWeights<"배포명">` 함수는 `namespace` 블록 밖에 정의합니다.
헤더에 정의할 때는 `inline`을 붙입니다.

체크포인트의 기본 저장 경로는 `data/<모델명>/ckpts/`입니다. 크기별 체크포인트가 있으면 하위 폴더로 나눠서 보관합니다.

```
data/gpt2/ckpts/124M/
data/gpt2/ckpts/355M/
data/gpt2/ckpts/774M/
data/gpt2/ckpts/1.5B/
```

`vai::Checkpoint`는 폴더의 모든 `.safetensors` 파일을 열고 키로 `Tensor`를 찾습니다. 따라서 가중치가 여러 파일에 나뉘어 있어도 같은 코드로 읽을 수 있습니다. 다른 체크포인트나 검증용 `.safetensors` 파일이 함께 있으면 읽을 파일을 목록으로 지정합니다. 파일 목록을 지정하는 방법은 [weights.h](../../vai/engine/weights.h)를 참조하면 됩니다.

## 4. 모델 실행하기

데모(`demo.cpp`)는 사용자가 모델을 실행하는 데 필요한 순서를 보여주는 좋은 예시입니다. 모델 생성 → 가중치 로딩 → 입력 준비 → 실행 → 출력 확인 순서로 작성합니다. 파일 읽기, 전처리, 후처리는 데모에서 처리하고 모델은 `Tensor`를 입출력으로 사용합니다. 모델 구조를 정하는 설정 파일이 있으면 먼저 읽습니다. 그리고 그 설정으로 모델을 만든 뒤 가중치를 불러옵니다.

### 4.1 입출력 조건

사용자가 다음 내용을 알 수 있도록 작성합니다.

| 항목 | 명시할 내용 |
|---|---|
| 이름 | 그래프에 정의한 입력과 출력 이름 |
| 형상 | 각 축의 의미와 지원하는 크기 |
| dtype | 입력과 출력의 데이터 타입 |
| 값의 의미 | 입력 정규화 방법과 출력의 해석 방법 |

원본 입력을 전처리한 뒤 이 조건에 맞는 `Tensor`를 만듭니다. 그리고 부동소수점 입력은 `setFloats()`로 `Tensor`의 dtype에 맞춰 넣을 수 있습니다. `setData<T>()`는 dtype 변환 없이 데이터를 복사하므로 저장 형식이 맞는지도 추가로 확인합니다.

다음은 [`NeuralNet::operator()`](../../vai/engine/neural-net.h)의 호출 형식을 보여주는 설명용 예제입니다. `model`과 `input`은 준비되어 있고, 필요한 상태 갱신도 마친 것으로 가정합니다. 입출력 이름은 `in0`과 `out0`이며 출력 dtype은 FP32입니다.

```cpp
vai::Tensor output;
model({{"in0", input}}, {{"out0", &output}});
const float* result = output.getData<float>();
```

입력은 이름과 `Tensor`를 묶어 전달합니다. 그리고 출력은 결과를 받을 `Tensor`의 주소로 지정합니다. 출력 `Tensor`는 빈 상태로도 선언할 수 있습니다.

### 4.2 실행 메서드의 상태 관리

반복 호출에 필요한 준비가 있으면 준비 코드와 그래프 호출을 하나의 메서드로 묶습니다. 이 때 캐시와 카운터의 관리는 모델 내부에 둡니다. 그리고 데모에는 실행 메서드와 초기화 메서드의 호출만 남깁니다.

[GPT-2의 `prefill()`과 `decode()`](../../models/gpt2/model.h)가 이런 실행 인터페이스의 예시입니다.

엔진은 기록한 GPU 명령을 재사용하므로, `Node`의 명령 기록 코드는 매 호출 실행되지 않습니다.
CPU에서 매 호출 갱신할 값은 모델의 실행 메서드에서 처리합니다. 그리고 GPU가 읽는 상태를 CPU에서 바꾸기 전에는 이전 GPU 작업의 완료를 기다립니다.

실행 사이에 유지할 값과 초기화 조건은 2절에서 정한 규칙을 따릅니다.

## 5. 빌드하고 검증하기

### 5.1 실행 타겟 등록

`examples/<모델명>/CMakeLists.txt`에 실행 타겟을 등록합니다.
타겟 이름은 `demo-<모델명>`으로 정합니다.

다음은 등록 패턴을 보여주기 위해 [GPT-2 빌드 구성](../../examples/gpt2/CMakeLists.txt)의 주석과 줄바꿈을 줄인 예제입니다.

```cmake
cmake_minimum_required(VERSION 3.16)
project(demo-gpt2)
add_executable(${PROJECT_NAME} demo.cpp)
```

[examples/CMakeLists.txt](../../examples/CMakeLists.txt)에 `add_subdirectory(<모델명>)`을 추가합니다.
이 경로의 예제는 상위 설정에서 `vAi`를 링크합니다.
모델 전용 헤더는 사용하는 파일에 포함합니다.

전용 커널이 있으면 실행 타겟에 `add_dependencies(demo-<모델명> <모델명>-shaders)`를 추가합니다. [모델 빌드 설정](../../models/CMakeLists.txt)이 `models/<모델명>/shaders/*.comp`를 수집해 이 셰이더 타겟을 만듭니다. 그리고 컴파일된 커널의 경로는 `<MODEL>_SHADER_OUTPUT_DIR` 매크로로 전달됩니다.

[YOLOv8 빌드 구성](../../examples/yolov8/CMakeLists.txt)이 그 예입니다.

`Node`에서 이 경로를 사용하는 구체적인 방법은 [커스텀 노드와 커널 작성](07-contributing-nodes.md)에서 다룹니다.

### 5.2 데모 빌드와 기준 출력 비교

빌드 환경은 [빠른 실행](02-quickstart.md)을 확인하시면 됩니다. 아래 빌드는 데모 타겟을 빌드하는 예제입니다. 명령은 저장소 루트에서 실행하시면 됩니다. 새 모델을 추가했으면 타겟 이름을 해당 모델의 이름으로 바꾸면 됩니다.

```bash
cmake -S . -B build -DVAI_BUILD_EXAMPLES=ON
cmake --build build --target demo-gpt2 --config Release
./build/bin/demo-gpt2
```

제출 전에 다음을 확인해주세요.

- 같은 가중치와 입력으로 기준 구현의 출력과 비교해야 합니다. 전처리와 후처리 조건도 맞춥니다.
- 지원하는 형상과 dtype으로 실행합니다. 반복 실행과 상태 초기화도 확인합니다.
- 확률적인 출력을 사용하는 모델은 난수와 샘플링 조건을 맞춥니다. 가능하면 선택 전의 수치 출력도 비교합니다.
- 비교에 사용한 입력, dtype, 허용 오차, 결과를 기록합니다.
- 다른 기여자가 재현할 수 있도록 다운로드, 변환, 빌드, 실행 방법을 적습니다.

### 5.3 코드와 주석 작성 규칙

| 항목 | 규칙 |
|---|---|
| 코드 형식 | [공통 코드 컨벤션](index.md#3-공통-코드-컨벤션)을 따릅니다. |
| 모델과 그룹 | 모델과 모델 전용 그룹은 전용 네임스페이스에 둡니다. `struct`를 사용해 모델은 `vai::NeuralNet`, 그룹은 `vai::NodeGroup`을 `public`으로 상속합니다. |
| 모델 이름 | 모델 구조체 이름은 `Model`로 고정합니다. `gpt2::Model`, `llama3::Model`처럼 네임스페이스로 구분합니다. |
| 헤더 스탬프 | 파일 상단에 `/* === ... === */` 형식의 블록 주석을 둡니다. 첫 항목은 `Written by`입니다. |
| 모델 스탬프 | `Model`에 모델 이름과 지원 크기를 적습니다. |
| 로더 스탬프 | `Checkpoint`에 출처, 저장 경로, 필요한 변환 방법을 적습니다. |
| 데모 스탬프 | `Task`에 실행 목적을 적습니다. `Assets`에 필요한 외부 파일의 경로와 출처를 적습니다. |
| 체크포인트 | 다운로드 방법을 제공합니다. 저장소에 포함할 파일은 배포 조건을 확인하고 `.gitignore`에 예외를 등록합니다. |

**모델과 그룹의 선언**

타입 이름은 `PascalCase`로 씁니다. 각 단어의 첫 글자를 대문자로 씁니다. (예: `ModelConfig`, `RuntimeConfig`.) 함수와 변수 이름은 `camelCase`로 씁니다. 첫 단어는 소문자로 시작하고 다음 단어의 첫 글자는 대문자로 씁니다. (예: `resetState()`, `numLayers`, `modelCfg`.) 함수의 여는 중괄호는 다음 줄에 둡니다.

다음은 `model.h`의 작성 형식을 보여주는 예제입니다. `Node` 구성과 생성자는 생략했습니다. `model_name`과 주석의 모델명과 작성자 이메일은 기여자의 정보로 바꾸면 됩니다. 이때 구조체 이름 `Model`은 유지합니다.

```cpp
/*=============================================================================
  Written by   contributor@example.com
  Model        <모델명> (<지원 크기>)
=============================================================================*/

namespace model_name {

struct Block : public vai::NodeGroup {
    // ...
};

struct Model : public vai::NeuralNet {
    uint32_t numLayers = 0;

    void resetState()
    {
        // ...
    }

    // ...
};

} // namespace model_name
```

**체크포인트 로더의 주석**

체크포인트 로더 상단에는 다운로드 출처와 저장 경로를 적습니다. 변환이 필요하면 원본 경로와 변환 방법도 적습니다. 다음은 변환이 필요한 체크포인트의 작성 예시입니다.

```cpp
/*=============================================================================
  Written by   contributor@example.com
  Checkpoint   Each tier reads data/yolov8/ckpts/<tierName>/model.safetensors.
               Ultralytics ships .pt, so the file takes one conversion:
                 download   huggingface.co/Ultralytics/YOLOv8
                              -> data/yolov8/ckpts/<tierName>/model.pt
                 convert    ultralytics-weights-convertor.py
                              -> data/yolov8/ckpts/<tierName>/model.safetensors
=============================================================================*/
```

**데모의 주석**

`Task`에는 실행 목적을 적습니다. `Assets`에는 필요한 외부 파일의 저장 경로를 먼저 적고, 다음 줄에 출처를 적습니다. 저장소에 포함된 파일이면 그 사실을 적습니다.

```cpp
/*=============================================================================
  Written by   contributor@example.com
  Task         GPT-2 — completes one prompt.
  Assets       data/gpt2/tokenizer.json
                 from  huggingface.co/openai-community/gpt2
=============================================================================*/
```
