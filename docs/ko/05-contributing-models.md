# 커스텀 모델 작성

> **커스텀 모델은 원하는 대로 자유롭게 만들 수 있습니다**. 다만 만든 모델을 공식 배포판에 포함하려면 아래 컨벤션과 규칙에 맞춰 작성한 뒤 PR을 보내주세요.

## 구성 파일

모델 하나는 세 파일이 기본이며 전용 연산이 필요하면 둘을 더 둡니다.

| 파일                             |    | 내용                   |
| ------------------------------ | -- | -------------------- |
| `models/<impl>/<모델명>/net.h`         | 필수 | 모델 정의                |
| `models/<impl>/<모델명>/nodes/*.h`       | 선택 | 기본 제공에 없는 노드         |
| `models/<impl>/<모델명>/shaders/*.comp`  | 선택 | 위 노드에서 사용하는 커널       |
| `models/<impl>/<모델명>/<배포명>-weights.h` | 필수 | 특정 배포자의 체크포인트를 읽는 로더 |
| `models/<impl>/<모델명>/demo.cpp`      | 필수 | 모델을 실제로 구동해 보이는 데모   |

`<impl>`은 정식 구현이면 `production`, 아직 진행 중이면 `experimental`입니다.
체크포인트는 `data/ckpts/<모델명>/<배포명>/`, 토크나이저는 `data/tokenizers/<모델명>/`, 입력 이미지는 `data/images/`에 둡니다.

모델을 만드는 방법 자체는 [vAi 사용법](03-usage/README.md)에서 다룹니다. 반복 블록을
묶는 방법은 [노드그룹](03-usage/node-group.md), 전체를 엮고 입출력 경계를 지정하는 방법은
[신경망](03-usage/neural-net.md) 참조. 여기서는 모델을 저장소에 올릴 때 추가로
지켜야 할 규칙만 다룹니다.

## 공통 규칙

[개요의 공통 코드 컨벤션](index.md#공통-코드-컨벤션)을 지킵니다.

## net.h

#### 1. 파일 헤드 스탬프

파일 최상단에 저자와 내용을 명시합니다. `Written by`는
작성자의 서명(이메일)이며 항상 첫 줄에 둡니다. `Model`에는 구현하는 모델과 지원
사이즈를, `Modules`에는 이 파일에서 정의하는 노드그룹을 빠짐없이 적습니다.

```cpp
/*=============================================================================
  Written by   oneway@vaistrata.com
  Model        GPT-2 (124M / 355M / 774M / 1.5B)
  Modules      MultiHeadAttention, FeedForward, TransformerBlock
=============================================================================*/
```

#### 2. 네임스페이스

이 파일에서 정의하는 모델과 노드그룹은 모두 전용
네임스페이스로 묶습니다.

```cpp
namespace <모델명> {

...

} // namespace <모델명>
```

#### 3. `struct`로 상속

`vai::NodeGroup`과 `vai::NeuralNet`은 `class`가 아니라
`struct`로 상속합니다.

```cpp
struct FeedForward : public vai::NodeGroup { ... };
struct Net : public vai::NeuralNet { ... };
```

#### 4. 모델 구조체의 이름은 `Net`으로 고정

바깥에서는 `model::production::gpt2::Net`,
`model::production::llama3::Net`처럼 네임스페이스로 모델을 구분합니다. 한 파일에 여러 모델을 담아야
한다면 네임스페이스를 중첩해 나눕니다. 모델을 쓰는 코드에서는
`namespace gpt2 = model::production::gpt2;`처럼 별칭으로 줄여 씁니다.

#### 5. 노드는 값으로 소유

멤버로 두는 `vai::Node`와 `vai::NodeGroup`은 값으로
선언합니다. 로우 포인터나 레퍼런스로 소유하지 않습니다. 노드가 아닌 일반 멤버에는
이 제약이 없습니다.

```cpp
struct Net : public vai::NeuralNet {
    vai::node::TokenEmbedding embedding;          // OK
    MultiHeadAttention mhAttn;             // OK. 노드그룹도 값으로
    std::vector<TransformerBlock> blocks;  // OK. 값을 담는 컨테이너
    std::unique_ptr<Tokenizer> tokenizer;  // OK. 노드가 아니면 포인터도 자유
    vai::node::Linear* lmHead;               // X. 로우 포인터
    vai::node::RmsNorm& norm;                // X. 레퍼런스
};
```

노드를 컨테이너에 담을 때의 주의사항은
[노드그룹의 컨테이너 사용 시 주의할 점](03-usage/node-group.md#컨테이너-사용-시-주의할-점) 참조.

## 배포명-weights.h

모델을 실제로 구동하려면 학습된 가중치를 불러와야 합니다. 가중치는 배포자마다
다양한 포맷으로 제공되며 현재 vAi는 HuggingFace에서 쉽게 구할 수 있는
`.safetensors` 포맷을 지원합니다.

다만 같은 `.safetensors`라도 체크포인트마다 키 이름, 텐서 형상, 레이어 퓨전 방식
등이 제각각입니다. 그래서 체크포인트의 가중치를 vAi 모델 구조에 맞게 매핑하고
변환하는 로딩 헬퍼가 필요합니다. 이 헬퍼는 각 모델의 기여자가 직접 작성해 함께
제공해야 합니다. 그러면 하나의 `net.h` 구현으로 다양한 체크포인트를
불러올 수 있습니다.

파일명은 해당 체크포인트를 학습하거나 배포한 곳의 이름을 따릅니다. 예를 들어
OpenAI가 배포한 체크포인트라면 `openai-weights.h`로 작성합니다. 같은 모델의
체크포인트를 여러 곳에서 배포한다면 배포자마다 파일을 하나씩 추가합니다.

이 파일이 제공해야 하는 것은 다음과 같습니다:

#### 1. 파일 헤드 스탬프

체크포인트의 출처와 저장 경로는 반드시 명시합니다.
사용자는 파일 헤드의 스탬프만 보고 체크포인트를 어디에서 받아 어디에 두어야 하는지
바로 알 수 있어야 합니다.

```cpp
/*=============================================================================
  Written by   oneway@vaistrata.com
  Checkpoint   Each size reads data/ckpts/gpt2/openai/<ModelConfig::variant>/model.safetensors,
               taken from the matching HuggingFace repo:
                 124M   openai-community/gpt2
                 355M   openai-community/gpt2-medium
                 774M   openai-community/gpt2-large
                 1.5B   openai-community/gpt2-xl
=============================================================================*/
```

#### 2. `loadWeights` 특수화

외부에서는 `loadWeights<"배포명">(model)` 한 줄로
체크포인트를 불러올 수 있어야 합니다. 호출하는 쪽에서는 체크포인트의 키 이름이나
실제 파일 경로를 알 필요가 없습니다.

진입점은 [`vai/engine/weights.h`](../../vai/engine/weights.h)에 선언된 함수 템플릿이며
배포명을 문자열 NTTP로 넘겨 배포자별 구현을 특수화합니다.

```cpp
namespace model::production::gpt2::openai {

inline void loadWeights(Model& net, const vai::SafetensorsParser& parser)
{
    // 체크포인트의 키를 모델 노드의 슬롯으로 매핑
}

} // namespace model::production::gpt2::openai


template<>
inline void loadWeights<"openai", model::production::gpt2::Net>(model::production::gpt2::Net& net)
{
    std::string path = std::string(VAI_DATA_DIR "/ckpts/gpt2/openai/")
                     + net.modelCfg.variant + "/model.safetensors";
    model::production::gpt2::openai::loadWeights(net, vai::SafetensorsParser(path.c_str()));
}
```

체크포인트 경로 역시 이 특수화 안에서 조립합니다. 따라서 데모나 모델을 사용하는
코드에서는 체크포인트의 구체적인 경로나 파일 구성을 알 필요가 없습니다.


#### 3. (선택) 변환 스크립트

원본 체크포인트가 `.safetensors`가 아닌 포맷으로
배포된다면 변환 스크립트도 함께 제공합니다. 파일명은
`<배포명>-weights-convertor.py`로 통일합니다. 스크립트에는 스탬프를 두지 않고
원본의 출처와 변환 전후 경로를 위 `Checkpoint` 스탬프에 함께 적습니다.

예를 들어 yolov8은 Ultralytics에서 배포하는 `.pt` 체크포인트를 그대로 읽을 수 없기
때문에
[`ultralytics-weights-convertor.py`](../../models/internal/yolov8/ultralytics-weights-convertor.py)를
함께 제공합니다. 이 스크립트가 vAi가 읽을 수 있는 `.safetensors`로 바꿉니다.

```cpp
/*=============================================================================
  Written by   oneway@vaistrata.com
  Checkpoint   Each tier reads data/ckpts/yolov8/ultralytics/<tierName>/model.safetensors.
               Ultralytics ships .pt, so the file takes one conversion:
                 download   huggingface.co/Ultralytics/YOLOv8
                              -> data/ckpts/yolov8/ultralytics/<tierName>/model.pt
                 convert    ultralytics-weights-convertor.py
                              -> data/ckpts/yolov8/ultralytics/<tierName>/model.safetensors
=============================================================================*/
```

## demo.cpp

`demo.cpp`는 모델의 대표적인 사용 예를 보여주는 실행 프로그램입니다. 어떤 태스크를 선택할지는 자유입니다.

#### 1. 파일 헤드 스탬프

`Task`에는 이 프로그램이 수행하는 일을, `Assets`에는
실행에 필요한 외부 파일을 적습니다. 각 파일의 저장 경로를 먼저 적고 다음 줄을 한 단계 들여 출처를 명시합니다. 저장소에 이미 포함된 파일이라면 그 사실을 적습니다.

```cpp
/*=============================================================================
  Written by   oneway@vaistrata.com
  Task         GPT-2 — completes one prompt, profiles prefill and decode.
  Assets       data/tokenizers/gpt2/tokenizer.json
                 from  huggingface.co/openai-community/gpt2
=============================================================================*/
```

#### 2. 프로젝트 등록

`models/<impl>/<모델명>/CMakeLists.txt`에 등록하며 프로젝트명은
`project(demo-<모델명>)`으로 합니다. 다른 태스크를 보여주는 프로그램은 `examples/chat-llama3/`처럼 `examples/<태스크>-<모델명>/`에 둡니다.

```cmake
cmake_minimum_required(VERSION 3.16)

project(demo-llama3)

# demo-llama3 — one-prompt completion with timings
add_executable(${PROJECT_NAME}
    demo.cpp
)
```


## 전용 노드와 커널 (선택)

vAi의 기본 노드로 표현할 수 없는 연산이 있거나 특별한 인터페이스, 레이어 퓨전, 특정 타겟 최적화 등이 필요하면 모델 전용 노드와 커널을 추가할 수 있습니다.

파일명은 `<모델명>-<노드명>` 형식으로 맞추고 가능하면 노드와 커널의 이름을 동일하게 유지합니다.

```
models/<impl>/<모델명>/
  nodes/<모델명>-<노드명>.h        노드
  shaders/<모델명>-<노드명>.comp   커널
```

`shaders/`의 커널은 `models/CMakeLists.txt`가 `<impl>-<모델명>-shaders` 타깃으로 컴파일하고, 출력 디렉토리는 `<IMPL>_<모델명>_SHADER_OUTPUT_DIR` 매크로로 코드에 전달됩니다. 커널을 쓰는 프로그램은 의존성만 추가합니다.

다음은 `models/internal/yolov8/CMakeLists.txt`의 예입니다.

```cmake
add_executable(${PROJECT_NAME}
    demo.cpp
)

add_dependencies(${PROJECT_NAME} internal-yolov8-shaders)
```

노드는 이 경로에서 컴파일된 `.spv` 파일을 불러옵니다. 다음은 `decode.h`의 예입니다.

```cpp
pipe = pipelinePool().request(INTERNAL_YOLOV8_SHADER_OUTPUT_DIR, "decode", { ... });
```

구체적인 노드와 커널 작성법은
[커스텀 노드와 커널 작성](07-contributing-nodes.md) 참조.

## vAi 공식 노드 승격

모델 폴더에 둔 노드라도 아래에 해당하면 내부 테스트를 거쳐 연관 커널과 함께 공식 노드로 승격됩니다.

- 현재 vAi가 제공하지 않지만 범용으로 사용 가능한 노드
- 현재 vAi에 존재하지만 다양한 환경에서 성능이 더 잘 나오는 노드
- 현재 vAi에 존재하고 성능이 비슷하지만 사용성이 더 좋은 노드

## 체크포인트

`data/ckpts/<모델명>/<배포명>/` 아래에 두고 그 경로는 로더의 `Checkpoint` 스탬프에 밝힙니다.

고용량이거나 라이선스가 걸린 체크포인트는 커밋하지 않습니다.
[`.gitignore`](../../.gitignore)가 `data/` 아래의 `*.safetensors`, `*.json`, `*.pt`를
막습니다. 크지 않고 직접 학습시킨 체크포인트만 `.gitignore`에 예외를 더해 올릴 수 있습니다.
`data/ckpts/mnist/jroot/model.safetensors`가 대표적인 사례입니다.
