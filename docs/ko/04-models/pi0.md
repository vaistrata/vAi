# π0

π0(Black et al., 2024)는 Physical Intelligence에서 개발한 비전-언어-행동(VLA) 모델입니다. PaliGemma 3B가 카메라 이미지와 작업 지시문을 읽어 KV 캐시를 채우고, Gemma 300M 크기의 행동 전문가(action expert)가 그 캐시를 참조해 로봇의 행동 청크를 만듭니다. 행동은 flow matching으로 생성하며, 노이즈에서 시작해 정해진 횟수만큼 속도장을 따라 적분합니다. 모델은 한 번에 32차원 행동 50개를 예측합니다.

**참고자료**: [논문](https://arxiv.org/abs/2410.24164), [체크포인트(HuggingFace)](https://huggingface.co/lerobot/pi0_base), [저장소](https://github.com/Physical-Intelligence/openpi)

## 설정

```cpp
namespace pi0 = model::production::pi0;

auto modelCfg = pi0::ModelConfig::_3B();

pi0::Net net({
    .wDefaultDtype = Dtype::Fp16,
}, modelCfg);
```

### NeuralNetConfig

π0는 기본값을 그대로 쓰고, 로더가 각 norm의 파라미터와 스텝별 시간 테이블(`stepTimeBias`)만 fp32로 읽습니다. 행동 출력(`velocityHead`)은 `oDefaultDtype`과 상관없이 fp32로 고정합니다.
세부사항은 [신경망](../03-usage/neural-net.md) 참조.

### ModelConfig

팩토리는 `_3B()` 하나이고, `variant`는 `"3B"`입니다.

| 필드 | 값 | 설명 |
|---|---|---|
| `maxCameras` | 3 | 학습에 쓴 카메라 수, 한 번에 넣을 수 있는 최대 이미지 수 |
| `maxPromptTokens` | 48 | 작업 지시문의 최대 토큰 수 |
| `vocabSize` | 257,152 | 텍스트 256,000개, `<loc>` 1,024개, `<seg>` 128개 |
| `actionDim` | 32 | 행동 하나의 차원. 상태도 같은 폭으로 패딩 |
| `actionChunk` | 50 | 한 청크에 담기는 행동 수 |

`visionTrf`는 [`models/common.h`](../../../models/common.h)의 `so400m14(224)`가 만드는 `VisionTransformerConfig`이고, `languageTrf`와 `actionTrf`는 `gemma("2B")`, `gemma("300M")`이 만드는 `TransformerConfig`입니다.

| 멤버 | `nBlocks` | `embedDim` | `headDim` | `numQHeads` | `numKVHeads` | `ffnHiddenDim` |
|---|---|---|---|---|---|---|
| `visionTrf` (SigLIP So400m/14) | 27 | 1152 | 72 | 16 | 16 | 4304 |
| `languageTrf` (Gemma 2B) | 18 | 2048 | 256 | 8 | 1 | 16384 |
| `actionTrf` (Gemma 300M) | 18 | 1024 | 256 | 8 | 1 | 4096 |

`languageTrf`와 `actionTrf`는 폭만 다르고 블록 수, `headDim`, KV 헤드 수가 같아서, 두 스택이 블록마다 하나의 KV 캐시를 함께 씁니다. `maxContext()`는 KV 캐시가 담는 최대 위치 수로, 카메라 3대의 이미지 토큰(3 × 256), 지시문 48개, 상태 토큰 1개, 행동 50개를 더한 867입니다.

### RuntimeConfig

디노이즈 스텝 수와 KV 캐시의 dtype을 정합니다.

| 필드 | 기본값 |
|---|---|
| `denoiseSteps` | `10` |
| `kvCacheDtype` | `Dtype::Fp16` |

`denoiseSteps`는 행동 청크를 만들 때 반복하는 Euler 스텝 수입니다. 생성자와 로더가 이 값으로 스텝 크기와 시간 테이블을 정하므로, `RuntimeConfig`는 생성자의 세 번째 인자로 넘겨야 하고 생성한 뒤에는 바꾸면 안 됩니다.

## vAi 그래프

vAi에서 모델은 노드를 `-`로 이어 만든 DAG입니다. 세부사항은 [노드](../03-usage/node.md)와 [노드그룹](../03-usage/node-group.md) 문서를 참조.

*`B`: 카메라 수, `T` 지시문 토큰 수, `D` 언어 임베딩 차원(2048), `Da` 행동 임베딩 차원(1024), `A` 행동 차원(32), `C` 청크 길이(50).*

### 모델 아키텍처

```cpp
input("inImages") - patchEmbed - visionEncoder - projector - mergeCameras - "in0"/catCamsPrompt;
input("inPromptTokens")                                     - promptEmbed - "in1"/catCamsPrompt;
catCamsPrompt - languageEncoder;

input("inState") - stateProj - output("outStateToken");

input("inStateToken")                                                       - "in0"/catStateActs;
input("inActions") - actionProj              - actionTimeIn - actionTimeOut - "in1"/catStateActs;
input("inActions") - stepTimeBias - "residual"/actionTimeIn;
input("inActions")               - "residual"/velocityHead;
catStateActs - actionDecoder - sliceActions - velocityHead - output("outActions");
```

그래프 하나에 두 단계가 들어 있고, 단계마다 넣는 입력이 다릅니다.

**prefill**(`Net::prefill`)은 이미지, 지시문, 상태를 받아 한 번 실행합니다.

- `patchEmbed`와 `visionEncoder`가 카메라 `B`대를 배치로 처리해 `[B, 256, 1152]`를 만들고, `projector`가 `[B, 256, D]`로 투영한 뒤 `mergeCameras`가 `[B * 256, D]`로 폅니다.
- `catCamsPrompt`가 지시문 임베딩을 뒤에 이어 `[B * 256 + T, D]`를 만들고, `languageEncoder`가 이 프리픽스로 KV 캐시를 채웁니다. 프리픽스는 서로를 모두 봅니다.
- `languageEncoder`의 마지막 블록 `gate`는 항상 닫혀 있어, 그 블록의 `norm2`와 FFN, `finalNorm`은 실행하지 않습니다. 필요한 것은 KV 캐시뿐입니다.
- `stateProj`가 상태 `[1, A]`를 상태 토큰 `[1, Da]`로 투영해 둡니다.

**decode**(`Net::decode`)는 노이즈로 초기화한 행동 `[C, A]`를 받아 `denoiseSteps`번 실행하고, 같은 텐서에 결과를 덮어씁니다.

- `actionProj`가 행동을 `[C, Da]`로 투영합니다. `actionTimeIn`은 여기에 이번 스텝의 시간 항(`stepTimeBias`)을 `residual`로 더한 뒤 SiLU를 걸고, `actionTimeOut`이 다시 투영합니다.
- `catStateActs`가 상태 토큰을 앞에 붙여 `[1 + C, Da]`를 만들고, `actionDecoder`가 프리픽스 KV 캐시를 참조해 계산합니다.
- `sliceActions`가 상태 토큰 행을 떼어 `[C, Da]`만 남기고, `velocityHead`가 속도 `v`를 `[C, A]`로 투영합니다.

`velocityHead`는 입력 행동을 `residual`로 받고 `scalarScale`이 `-1 / denoiseSteps`라서, 출력이 곧 Euler 한 스텝 `x - v / denoiseSteps`입니다. 별도 덧셈 노드 없이 `t = 1`(노이즈)에서 `t = 0`(행동)으로 적분합니다.

`visionEncoder`, `languageEncoder`, `actionDecoder`는 공통 그룹 [`Transformer`](../../../vai/groups/transformer.h)입니다. 블록을 차례로 잇고 마지막에 `finalNorm`을 둡니다.

```cpp
for (size_t i = 1; i < blocks.size(); ++i)
    blocks[i - 1] - blocks[i];
blocks.back() - finalNorm;
```

- **`visionEncoder`**: SigLIP So400m/14입니다. `Transformer<Classic>`이라 `LayerNorm`과 `FeedForward`를 쓰고, `allowBatch`로 카메라 축을 유지해 카메라마다 자기 패치끼리만 어텐션합니다. KV 캐시는 없습니다.
- **`languageEncoder`**: Gemma 2B입니다. `Transformer<Default>`라 `RmsNorm`과 `GatedFeedForward`를 쓰고, `maxContext()` 위치의 KV 캐시를 둡니다.
- **`actionDecoder`**: Gemma 300M입니다. `Transformer<Default>`이고, `languageEncoder`와 KV 캐시를 함께 씁니다.

#### 시간 임베딩

체크포인트의 `action_time_mlp_in`은 `[행동 임베딩, 시간 임베딩]`을 이어 받는 `[Da, 2Da]` 투영입니다. 로더가 이를 입력 방향으로 둘로 나눕니다.

- 행동 절반은 `actionTimeIn`의 가중치 `[Da, Da]`가 됩니다.
- 시간 절반과 bias는 스텝 `k`마다 `W_t · emb(t_k) + b`(`t_k = 1 - k / denoiseSteps`)를 미리 계산해 `stepTimeBias`([`CyclicTableReader`](../06-nodes.md#cyclictablereader))의 테이블 `[denoiseSteps, 1, Da]`에 넣습니다.

`stepTimeBias`는 실행될 때마다 다음 행을 내보내므로, decode 루프가 호스트에서 시간 값을 쓰지 않아도 스텝마다 맞는 시간 항이 들어갑니다.

#### KV 캐시 공유와 마스크

`languageEncoder`와 `actionDecoder`는 블록마다 같은 `K_cache`, `V_cache`를 쓰고, KV 캐시 토큰 카운터도 함께 씁니다.

- prefill은 카운터를 0으로 두고 프리픽스를 캐시 앞쪽에 씁니다.
- decode는 카운터를 프리픽스 길이에 고정한 채 스텝을 반복합니다. 스텝마다 상태 토큰과 행동 토큰의 K/V를 프리픽스 바로 뒤에 쓰므로, 앞 스텝이 쓴 값을 덮어씁니다.

`actionDecoder`에는 블록 마스크를 설정합니다. 상태 토큰 `[0, 1)`과 행동 토큰 `[1, 1 + C)`를 두 블록으로 나누고 둘 다 causal이 아니므로, 상태 토큰은 프리픽스와 자기 자신만 보고, 행동 토큰은 프리픽스, 상태 토큰, 행동 토큰 전체를 봅니다.

### NodeGroup

π0가 사용하는 [노드그룹](../03-usage/node-group.md)입니다. 모두 [공통 그룹](../../../vai/groups/)에 있습니다.
`Transformer`가 `TransformerBlock`를 쌓고, `TransformerBlock`가 `MultiHeadAttention`과 `FeedForward`(비전) 또는 `GatedFeedForward`(언어, 행동)를 담는 중첩 구조입니다.

비전 인코더와 언어 인코더는 PaliGemma 3B와 같은 구조입니다([PaliGemma](paligemma.md)).

#### PatchEmbedding

카메라 이미지를 `patchSize`(14) 크기의 겹치지 않는 패치로 잘라, 패치마다 한 행으로 편 뒤 `Linear` 하나로 투영합니다. `allowBatch`가 켜져 있어 카메라 축 `B`를 끝까지 유지합니다.

```cpp
cropRows - cropCols - splitPatches - perPatches - mergePatches - project;
// [B, 3, 224, 224] -> [B, 3, 16, 14, 16, 14] -> [B, 16, 16, 3, 14, 14]
//                  -> [B, 256, 588] -> [B, 256, 1152]
```

224가 14의 배수라 `cropRows`와 `cropCols`는 입력을 그대로 넘깁니다. SigLIP의 위치 임베딩은 로더가 `project`의 bias에 더해 `[256, 1152]` 행렬로 넣으므로, 위치를 더하는 노드가 따로 없습니다.

#### TransformerBlock

```cpp
input - norm1 -    mhAttn - lastRow - gate/"then" - norm2 -    ffn;
input - "residual"/mhAttn - lastRow - gate/"then" - "residual"/ffn;
```

별도 `Add`를 두지 않고 각 서브층의 마지막 투영이 `acceptResidual`로 받아 커널 안에서 더합니다.
`lastRow`와 `gate`는 설정에 따라 생기는 `optional` 멤버이고, 없으면 연결에서 빠집니다. π0는 `lastRow`를 쓰지 않고, `gate`는 `languageEncoder`의 마지막 블록에만 둡니다.

#### MultiHeadAttention

```cpp
input - projQ - splitQ - normQ - perQ - ropeQ - "Q"/attn;
input - projK - splitK - normK - perK - ropeK - "K"/attn;
input - projV - splitV         - perV         - "V"/attn - mergeOut - projOut;
```

- **`visionEncoder`**: Q, K, V가 모두 16개 헤드입니다. RoPE가 없어 `ropeQ`, `ropeK`가 생기지 않습니다.
- **`languageEncoder`, `actionDecoder`**: GQA로 쿼리 8개 헤드가 K/V 1개 헤드를 공유하고, RoPE는 Q와 K에만 적용됩니다. `actionDecoder`는 입력 폭이 1024라 `projQ`가 1024 → 2048, `projK`와 `projV`가 1024 → 256으로 투영하고, `projOut`이 2048 → 1024로 되돌립니다.

세 스택 모두 QK 정규화를 쓰지 않으므로 `normQ`, `normK`는 생기지 않습니다. `projOut`이 `acceptResidual = true`라 블록의 잔차를 `residual`로 받아 투영 결과에 더합니다.

`Rope`는 KV 캐시 토큰 카운터를 읽으므로, decode의 상태 토큰과 행동 토큰은 프리픽스 뒤의 절대 위치로 회전합니다.

#### FeedForward

`visionEncoder`의 FFN입니다. `Linear` 둘을 연결하고, GELU(tanh 근사)는 앞쪽 `Linear`의 에필로그로 붙어 GEMM 안에서 처리됩니다.

```cpp
fc1 - fc2;   // 1152 -> 4304 -> 1152
```

`fc2`는 `acceptResidual = true`로 블록의 잔차를 `residual`로 받습니다.

#### GatedFeedForward

`languageEncoder`와 `actionDecoder`의 FFN입니다. `gate`와 `value`가 같은 입력을 읽고 가중치만 다르므로 `Linear` 하나로 2배 폭까지 한 번에 계산합니다. `GatedMul`가 앞 절반에 GELU(tanh 근사)를 걸어 뒤 절반과 곱합니다(GeGLU).

```cpp
gateValue - gatedMul - out;   // languageEncoder: 2048 -> 32768 -> 16384 -> 2048
                              // actionDecoder:   1024 -> 8192  -> 4096  -> 1024
```

`out`도 `acceptResidual = true`로 블록의 잔차를 `residual`로 받습니다.

### 사용 노드

[`TokenEmbedding`](../06-nodes.md#tokenembedding) ·
[`LayerNorm`](../06-nodes.md#layernorm) ·
[`RmsNorm`](../06-nodes.md#rmsnorm) ·
[`Linear`](../06-nodes.md#linear) ·
[`Reshape`](../06-nodes.md#reshape) ·
[`Permute`](../06-nodes.md#permute) ·
[`Slice`](../06-nodes.md#slice) ·
[`Concat`](../06-nodes.md#concat) ·
[`Rope`](../06-nodes.md#rope) ·
[`Attention`](../06-nodes.md#attention) ·
[`GatedMul`](../06-nodes.md#gatedmul) ·
[`Identity`](../06-nodes.md#identity) ·
[`Gate`](../06-nodes.md#gate) ·
[`CyclicTableReader`](../06-nodes.md#cyclictablereader)

## 체크포인트

체크포인트는 `data/ckpts/pi0/lerobot/3B/`(`ModelConfig::variant`)에, 토크나이저는 `data/tokenizers/paligemma/`에 둡니다. 토크나이저는 PaliGemma의 것을 그대로 씁니다.

```
data/
├── ckpts/pi0/lerobot/
│   └── 3B/
│       └── model.safetensors
└── tokenizers/paligemma/
    └── tokenizer.json
```

LeRobot이 옮긴 [`lerobot/pi0_base`](https://huggingface.co/lerobot/pi0_base)에서 받아 별도 변환 없이 그대로 씁니다. 가중치는 [Gemma Terms of Use](https://ai.google.dev/gemma/terms)를 따릅니다.

`lerobot/pi0_base`에는 토크나이저가 없어, `tokenizer.json`은 [`google/paligemma-3b-mix-224`](https://huggingface.co/google/paligemma-3b-mix-224)에서 받습니다. 이 저장소는 게이트가 걸려 있어, 저장소 페이지에서 라이선스에 동의한 뒤 HuggingFace 액세스 토큰(`HF_TOKEN`)을 넘겨야 합니다.

```bash
mkdir -p data/ckpts/pi0/lerobot/3B data/tokenizers/paligemma
curl -L -o data/ckpts/pi0/lerobot/3B/model.safetensors \
  https://huggingface.co/lerobot/pi0_base/resolve/main/model.safetensors
curl -L -H "Authorization: Bearer $HF_TOKEN" -o data/tokenizers/paligemma/tokenizer.json \
  https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/tokenizer.json
```

HuggingFace 저장소의 **Files and versions** 탭에서 직접 내려받거나 위 명령으로 받으면 됩니다.

### 가중치 로딩

체크포인트를 그래프 슬롯에 매핑하는 코드는 [`models/production/pi0/lerobot-weights.h`](../../../models/production/pi0/lerobot-weights.h)에 있습니다.
실행하는 쪽은 한 줄이면 됩니다. 경로는 특수화가 조립합니다(`VAI_DATA_DIR "/ckpts/pi0/lerobot/" + modelCfg.variant`). 배포명 태그의 원리는 [커스텀 모델 작성의 가중치 로딩 함수 작성](../contributing/models.md#33-가중치-로딩-함수-작성) 참조.

```cpp
loadWeights<"lerobot">(net);
```

## 실행

### 폴더 구조

```
models/production/pi0/
├── net.h                         모델 정의
├── lerobot-weights.h             LeRobot(HuggingFace `lerobot/pi0_base`) 체크포인트 매핑
├── processing.h                  flow matching 초기 노이즈 (sampleNoise)
├── demo.cpp                      demo-pi0, 카메라 3대 + 지시문 + 상태로 행동 청크 하나 생성 + 시간 측정
└── CMakeLists.txt

data/
├── ckpts/pi0/lerobot/
│   └── 3B/                       ModelConfig::variant
│       └── model.safetensors
├── tokenizers/paligemma/
│   └── tokenizer.json
└── images/                       입력 이미지
```

### 빌드 및 실행

```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo
```

*`demo-pi0`는 기본으로 빌드됩니다.*

### demo-pi0

```bash
./build/bin/demo-pi0
# Windows (Visual Studio, Ninja Multi-Config): build\bin\RelWithDebInfo\demo-pi0.exe
```

[`demo.cpp`](../../../models/production/pi0/demo.cpp)는 `data/images/`의 카메라 이미지 3장, 지시문 `prompt`, 임의의 관절 상태를 넣어 실행합니다. prefill과 denoise 시간을 각각 측정해 행동 청크의 앞 5스텝 × 8차원과 함께 출력합니다.
