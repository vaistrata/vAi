# PaliGemma

PaliGemma(Beyer et al., 2024)는 Google에서 개발한 비전-언어 모델입니다. SigLIP So400m/14 비전 인코더가 이미지를 토큰으로 바꾸고, 선형 투영을 거쳐 Gemma 2B 디코더의 프롬프트 앞에 붙입니다. 이미지 토큰과 프롬프트로 이루어진 프리픽스는 서로를 모두 보고, 생성하는 토큰만 앞쪽을 봅니다. 모델은 257,152개 어휘(텍스트 256,000개, `<loc>` 1,024개, `<seg>` 128개)의 확률 분포를 계산해 다음 토큰을 예측합니다.

**참고자료**: [논문](https://arxiv.org/abs/2407.07726), [체크포인트(HuggingFace)](https://huggingface.co/google/paligemma-3b-mix-224), [저장소](https://github.com/google-research/big_vision)

## 설정

```cpp
namespace paligemma = model::production::paligemma;

auto modelCfg = paligemma::ModelConfig::_3B_224();

paligemma::Net net({
    .wDefaultDtype = Dtype::Fp16,
}, modelCfg, {
    .maxRuntimeContext = 2048,
});
```

### NeuralNetConfig

PaliGemma는 기본값을 그대로 쓰고, 로더가 각 norm의 파라미터만 fp32로 읽습니다.
세부사항은 [신경망](../03-usage/neural-net.md) 참조.

### ModelConfig

두 팩토리는 입력 해상도만 다릅니다.

| 팩토리 | `variant` | `visionTrf.imageSize` | 이미지 토큰 수 |
|---|---|---|---|
| `_3B_224()` | `"3B-224"` | 224 | 256 (16 × 16) |
| `_3B_448()` | `"3B-448"` | 448 | 1024 (32 × 32) |

`visionTrf`는 [`models/common.h`](../../../models/common.h)의 `so400m14(224)` 또는 `so400m14(448)`이 만드는 `VisionTransformerConfig`이고, `languageTrf`는 `gemma("2B")`가 만드는 `TransformerConfig`입니다.

| 멤버 | `nBlocks` | `embedDim` | `headDim` | `numQHeads` | `numKVHeads` | `ffnHiddenDim` |
|---|---|---|---|---|---|---|
| `visionTrf` (SigLIP So400m/14) | 27 | 1152 | 72 | 16 | 16 | 4304 |
| `languageTrf` (Gemma 2B) | 18 | 2048 | 256 | 8 | 1 | 16384 |

`vocabSize` 257,152, `maxTrainedContext` 8,192는 공통입니다.

### RuntimeConfig

KV 캐시의 dtype과 크기를 정합니다. `maxRuntimeContext`를 생략하면 `maxTrainedContext` 값을 그대로 쓰고, 더 작은 값을 지정하면 그만큼만 할당해 메모리를 아낍니다. `maxTrainedContext`보다 큰 값은 `maxTrainedContext`로 줄입니다. 데모는 2,048로 낮춰 잡습니다.

| 필드 | 기본값 |
|---|---|
| `kvCacheDtype` | `Dtype::Fp16` |
| `maxRuntimeContext` | `uint32_t(-1)`: `maxTrainedContext`까지 |

이미지 토큰도 KV 캐시를 차지합니다. `_3B_448()`은 이미지 한 장이 1,024개 위치를 쓰므로, `maxRuntimeContext`는 이미지 토큰, 프롬프트, 생성할 토큰을 합한 수 이상이어야 합니다.

## vAi 그래프

vAi에서 모델은 노드를 `-`로 이어 만든 DAG입니다. 세부사항은 [노드](../03-usage/node.md)와 [노드그룹](../03-usage/node-group.md) 문서를 참조.

*`N`: 이미지 토큰 수, `T` 프롬프트 토큰 수, `Dv` 비전 임베딩 차원(1152), `D` 언어 임베딩 차원(2048), `S` 입력 해상도.*

### 모델 아키텍처

```cpp
input("inImage")  - patchEmbed - encoder - projector - "in0"/catTokens;
input("inTokenIds")                      - embedding - "in1"/catTokens;
catTokens - decoder - lmHead - topkSampling - output("nextTokenId");
```

`patchEmbed`가 `[3, S, S]` 이미지를 `[N, Dv]` 패치 토큰으로 바꾸고, `encoder`(SigLIP)를 거친 뒤 `projector`가 `[N, D]`로 투영합니다. `catTokens`([`Concat`](../06-nodes.md#concat))는 이미지 토큰 뒤에 프롬프트 임베딩을 이어 `[N + T, D]`를 만듭니다. `allowPartialInputs`가 켜져 있어, 이미지 없이 직전 토큰만 넣는 decode 스텝에서는 `in1`만 받고 비전 쪽 노드는 실행하지 않습니다.

`embedding`은 출력에 `sqrt(D)`를 곱하고, `lmHead`는 임베딩 가중치를 함께 씁니다.

`encoder`와 `decoder`는 공통 그룹 [`Transformer`](../../../vai/groups/transformer.h)입니다. 블록을 차례로 잇고 마지막에 `finalNorm`을 둡니다.

```cpp
for (size_t i = 1; i < blocks.size(); ++i)
    blocks[i - 1] - blocks[i];
blocks.back() - finalNorm;
```

- **`encoder`**: `Transformer<Classic>`입니다. `LayerNorm`과 `FeedForward`를 쓰고, KV 캐시 없이 이미지 한 장의 패치끼리만 어텐션합니다.
- **`decoder`**: `Transformer<Default>`입니다. `RmsNorm`과 `GatedFeedForward`를 쓰고, `maxRuntimeContext` 위치의 KV 캐시를 둡니다. causal mask를 설정하지 않으므로 prefill에서 이미지 토큰과 프롬프트가 서로를 모두 봅니다(prefix-LM).

### NodeGroup

PaliGemma가 사용하는 [노드그룹](../03-usage/node-group.md)입니다. 모두 [공통 그룹](../../../vai/groups/)에 있습니다.
`Transformer`가 `TransformerBlock`를 쌓고, `TransformerBlock`가 `MultiHeadAttention`과 `FeedForward`(비전) 또는 `GatedFeedForward`(언어)를 담는 중첩 구조입니다.

#### PatchEmbedding

이미지를 `patchSize`(14) 크기의 겹치지 않는 패치로 잘라, 패치마다 한 행으로 편 뒤 `Linear` 하나로 투영합니다.

```cpp
cropRows - cropCols - splitPatches - perPatches - mergePatches - project;
// [3, S, S] -> [3, G*14, S] -> [3, G*14, G*14] -> [3, G, 14, G, 14]
//           -> [G, G, 3, 14, 14] -> [N, 588] -> [N, 1152]    (G = S / 14, N = G * G)
```

`S`가 14의 배수라 `cropRows`와 `cropCols`는 입력을 그대로 넘깁니다. SigLIP의 위치 임베딩은 로더가 `project`의 bias에 더해 `[N, Dv]` 행렬로 넣으므로, 위치를 더하는 노드가 따로 없습니다.

#### TransformerBlock

```cpp
input - norm1 -    mhAttn - lastRow - gate/"then" - norm2 -    ffn;
input - "residual"/mhAttn - lastRow - gate/"then" - "residual"/ffn;
```

별도 `Add`를 두지 않고 각 서브층의 마지막 투영이 `acceptResidual`로 받아 커널 안에서 더합니다.
`lastRow`와 `gate`는 설정에 따라 생기는 `optional` 멤버이고, 없으면 연결에서 빠집니다. PaliGemma는 `decoder`의 마지막 블록에만 `lastRow`를 두어, 다음 토큰을 정하는 마지막 행만 FFN으로 보냅니다. `gate`는 쓰지 않습니다.

#### MultiHeadAttention

```cpp
input - projQ - splitQ - normQ - perQ - ropeQ - "Q"/attn;
input - projK - splitK - normK - perK - ropeK - "K"/attn;
input - projV - splitV         - perV         - "V"/attn - mergeOut - projOut;
```

- **`encoder`**: Q, K, V가 모두 16개 헤드입니다. RoPE가 없어 `ropeQ`, `ropeK`가 생기지 않습니다.
- **`decoder`**: GQA로 쿼리 8개 헤드가 K/V 1개 헤드를 공유합니다. RoPE(base 10,000)는 Q와 K에만 적용됩니다.

두 경우 모두 QK 정규화를 쓰지 않으므로 `normQ`, `normK`는 생기지 않습니다. `projOut`이 `acceptResidual = true`라 블록의 잔차를 `residual`로 받아 투영 결과에 더합니다.

`Rope`는 KV 캐시 토큰 카운터를 공유해서, decode 스텝에서도 이번 forward 안의 오프셋이 아니라 누적 절대 위치로 회전합니다.

#### FeedForward

`encoder`의 FFN입니다. `Linear` 둘을 연결하고, GELU(tanh 근사)는 앞쪽 `Linear`의 에필로그로 붙어 GEMM 안에서 처리됩니다.

```cpp
fc1 - fc2;   // 1152 -> 4304 -> 1152
```

`fc2`는 `acceptResidual = true`로 블록의 잔차를 `residual`로 받습니다.

#### GatedFeedForward

`decoder`의 FFN입니다. `gate`와 `value`가 같은 입력을 읽고 가중치만 다르므로 `Linear` 하나로 2배 폭까지 한 번에 계산합니다. `GatedMul`가 앞 절반에 GELU(tanh 근사)를 걸어 뒤 절반과 곱합니다(GeGLU).

```cpp
gateValue - gatedMul - out;   // 2048 -> 32768 -> 16384 -> 2048
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
[`TopKSampling`](../06-nodes.md#topksampling)

## 체크포인트

체크포인트는 `data/ckpts/paligemma/google/` 아래 해상도별 폴더(`ModelConfig::variant`)에, 토크나이저는 `data/tokenizers/paligemma/`에 둡니다.
`tokenizer.json`은 두 해상도가 함께 씁니다. 로더가 폴더 안의 `*.safetensors`를 모두 읽으므로, 여러 파일로 나뉜 체크포인트도 그대로 씁니다.

```
data/
├── ckpts/paligemma/google/
│   ├── 3B-224/
│   │   ├── model-00001-of-00003.safetensors
│   │   ├── model-00002-of-00003.safetensors
│   │   └── model-00003-of-00003.safetensors
│   └── 3B-448/
│       └── ...
└── tokenizers/paligemma/
    └── tokenizer.json
```

해상도별 mix 저장소에서 받아 별도 변환 없이 그대로 씁니다. 가중치는 [Gemma Terms of Use](https://ai.google.dev/gemma/terms)를 따릅니다.

데모는 224를 씁니다. `google/paligemma-3b-mix-*` 저장소는 게이트가 걸려 있어, 저장소 페이지에서 라이선스에 동의한 뒤 HuggingFace 액세스 토큰(`HF_TOKEN`)을 넘겨야 합니다.

```bash
mkdir -p data/ckpts/paligemma/google/3B-224 data/tokenizers/paligemma
curl -L -H "Authorization: Bearer $HF_TOKEN" -o data/tokenizers/paligemma/tokenizer.json \
  https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/tokenizer.json
for f in model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors; do
  curl -L -H "Authorization: Bearer $HF_TOKEN" -o data/ckpts/paligemma/google/3B-224/$f \
    https://huggingface.co/google/paligemma-3b-mix-224/resolve/main/$f
done
```

HuggingFace 저장소의 **Files and versions** 탭에서 직접 내려받거나 위 명령으로 받으면 됩니다.

> 448은 같은 방법으로 [`google/paligemma-3b-mix-448`](https://huggingface.co/google/paligemma-3b-mix-448)에서 받아 `3B-448/`에 둡니다. 파일 구성은 224와 같습니다.

### 가중치 로딩

체크포인트를 그래프 슬롯에 매핑하는 코드는 [`models/production/paligemma/google-weights.h`](../../../models/production/paligemma/google-weights.h)에 있습니다.
실행하는 쪽은 한 줄이면 됩니다. 경로는 특수화가 조립하므로(`VAI_DATA_DIR "/ckpts/paligemma/google/" + modelCfg.variant`) 해상도만 정하면 됩니다. 배포명 태그의 원리는 [커스텀 모델 작성의 가중치 로딩 함수 작성](../contributing/models.md#33-가중치-로딩-함수-작성) 참조.

```cpp
loadWeights<"google">(net);
```

## 실행

### 폴더 구조

```
models/production/paligemma/
├── net.h                         모델 정의
├── google-weights.h              Google(HuggingFace `google/paligemma-3b-mix-*`) 체크포인트 매핑
├── demo.cpp                      demo-paligemma, 이미지 한 장에 대한 한 프롬프트 완성 + 시간 측정
└── CMakeLists.txt

data/
├── ckpts/paligemma/google/
│   └── <variant>/                ModelConfig::variant (3B-224, 3B-448)
│       └── *.safetensors
├── tokenizers/paligemma/
│   └── tokenizer.json
└── images/                       입력 이미지
```

### 빌드 및 실행

```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo
```

*`demo-paligemma`는 기본으로 빌드됩니다.*

### demo-paligemma

```bash
./build/bin/demo-paligemma
# Windows (Visual Studio, Ninja Multi-Config): build\bin\RelWithDebInfo\demo-paligemma.exe
```

224로 `data/images/dog.jpg`와 [`demo.cpp`](../../../models/production/paligemma/demo.cpp)의 `prompt`(`"caption en"`)를 함께 넣어 실행합니다. `prefill`과 `decode` 시간을 각각 측정해 생성된 답과 함께 출력합니다.
