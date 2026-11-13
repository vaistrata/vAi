# GPT-2

GPT-2(Radford et al., 2019)는 OpenAI에서 개발한 디코더 전용 트랜스포머 기반 언어 모델입니다. causal self-attention과 GELU feed-forward pre-norm 블록을 쌓은 구조입니다. 모델은 50,257개 어휘의 확률 분포를 계산해 다음 토큰을 예측합니다.

**참고자료**: [논문](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [체크포인트(HuggingFace)](https://huggingface.co/openai-community/gpt2), [저장소](https://github.com/openai/gpt-2)

## 설정

```cpp
namespace gpt2 = model::production::gpt2;

auto modelCfg = gpt2::ModelConfig::_124M();

gpt2::Net net({
    .device       = device,
    .defaultDtype = Dtype::Fp16,
    .ioDtypeHint  = Dtype::Fp32,
}, modelCfg);
```

### NeuralNetConfig

GPT-2는 기본값을 그대로 쓰고, 로더가 각 `LayerNorm`의 계수만 fp32로 읽습니다.
세부사항은 [신경망](../03-usage/neural-net.md) 참조.

### ModelConfig

| 팩토리 | `name` | `embedDim` | `numHeads` | `numLayers` |
|---|---|---|---|---|
| `_124M()` | `"124M"` | 768 | 12 | 12 |
| `_355M()` | `"355M"` | 1024 | 16 | 24 |
| `_774M()` | `"774M"` | 1280 | 20 | 36 |
| `_1_5B()` | `"1.5B"` | 1600 | 25 | 48 |

`vocabSize` 50,257, `maxTrainedContext` 1,024는 공통입니다.

### RuntimeConfig

KV 캐시를 얼마나 확보할지만 정합니다. 생략하면 `maxTrainedContext` 값을 그대로 쓰고, 값을 지정하면 그만큼만 할당해 메모리를 아낍니다.

| 필드 | 기본값 |
|---|---|
| `maxRuntimeContext` | `0` |

## vAi 그래프

vAi에서 모델은 노드를 `-`로 이어 만든 DAG입니다. 세부사항은 [노드](../03-usage/node.md)와 [노드그룹](../03-usage/node-group.md) 문서를 참조.

*`Tq`: 입력 토큰 수, `D` 임베딩 차원, `H` 헤드 수.*

### 모델 아키텍처

```cpp
input("inTokenIds") - embedding - layers[0];
for (size_t i = 1; i < layers.size(); ++i)
    layers[i - 1] - layers[i];
layers.back() - lastRowSlice - finalNorm - lmHead - topkSampling - output("nextTokenId");
```

### NodeGroup

GPT-2를 구현하며 정의한 [노드그룹](../03-usage/node-group.md)입니다.
`TransformerLayer`가 `MultiHeadAttention`과 `FeedForward`를 담는 중첩 구조입니다.

#### TransformerLayer

```cpp
input - norm1 - "in0"/mhAttn      - norm2 - "in0"/ffn;
input         - "residual"/mhAttn         - "residual"/ffn;
```

별도 `Add`를 두지 않고 각 서브층의 마지막 투영이 `acceptResidual`로 받아 커널 안에서 더합니다.

#### MultiHeadAttention

Q, K, V를 `{D, 3D}` 투영 하나로 계산한 뒤 헤드 축에서 세 조각으로 잘라냅니다.

```cpp
projQKV - splitHeads - perQKV - sliceQ - "Q"/attn - mergeOut - projOut;
                     perQKV - sliceK - "K"/attn;
                     perQKV - sliceV - "V"/attn;
```

`projOut`이 `acceptResidual = true`라 블록의 잔차를 `residual`로 받아 투영 결과에 더합니다.

#### FeedForward

`Linear` 둘을 연결한 그룹입니다. GELU는 별도 노드가 아니라 앞쪽 `Linear`의 에필로그로 붙어 GEMM 안에서 처리됩니다.

```cpp
fc1 - fc2;   // 768 -> 3072 -> 768
```

`fc2`도 `acceptResidual = true`로, 블록의 잔차를 `residual`로 받습니다.

### 사용 노드

[`TokenEmbedding`](../06-nodes.md#tokenembedding) ·
[`LayerNorm`](../06-nodes.md#layernorm) ·
[`Linear`](../06-nodes.md#linear) ·
[`Reshape`](../06-nodes.md#reshape) ·
[`Permute`](../06-nodes.md#permute) ·
[`Attention`](../06-nodes.md#attention) ·
[`Identity`](../06-nodes.md#identity) ·
[`Slice`](../06-nodes.md#slice) ·
[`TopKSampling`](../06-nodes.md#topksampling)

## 체크포인트

체크포인트는 `data/ckpts/gpt2/openai/` 아래 사이즈별 폴더(`ModelConfig::variant`)에, 토크나이저는 `data/tokenizers/gpt2/`에 둡니다.
`tokenizer.json`은 네 크기가 함께 씁니다.

```
data/
├── ckpts/gpt2/openai/
│   ├── 124M/
│   │   └── model.safetensors
│   ├── 355M/
│   │   └── model.safetensors
│   └── ...
└── tokenizers/gpt2/
    └── tokenizer.json
```

해당 HuggingFace 저장소에서 받아 별도 변환 없이 그대로 씁니다. 가중치 라이선스는 MIT입니다.

```bash
mkdir -p data/ckpts/gpt2/openai/124M data/tokenizers/gpt2
curl -L -o data/tokenizers/gpt2/tokenizer.json https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json
curl -L -o data/ckpts/gpt2/openai/124M/model.safetensors \
  https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
```

HuggingFace 저장소의 **Files and versions** 탭에서 `model.safetensors`와 `tokenizer.json`을 직접 내려받거나 위 명령으로 받으면 됩니다.

### 가중치 로딩

체크포인트를 그래프 슬롯에 매핑하는 코드는 [`models/production/gpt2/openai-weights.h`](../../../models/production/gpt2/openai-weights.h)에 있습니다.
실행하는 쪽은 한 줄이면 됩니다. 경로는 특수화가 조립하므로(`VAI_DATA_DIR "/ckpts/gpt2/openai/" + modelCfg.variant`)
사이즈만 정하면 됩니다. 배포명 태그의 원리는 [커스텀 모델 작성의 `배포명-weights.h`](../contributing/models.md#배포명-weightsh) 참조.

```cpp
loadWeights<"openai">(net);
```

## 실행

### 폴더 구조

```
models/production/gpt2/
├── net.h                         모델 정의 및 NodeGroup 정의
├── openai-weights.h              OpenAI(HuggingFace `openai-community/*`) 체크포인트 매핑
├── demo.cpp                      demo-gpt2: 한 프롬프트 완성 + 시간 측정
└── CMakeLists.txt

data/
├── ckpts/gpt2/openai/
│   └── <variant>/                    ModelConfig::variant: 124M, 355M, 774M, 1.5B
│       └── model.safetensors
└── tokenizers/gpt2/
    └── tokenizer.json
```

### 빌드 및 실행

```bash
cmake -B build
cmake --build build --config RelWithDebInfo

# Run
./build/bin/demo-gpt2
```

`prompt`에 적은 문장을 이어 새 문장을 생성합니다. `prefill`과 `decode` 시간을 각각 측정해 생성된 문장과 함께 출력합니다.

## 벤치마크 요약

llama.cpp(Vulkan)와 vAi를 **prompt 512 토큰** 기준으로 비교한 결과입니다. GPU와 prompt 길이별 전체 결과, 측정 환경과 조건, 재현 방법은 [`benchmarks/gpt2/README.md`](../../../benchmarks/gpt2/README.md) 참조.

| | | 890M (Strix Point) | 8060S (Strix Halo) | UHD Graphics (TGL GT1) | RTX 3060 Laptop | RTX 5070 | RTX 4090 | RTX 5090 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vAi | prefill (t/s) | 10,899 | 50,777 | 1,201 | 61,236 | 154,785 | 269,911 | 308,638 |
|  | decode (t/s) | 130.5 | 500.9 | 94.06 | 743.2 | 1,319 | 1,631 | 1,756 |
| llama.cpp | prefill (t/s) | 7,803 | 41,159 | 1,065 | 50,910 | 131,056 | 155,877 | 153,821 |
|  | decode (t/s) | 133.6 | 595.4 | 80.90 | 592.5 | 1,020 | 979.0 | 940.5 |
