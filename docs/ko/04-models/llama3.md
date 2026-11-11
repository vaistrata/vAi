# Llama 3

Llama 3(Grattafiori et al., 2024)는 Meta에서 개발한 디코더 전용 트랜스포머 기반 언어 모델입니다. GQA self-attention과 SwiGLU, feed-forward, pre-norm 블록을 쌓은 구조입니다. 정규화는 RMSNorm, 위치 정보는 RoPE로 넣으며 편향(bias)은 쓰지 않습니다. 모델은 128,256개 어휘의 확률 분포를 계산해 다음 토큰을 예측합니다.

**참고자료**: [논문](https://arxiv.org/abs/2407.21783), [체크포인트(HuggingFace)](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), [저장소](https://github.com/meta-llama/llama-models)

## 설정

```cpp
namespace llama3 = model::production::llama3;

auto modelCfg = llama3::ModelConfig::_3_2_1B();

llama3::Net net({
    .device       = device,
    .defaultDtype = Dtype::Fp16,
    .ioDtypeHint  = Dtype::Fp32,
}, modelCfg, {
    .maxRuntimeContext = 4096,
});
```

### NeuralNetConfig

Llama 3는 기본값을 그대로 쓰고, 로더가 각 `RmsNorm`의 scale만 fp32로 읽습니다.
세부사항은 [신경망](../03-usage/neural-net.md) 참조.

### ModelConfig

| 팩토리 | `name` | `embedDim` | `ffnHiddenDim` | `numHeads` | `numKVHeads` | `numLayers` |
|---|---|---|---|---|---|---|
| `_3_2_1B()` | `"3.2-1B"` | 2048 | 8192 | 32 | 8 | 16 |
| `_3_2_3B()` | `"3.2-3B"` | 3072 | 8192 | 24 | 8 | 28 |

`vocabSize` 128,256, `maxTrainedContext` 131,072는 공통입니다. `ropeConfig`도 두 크기가 같은 값(`ropeCommon(32.0f)`)을 씁니다. `factor` 32는 Llama 3 계열의 RoPE 주파수 스케일링 계수입니다.

### RuntimeConfig

KV 캐시를 얼마나 확보할지만 정합니다. 생략하면 `maxTrainedContext` 값을 그대로 쓰고, 값을 지정하면 그만큼만 할당해 메모리를 아낍니다. 학습 컨텍스트 131,072를 그대로 잡으면 메모리를 감당할 수 없어 데모와 벤치에서는 4,096으로 낮춰 잡습니다 (llama.cpp 벤치의 `n_ctx`와 같은 값).

| 필드 | 기본값 |
|---|---|
| `maxRuntimeContext` | `0` |

## vAi 그래프

vAi에서 모델은 노드를 `-`로 이어 만든 DAG입니다. 세부사항은 [노드](../03-usage/node.md)와 [노드그룹](../03-usage/node-group.md) 문서를 참조.

*`Tq`: 입력 토큰 수, `D` 임베딩 차원, `H` 쿼리 헤드 수, `Hkv` KV 헤드 수.*

### 모델 아키텍처

```cpp
input("inTokenIds") - embedding - layers[0];
for (size_t i = 1; i < layers.size(); ++i)
    layers[i - 1] - layers[i];
layers.back() - lastRowSlice - finalNorm - lmHead - topkSampling - output("nextTokenId");
```

### NodeGroup

Llama 3를 구현하며 정의한 [노드그룹](../03-usage/node-group.md)입니다.
`TransformerLayer`가 `MultiHeadAttention`과 `GatedFeedForward`를 담는 중첩 구조입니다.

#### TransformerLayer

```cpp
input - norm1 - "in0"/mhAttn      - norm2 - "in0"/ffn;
input         - "residual"/mhAttn         - "residual"/ffn;
```

별도 `Add`를 두지 않고 각 서브층의 마지막 투영이 `acceptResidual`로 받아 커널 안에서 더합니다.

#### MultiHeadAttention

GQA를 쓰기 때문에 K와 V는 쿼리보다 적은 수의 헤드로 투영됩니다(`H == 32`, `Hkv == 8`). RoPE는 Q와 K에만 적용되고 V는 그대로 통과합니다.

```cpp
input - projQ - splitQ - perQ - ropeQ - "Q"/attn - mergeOut - projOut;
input - projK - splitK - perK - ropeK - "K"/attn;
input - projV - splitV - perV         - "V"/attn;
```

`projOut`이 `acceptResidual = true`라 블록의 잔차를 `residual`로 받아 투영 결과에 더합니다.

`Rope`는 KV 캐시 토큰 카운터를 공유해서, decode 스텝에서도 이번 forward 안의 오프셋이 아니라 누적 절대 위치로 회전합니다.

#### GatedFeedForward

`gate`와 `value`가 같은 입력을 읽고 가중치만 다르므로 `Linear` 하나로 2배 폭까지 한 번에 계산합니다. `GatedMul`가 앞 절반에 SiLU를 걸어 뒤 절반과 곱합니다.

```cpp
gateValue - gatedMul - out;   // 2048 -> 16384 -> 8192 -> 2048
```

`out`도 `acceptResidual = true`로, 블록의 잔차를 `residual`로 받습니다.

### 사용 노드

[`TokenEmbedding`](../06-nodes.md#tokenembedding) ·
[`RmsNorm`](../06-nodes.md#rmsnorm) ·
[`Linear`](../06-nodes.md#linear) ·
[`Reshape`](../06-nodes.md#reshape) ·
[`Permute`](../06-nodes.md#permute) ·
[`Rope`](../06-nodes.md#rope) ·
[`Attention`](../06-nodes.md#attention) ·
[`GatedMul`](../06-nodes.md#gatedmul) ·
[`Identity`](../06-nodes.md#identity) ·
[`Slice`](../06-nodes.md#slice) ·
[`TopKSampling`](../06-nodes.md#topksampling)

## 체크포인트

체크포인트는 `data/ckpts/llama3/meta/` 아래 사이즈별 폴더(`ModelConfig::variant`)에, 토크나이저는 `data/tokenizers/llama3/`에 둡니다.
`tokenizer.json`은 두 크기가 함께 씁니다.

```
data/
├── ckpts/llama3/meta/
│   ├── 3.2-1B/
│   │   └── model.safetensors
│   ├── 3.2-3B/
│   │   └── model.safetensors
│   └── ...
└── tokenizers/llama3/
    └── tokenizer.json
```

해당 HuggingFace 저장소에서 받아 별도 변환 없이 그대로 씁니다. 가중치는 [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE)를 따릅니다.

```bash
mkdir -p data/ckpts/llama3/meta/3.2-1B data/tokenizers/llama3
curl -L -o data/tokenizers/llama3/tokenizer.json https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/tokenizer.json
curl -L -o data/ckpts/llama3/meta/3.2-1B/model.safetensors \
  https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/model.safetensors
```

HuggingFace 저장소의 **Files and versions** 탭에서 `model.safetensors`와 `tokenizer.json`을 직접 내려받거나 위 명령으로 받으면 됩니다.

> 원본 `meta-llama/*`는 게이트가 걸려 있어 승인과 토큰이 필요합니다. 위 명령은 같은 텐서를 담은 [unsloth 미러](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct)를 씁니다. 미러로 받아도 라이선스는 같습니다. **채팅 토큰이 학습된 Instruct 체크포인트가 필요합니다**.

### 가중치 로딩

체크포인트를 그래프 슬롯에 매핑하는 코드는 [`models/production/llama3/meta-weights.h`](../../../models/production/llama3/meta-weights.h)에 있습니다.
실행하는 쪽은 한 줄이면 됩니다. 경로는 특수화가 조립하므로(`VAI_DATA_DIR "/ckpts/llama3/meta/" + modelCfg.variant`) 사이즈만 정하면 됩니다. 배포명 태그의 원리는 [커스텀 모델 작성의 `배포명-weights.h`](../05-contributing-models.md#배포명-weightsh) 참조.

```cpp
loadWeights<"meta">(net);
```

## 실행

### 폴더 구조

```
models/production/llama3/
├── net.h                         모델 정의 및 NodeGroup 정의
├── meta-weights.h                Meta(HuggingFace `meta-llama/*`) 체크포인트 매핑
├── demo.cpp                      demo-llama3, 한 프롬프트 완성 + 시간 측정
└── CMakeLists.txt

examples/chat-llama3/
├── main.cpp                      chat-llama3, 다중 턴 대화
└── CMakeLists.txt

data/
├── ckpts/llama3/meta/
│   └── <variant>/                    ModelConfig::variant (3.2-1B, 3.2-3B)
│       └── model.safetensors
└── tokenizers/llama3/
    └── tokenizer.json
```

### 빌드 및 실행

```bash
cmake -B build -DVAI_BUILD_EXAMPLES=ON
cmake --build build --config RelWithDebInfo
```

### demo-llama3

```bash
./build/bin/demo-llama3
```

[`demo.cpp`](../../../models/production/llama3/demo.cpp)의 `prompt`에 적은 문장을 이어 새 문장을 생성합니다. `prefill`과 `decode` 시간을 각각 측정해 생성된 문장과 함께 출력합니다.

### chat-llama3

```bash
./build/bin/chat-llama3
```

[`main.cpp`](../../../examples/chat-llama3/main.cpp)의 다중 턴 대화입니다. 대화 이력이 KV 캐시에 남고 빈 줄을 넣거나 캐시가 차면 끝납니다.

## 벤치마크 요약

llama.cpp(Vulkan)와 vAi를 **prompt 512 토큰** 기준으로 비교한 결과입니다. GPU와 prompt 길이별 전체 결과, 측정 환경과 조건, 재현 방법은 [`benchmarks/llama3/README.md`](../../../benchmarks/llama3/README.md) 참조.

**Llama 3.2-1B**

| | | 890M (Strix Point) | 8060S (Strix Halo) | UHD Graphics (TGL GT1) | RTX 3060 Laptop | RTX 5070 | RTX 4090 | RTX 5090 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vAi | prefill (t/s) | 1,653 | 4,932 | 123.1 | 8,761 | 25,313 | 51,332 | 56,603 |
|  | decode (t/s) | 16.46 | 81.66 | 9.63 | 115.3 | 213.2 | 311.0 | 455.4 |
| llama.cpp | prefill (t/s) | 1,290 | 2,975 | 122.3 | 10,020 | 29,671 | 50,166 | 66,053 |
|  | decode (t/s) | 16.09 | 66.83 | 9.39 | 110.5 | 221.5 | 290.6 | 424.0 |

**Llama 3.2-3B**

| | | 8060S (Strix Halo) | RTX 5070 | RTX 4090 | RTX 5090 |
|---|---|---:|---:|---:|---:|
| vAi | prefill (t/s) | 1,726 | 9,091 | 17,219 | 21,013 |
|  | decode (t/s) | 33.87 | 87.74 | 128.0 | 194.0 |
| llama.cpp | prefill (t/s) | 1,609 | 12,644 | 22,849 | 27,118 |
|  | decode (t/s) | 32.63 | 88.81 | 124.8 | 185.7 |
