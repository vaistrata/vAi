# Gemma 3

Gemma 3(Gemma Team, 2025)는 Google DeepMind에서 개발한 디코더 전용 트랜스포머 기반 언어 모델입니다. GQA self-attention과 GeGLU feed-forward 블록을 쌓은 구조이고, 각 서브층의 입력과 출력 양쪽에 RMSNorm을 둡니다. sliding window attention을 쓰는 로컬 블록 다섯 개와 전체 컨텍스트를 보는 글로벌 블록 하나가 번갈아 놓입니다. 4B와 12B는 SigLIP 비전 인코더가 붙어 프롬프트에 이미지를 함께 넣을 수 있습니다. 모델은 262,144개(4B, 12B는 262,208개) 어휘의 확률 분포를 계산해 다음 토큰을 예측합니다.

**참고자료**: [논문](https://arxiv.org/abs/2503.19786), [체크포인트(HuggingFace)](https://huggingface.co/google/gemma-3-4b-it), [저장소](https://github.com/google-deepmind/gemma)

## 설정

```cpp
namespace gemma3 = model::production::gemma3;

auto modelCfg = gemma3::ModelConfig::get("4B");

gemma3::Net net({
    .wDefaultDtype = Dtype::Fp16,
}, modelCfg, {
    .maxRuntimeContext = 4096,
});
```

### NeuralNetConfig

Gemma 3는 기본값을 그대로 쓰고, 로더가 각 norm의 파라미터만 fp32로 읽습니다.
세부사항은 [신경망](../03-usage/neural-net.md) 참조.

### ModelConfig

`ModelConfig::get()`에 크기를 문자열로 넘깁니다. 표의 값은 `trf` 멤버(`TransformerConfig`)의 필드라서 `modelCfg.trf.embedDim`처럼 읽습니다.

| `get()` | `embedDim` | `ffnHiddenDim` | `numQHeads` | `numKVHeads` | `nBlocks` | `slidingWindow` |
|---|---|---|---|---|---|---|
| `"270M"` | 640 | 2048 | 4 | 1 | 18 | 512 |
| `"1B"` | 1152 | 6912 | 4 | 1 | 26 | 512 |
| `"4B"` | 2560 | 10240 | 8 | 4 | 34 | 1024 |
| `"12B"` | 3840 | 15360 | 16 | 8 | 48 | 1024 |

`headDim` 256은 모든 크기가 같습니다. 270M과 1B는 텍스트 전용이고 `maxTrainedContext`는 32,768입니다. 4B와 12B는 `vision`이 켜져 이미지를 받고, `maxTrainedContext` 131,072를 쓰며, 글로벌 블록의 RoPE를 `globalRopeFactor` 8로 스케일링합니다.

### RuntimeConfig

KV 캐시의 dtype과 크기, 프롬프트를 나눠 넣는 단위를 정합니다. `maxRuntimeContext`를 생략하면 `maxTrainedContext` 값을 그대로 쓰고, 값을 지정하면 그만큼만 할당해 메모리를 아낍니다. 데모와 채팅 예제는 4,096으로 낮춰 잡습니다.

| 필드 | 기본값 |
|---|---|
| `kvCacheDtype` | `Dtype::Fp16` |
| `maxRuntimeContext` | `uint32_t(-1)`: `maxTrainedContext`까지 |
| `prefillChunk` | `512` (`0`이면 `slidingWindow`) |

프롬프트는 `prefillChunk` 토큰씩 조각으로 나눠 넣습니다. 조각은 이미지를 가르지 않으므로, `vision`이 켜진 크기에서는 `prefillChunk`가 이미지 한 장의 토큰 수인 256 이상이어야 합니다.

## vAi 그래프

vAi에서 모델은 노드를 `-`로 이어 만든 DAG입니다. 세부사항은 [노드](../03-usage/node.md)와 [노드그룹](../03-usage/node-group.md) 문서를 참조.

*`D`: 임베딩 차원, `H` 쿼리 헤드 수, `Hkv` KV 헤드 수, `B` 이미지 수.*

### 모델 아키텍처

```cpp
input("inTokenIds") - embedding -                           "in0"/imageCopy;
input("inImages") - visionEncoder - projector - imageRows - "in1"/imageCopy;
imageCopy - decoder - lmHead - topkSampling - output("nextTokenId");
```

`visionEncoder`와 `projector`는 4B와 12B에서만 생기는 `optional` 멤버입니다. `imageRows`([`Reshape`](../06-nodes.md#reshape))가 이미지 토큰을 `[B, 256, D]`에서 `[B * 256, D]`로 펴고, `imageCopy`([`IndexCopy`](../06-nodes.md#indexcopy))가 이를 프롬프트에서 `<image_soft_token>` 자리표시자가 있는 행 위에 덮어씁니다. 이미지가 없는 스텝에서는 임베딩을 그대로 넘깁니다. `embedding`은 출력에 `sqrt(D)`를 곱하고, `lmHead`는 임베딩 가중치를 함께 씁니다.

`decoder`는 Gemma 3 전용 [`Transformer`](../../../models/production/gemma3/groups/transformer.h) 그룹입니다. 블록을 차례로 잇고 마지막에 `finalNorm`을 둡니다.

```cpp
for (size_t i = 1; i < blocks.size(); ++i)
    blocks[i - 1] - blocks[i];
blocks.back() - finalNorm;
```

여섯 번째 블록마다 글로벌이고, 나머지는 로컬입니다.

- **로컬**: 최근 `slidingWindow`개 토큰만 보고 RoPE base 10,000을 씁니다. KV 캐시는 최근 위치만 남기는 링이고, 크기는 `slidingWindow`와 `prefillChunk`로 정해지며 `maxRuntimeContext`에 따라 커지지 않습니다.
- **글로벌**: 앞의 전체 토큰을 보고 RoPE base 1,000,000을 씁니다. KV 캐시는 `maxRuntimeContext` 위치를 담습니다.

### NodeGroup

Gemma 3가 사용하는 [노드그룹](../03-usage/node-group.md)입니다. `Transformer`, `TransformerBlock`, `VisionEncoder`는 [`models/production/gemma3/groups/`](../../../models/production/gemma3/groups/)에, 나머지는 [공통 그룹](../../../vai/groups/)에 있습니다.
`Transformer`가 `TransformerBlock`를 쌓고, `TransformerBlock`가 `MultiHeadAttention`과 `GatedFeedForward`를 담는 중첩 구조입니다.

#### TransformerBlock

```cpp
input - norm1 - mhAttn - norm2 - lastRow - gate/"then" - norm3 - ffn - norm4;
input -       "residual"/norm2 - lastRow - gate/"then" -    "residual"/norm4;
```

각 서브층은 입력(`norm1`, `norm3`)과 출력(`norm2`, `norm4`) 양쪽에서 정규화합니다. 출력 쪽 norm이 `acceptResidual`로 잔차를 받아, 별도 `Add` 없이 커널 안에서 정규화한 뒤에 더합니다.
`lastRow`와 `gate`는 마지막 블록에만 있습니다. `lastRow`는 다음 토큰을 정하는 마지막 행만 FFN으로 보내고, `gate`는 토큰을 내지 않는 조각에서 닫혀서 `norm3` 이후의 계산을 건너뜁니다.

#### MultiHeadAttention

GQA를 쓰기 때문에 K와 V는 쿼리보다 적은 수의 헤드로 투영됩니다(4B는 `H == 8`, `Hkv == 4`). RoPE는 Q와 K에만 적용되고 V는 그대로 통과합니다.

```cpp
input - projQ - splitQ - normQ - perQ - ropeQ - "Q"/attn;
input - projK - splitK - normK - perK - ropeK - "K"/attn;
input - projV - splitV         - perV         - "V"/attn - mergeOut - projOut;
```

Gemma 3에서는 `normQ`와 `normK`가 생겨 RoPE 전에 Q와 K를 헤드별로 RMS 정규화합니다. `projOut`은 잔차를 받지 않고, `norm2`가 더합니다.

모델은 이번 입력의 첫 토큰 위치를 `Rope::setStartPos()`로 연결합니다. RoPE는 이 시작 위치에 입력 안의 토큰 인덱스를 더해 회전 위치를 구합니다.

#### GatedFeedForward

`gate`와 `value`가 같은 입력을 읽고 가중치만 다르므로 `Linear` 하나로 2배 폭까지 한 번에 계산합니다. `GatedMul`가 앞 절반에 GELU(tanh 근사)를 걸어 뒤 절반과 곱합니다(GeGLU).

```cpp
gateValue - gatedMul - out;   // 2560 -> 20480 -> 10240 -> 2560
```

`out`은 잔차를 받지 않고, `norm4`가 더합니다.

#### VisionEncoder

4B와 12B에서만 생깁니다. SigLIP So400m/14 인코더가 896 × 896 이미지를 64 × 64 패치로 나누고, 4×4 평균 풀링으로 이미지당 256개 토큰으로 줄입니다.

```cpp
patchEmbed - encoder - toGrid - pool - toTokens - imageNorm;
// [B, 3, 896, 896] -> [B, 4096, 1152] -> [B, 4096, 1152] -> [B, 64, 64, 1152]
//                  -> [B, 16, 16, 1152] -> [B, 256, 1152] -> [B, 256, 1152]
```

그다음 신경망의 `projector`가 이미지 토큰을 `D`로 투영합니다.

### 사용 노드

[`TokenEmbedding`](../06-nodes.md#tokenembedding) ·
[`RmsNorm`](../06-nodes.md#rmsnorm) ·
[`LayerNorm`](../06-nodes.md#layernorm) ·
[`Linear`](../06-nodes.md#linear) ·
[`Reshape`](../06-nodes.md#reshape) ·
[`Permute`](../06-nodes.md#permute) ·
[`Rope`](../06-nodes.md#rope) ·
[`Attention`](../06-nodes.md#attention) ·
[`GatedMul`](../06-nodes.md#gatedmul) ·
[`Identity`](../06-nodes.md#identity) ·
[`Slice`](../06-nodes.md#slice) ·
[`Gate`](../06-nodes.md#gate) ·
[`IndexCopy`](../06-nodes.md#indexcopy) ·
[`Pool2d`](../06-nodes.md#pool2d) ·
[`TopKSampling`](../06-nodes.md#topksampling)

## 체크포인트

체크포인트는 `data/ckpts/gemma3/google/` 아래 사이즈별 폴더(`ModelConfig::variant`)에, 토크나이저는 `data/tokenizers/gemma3/`에 둡니다.
`tokenizer.json`은 모든 크기가 함께 씁니다. 로더가 폴더 안의 `*.safetensors`를 모두 읽으므로, 4B처럼 여러 파일로 나뉜 체크포인트도 그대로 씁니다.

```
data/
├── ckpts/gemma3/google/
│   ├── 1B/
│   │   └── model.safetensors
│   ├── 4B/
│   │   ├── model-00001-of-00002.safetensors
│   │   └── model-00002-of-00002.safetensors
│   └── ...
└── tokenizers/gemma3/
    └── tokenizer.json
```

크기별 Instruct 저장소에서 받아 별도 변환 없이 그대로 씁니다. 가중치는 [Gemma Terms of Use](https://ai.google.dev/gemma/terms)를 따릅니다.

데모와 채팅 예제는 4B를 씁니다. `google/gemma-3-*-it` 저장소는 게이트가 걸려 있어, 저장소 페이지에서 라이선스에 동의한 뒤 HuggingFace 액세스 토큰(`HF_TOKEN`)을 넘겨야 합니다.

```bash
mkdir -p data/ckpts/gemma3/google/4B data/tokenizers/gemma3
curl -L -H "Authorization: Bearer $HF_TOKEN" -o data/tokenizers/gemma3/tokenizer.json \
  https://huggingface.co/google/gemma-3-4b-it/resolve/main/tokenizer.json
for f in model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do
  curl -L -H "Authorization: Bearer $HF_TOKEN" -o data/ckpts/gemma3/google/4B/$f \
    https://huggingface.co/google/gemma-3-4b-it/resolve/main/$f
done
```

HuggingFace 저장소의 **Files and versions** 탭에서 직접 내려받거나 위 명령으로 받으면 됩니다.

> 다른 크기도 같은 방법으로 [`google/gemma-3-270m-it`](https://huggingface.co/google/gemma-3-270m-it), [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it), [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it)에서 받습니다. 270M과 1B는 `model.safetensors` 하나, 12B는 샤드 5개로 되어 있습니다. **채팅 토큰이 학습된 Instruct 체크포인트가 필요합니다**.

### 가중치 로딩

체크포인트를 그래프 슬롯에 매핑하는 코드는 [`models/production/gemma3/google-weights.h`](../../../models/production/gemma3/google-weights.h)에 있습니다.
실행하는 쪽은 한 줄이면 됩니다. 경로는 특수화가 조립하므로(`VAI_DATA_DIR "/ckpts/gemma3/google/" + modelCfg.variant`) 사이즈만 정하면 됩니다. 배포명 태그의 원리는 [커스텀 모델 작성의 가중치 로딩 함수 작성](../contributing/models.md#33-가중치-로딩-함수-작성) 참조.

```cpp
loadWeights<"google">(net);
```

## 실행

### 폴더 구조

```
models/production/gemma3/
├── net.h                         모델 정의
├── groups/                       NodeGroup 정의 (Transformer, VisionEncoder)
├── processing.h                  이미지 전처리와 이미지 자리표시자 확장
├── google-weights.h              Google(HuggingFace `google/gemma-3-*-it`) 체크포인트 매핑
├── demo.cpp                      demo-gemma3, 이미지 한 장으로 한 프롬프트 완성 + 시간 측정
└── CMakeLists.txt

examples/chat-gemma3/
├── main.cpp                      chat-gemma3, 이미지를 넣을 수 있는 다중 턴 대화
└── CMakeLists.txt

data/
├── ckpts/gemma3/google/
│   └── <variant>/                ModelConfig::variant (270M, 1B, 4B, 12B)
│       └── *.safetensors
├── tokenizers/gemma3/
│   └── tokenizer.json
└── images/                       입력 이미지
```

### 빌드 및 실행

```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVAI_BUILD_EXAMPLES=ON
cmake --build build --config RelWithDebInfo
```

*`demo-gemma3`는 기본으로 빌드되고, 예제인 `chat-gemma3`는 `-DVAI_BUILD_EXAMPLES=ON`이 필요합니다.*

### demo-gemma3

```bash
./build/bin/demo-gemma3
# Windows (Visual Studio, Ninja Multi-Config): build\bin\RelWithDebInfo\demo-gemma3.exe
```

4B로 [`demo.cpp`](../../../models/production/gemma3/demo.cpp)의 `prompt`와 `data/images/paper_captured0.jpg`를 함께 넣어 실행합니다. `prefill`과 `decode` 시간을 각각 측정해 생성된 문장과 함께 출력합니다. 항상 이미지를 넣으므로 4B나 12B가 필요합니다.

### chat-gemma3

```bash
./build/bin/chat-gemma3
# Windows (Visual Studio, Ninja Multi-Config): build\bin\RelWithDebInfo\chat-gemma3.exe
```

[`main.cpp`](../../../examples/chat-gemma3/main.cpp)의 4B 다중 턴 대화 예제입니다. 메시지에 `<image:파일명>`을 쓰면 `data/images/파일명`을 그 턴에 함께 넣고, 한 턴에 2장까지 넣을 수 있습니다. 대화 이력이 KV 캐시에 남고 빈 줄을 넣거나 캐시가 차면 끝납니다.
