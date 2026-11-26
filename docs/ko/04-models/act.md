# ACT

ACT(Action Chunking with Transformers, Zhao et al., 2023)는 로봇 모방 학습 정책입니다. 카메라마다 ResNet-18로 특징 맵을 뽑고, 로봇 상태와 함께 DETR 형식의 트랜스포머 인코더-디코더에 넣어 앞으로 실행할 행동 여러 개(청크)를 한 번의 forward로 예측합니다. 학습 때는 CVAE로 행동의 잠재 변수를 함께 학습하지만, 추론에서는 잠재 변수를 0으로 둡니다. vAi는 [LeRobot](https://github.com/huggingface/lerobot)이 학습한 ACT 정책을 읽습니다.

**참고자료**: [논문](https://arxiv.org/abs/2304.13705), [체크포인트(HuggingFace)](https://huggingface.co/mair-robosyn/ACT_sim_item_assembly_official), [저장소](https://github.com/tonyzhaozh/act)

## 설정

```cpp
namespace act = model::production::act;

auto modelCfg = act::ModelConfig::get({
    .variant = "mair-robosyn/item-assembly",
    .cameras = { "cam_high", "cam_right_wrist", "cam_left_wrist" },
    .imageSize = { 480, 640 },
    .stateDim = 14,
    .actionDim = 14,
    .actionChunk = 50,
});

act::Net net({
    .wDefaultDtype = Dtype::Fp16,
}, modelCfg);
```

### NeuralNetConfig

ACT는 기본값을 그대로 쓰고, 로더가 각 `LayerNorm`의 계수만 fp32로 읽습니다. 행동 출력(`actionHead`)은 `oDefaultDtype`과 상관없이 fp32로 고정합니다.
세부사항은 [신경망](../03-usage/neural-net.md) 참조.

### ModelConfig

ACT에는 크기별 팩토리가 없습니다. 체크포인트의 `config.json`이 정하는 값을 `CheckpointConfig`로 받아 `ModelConfig::get()`에 넘깁니다.

| 필드 | `config.json` | 데모 값 |
|---|---|---|
| `variant` | (없음) `data/ckpts/act/` 아래 폴더 | `"mair-robosyn/item-assembly"` |
| `cameras` | `observation.images.<name>`의 `<name>`, 순서대로 | `{ "cam_high", "cam_right_wrist", "cam_left_wrist" }` |
| `imageSize` | 카메라 `shape`의 `{H, W}` | `{ 480, 640 }` |
| `stateDim` | `observation.state`의 `shape` | 14 |
| `actionDim` | `action`의 `shape` | 14 |
| `actionChunk` | `chunk_size` | 50 |

카메라 이름 없이 `observation.image` 하나만 쓰는 정책은 `cameras`를 `{}`로 둡니다.

나머지 구조는 LeRobot ACT 기본값으로 고정되어 있습니다.

| 필드 | `config.json` | 값 |
|---|---|---|
| `embedDim` | `dim_model` | 512 |
| `numHeads` | `n_heads` | 8 |
| `ffnHiddenDim` | `dim_feedforward` | 3200 |
| `numEncoderBlocks` | `n_encoder_layers` | 4 |
| `normEps` | (없음) | 1e-5 |

디코더 블록은 1개(`n_decoder_layers`)이고 FFN 활성화는 ReLU입니다. ResNet-18이 이미지를 32배로 줄이므로 카메라 한 대의 특징 맵은 `ceil(H / 32) × ceil(W / 32)`(480 × 640이면 15 × 20)이고, 인코더 토큰 수(`encoderTokens()`)는 잠재·상태 토큰 2개에 모든 카메라의 특징 맵 칸 수를 더한 값입니다.

### RuntimeConfig

ACT에는 `RuntimeConfig`가 없습니다. forward 한 번이 청크 전체를 내고, KV 캐시나 반복 스텝을 쓰지 않습니다.

### 입출력 정규화

LeRobot ACT는 정규화된 입력과 출력으로 학습합니다(`MEAN_STD`). 로더가 체크포인트의 통계를 `Net`의 멤버에 채우고, 호출하는 쪽이 이 값으로 입력을 정규화하고 출력을 되돌립니다.

| 멤버 | 내용 | 쓰는 곳 |
|---|---|---|
| `imageMean`, `imageStd` | 카메라 이름별 채널 3개의 평균과 표준편차 (`cameras`가 `{}`이면 키는 `"image"`) | `[0, 1]`로 바꾼 이미지를 표준화 |
| `stateMean`, `stateStd` | `[stateDim]` | `normalizeState()` |
| `actionMean`, `actionStd` | `[actionDim]` | `denormalizeActions()` |

`predict()`의 출력은 정규화된 단위이므로, `denormalizeActions()`를 거쳐야 관절 단위가 됩니다.

## vAi 그래프

vAi에서 모델은 노드를 `-`로 이어 만든 DAG입니다. 세부사항은 [노드](../03-usage/node.md)와 [노드그룹](../03-usage/node-group.md) 문서를 참조.

*`B`: 카메라 수, `H × W` 입력 해상도, `h × w` 특징 맵 크기, `D` 임베딩 차원(512), `Dh` 헤드 차원(64), `T`·`Tk` 인코더 토큰 수(`encoderTokens()`), `C` 청크 길이, `S` 상태 차원, `A` 행동 차원.*

### 모델 아키텍처

```cpp
input("inState") - latentStateProj - splitLatentState -                             "in0"/catTokens;
input("inImages") - imageEncoder - flatten - transpose - imageProj - mergeCameras - "in1"/catTokens;

catTokens - encoder[0];
for (uint32_t i = 1; i < modelCfg.numEncoderBlocks; ++i)
    encoder[i - 1] - encoder[i];

encoder.back() -         "memory"/decoder;
input("inDecoderQuery") - "query"/decoder;
decoder - decoderNorm - actionHead - output("outActions");
```

- **상태**: `latentStateProj`가 상태 `[1, S]`를 `[1, 2D]`로 투영하고, `splitLatentState`가 `[2, D]`로 나눕니다. 첫 행은 잠재 토큰, 둘째 행은 상태 토큰입니다. 추론에서 잠재 변수는 0이므로, 잠재 토큰은 로더가 넣은 bias 그대로입니다.
- **이미지**: `imageEncoder`(ResNet-18)가 카메라 `B`대를 배치로 처리해 `[B, 3, H, W]`를 `[B, 512, h, w]`로 바꿉니다. `flatten`, `transpose`, `imageProj`가 칸마다 한 토큰인 `[B, h*w, D]`로 바꾸고, `mergeCameras`가 카메라 순서대로 이어 `[B*h*w, D]`로 폅니다.
- **인코더**: `catTokens`가 잠재·상태 토큰 뒤에 이미지 토큰을 이어 `[2 + B*h*w, D]`를 만들고, `EncoderBlock` 4개가 차례로 처리합니다.
- **디코더**: `DecoderBlock`이 인코더 출력(`memory`)에 cross-attention을 하고, `decoderNorm`과 `actionHead`를 거쳐 행동 청크 `[C, A]`를 냅니다. `query`는 로더가 미리 계산해 둔 `decoderQuery` `[8, C, Dh]`이고, `predict()`가 매번 입력으로 넣습니다.

### NodeGroup

ACT가 사용하는 [노드그룹](../03-usage/node-group.md)입니다. `EncoderBlock`과 `DecoderBlock`은 [`models/production/act/groups/`](../../../models/production/act/groups/)에, 나머지는 [공통 그룹](../../../vai/groups/)에 있습니다.
`ResNet18`이 `BasicBlock`를 쌓고, `EncoderBlock`이 `MultiHeadAttention`과 `FeedForward`를, `DecoderBlock`이 `FeedForward`를 담는 구조입니다.

#### ResNet18

torchvision ResNet-18에서 분류 헤드를 뺀 그룹입니다. stem 뒤에 `BasicBlock` 8개(층마다 2개)가 이어집니다.

```cpp
conv1 - maxPool - blocks.front();
for (size_t i = 1; i < blocks.size(); ++i)
    blocks[i - 1] - blocks[i];
// [B, 3, 480, 640] -> [B, 64, 240, 320] -> [B, 64, 120, 160]
//   -> layer1 [B, 64, 120, 160] -> layer2 [B, 128, 60, 80]
//   -> layer3 [B, 256, 30, 40]  -> layer4 [B, 512, 15, 20]
```

`BasicBlock`은 3x3 conv 둘을 잇고, 두 번째 conv가 입력(또는 1x1 `shortcut`)을 `residual`로 받아 더한 뒤 ReLU를 겁니다.

```cpp
input - conv1 -               conv2;
input - shortcut - "residual"/conv2;
```

BatchNorm은 노드로 두지 않습니다. 로더가 BN 통계를 conv의 `scale`과 `bias`에 합쳐 넣습니다.

#### EncoderBlock

DETR의 post-norm 블록입니다. 서브층을 지난 뒤 잔차를 더하고 정규화합니다.

```cpp
input -            mhAttn - norm1 -            ffn - norm2;
input - "residual"/mhAttn;  norm1 - "residual"/ffn;
```

별도 `Add`를 두지 않고 `mhAttn`과 `ffn`의 마지막 투영이 `acceptResidual`로 잔차를 받아 더한 뒤, `norm1`과 `norm2`가 정규화합니다.

#### MultiHeadAttention

`EncoderBlock`의 `mhAttn`은 `MultiHeadAttention<Fused>`입니다. Q, K, V를 가중치가 `[3D, D]`인 투영 하나로 계산한 뒤 헤드 축에서 세 조각으로 잘라냅니다.

```cpp
projQKV - splitHeads - perQKV - sliceQ - normQ - ropeQ - "Q"/attn;
perQKV - sliceK - normK - ropeK - "K"/attn;
perQKV - sliceV                 - "V"/attn - mergeOut - projOut;
```

ACT는 QK 정규화와 RoPE를 쓰지 않으므로 `normQ`, `normK`, `ropeQ`, `ropeK`는 생기지 않습니다.

DETR는 Q와 K에만 위치를 더해 `x + pos`를 읽고, V는 `x`를 읽습니다. `(x + pos)Wᵀ + b = xWᵀ + (b + pos·Wᵀ)`이므로, 로더가 `projQKV`의 bias를 `[T, 3D]`의 토큰별 행으로 만들어 Q와 K 부분에 `pos·Wᵀ`를 미리 더해 둡니다. 그래서 투영 하나가 `x`만 읽고 세 값을 모두 만듭니다. KV 캐시와 마스크가 없어 모든 토큰이 서로를 봅니다.

#### FeedForward

`EncoderBlock`과 `DecoderBlock`의 FFN입니다. `Linear` 둘을 연결하고, ReLU는 앞쪽 `Linear`의 에필로그로 붙어 GEMM 안에서 처리됩니다.

```cpp
fc1 - fc2;   // 512 -> 3200 -> 512
```

#### DecoderBlock

ACT 디코더 블록 하나에서 cross-attention과 FFN 부분만 그래프로 둡니다.

```cpp
query                                      - "Q"/attn;
memory - projKV - splitKV - perKV - sliceK - "K"/attn;
perKV - sliceV - "V"/attn - mergeOut - projOut - norm1 -            ffn - norm2;
                                                 norm1 - "residual"/ffn;
// projKV [Tk, 2D] -> splitKV [Tk, 16, 64] -> perKV [16, Tk, 64] -> sliceK/V [8, Tk, 64]
// attn [C, 8, 64] -> mergeOut [C, D]
```

- `projKV`는 K와 V를 한 번에 투영합니다. 인코더와 같은 방식으로 K 절반의 bias에 위치를 미리 더해 둡니다.
- 디코더의 입력은 0이므로, 그 앞의 self-attention과 정규화는 가중치만으로 정해집니다. 로더가 이 결과 `x1`을 미리 계산해, `x1`에 쿼리 위치를 더해 투영한 값을 `query`로 넣고, `x1`(cross-attention의 잔차)은 `projOut`의 bias에 더해 둡니다.

### 사용 노드

[`Convolution`](../06-nodes.md#convolution) ·
[`Pool2d`](../06-nodes.md#pool2d) ·
[`Reshape`](../06-nodes.md#reshape) ·
[`Permute`](../06-nodes.md#permute) ·
[`Linear`](../06-nodes.md#linear) ·
[`Concat`](../06-nodes.md#concat) ·
[`Slice`](../06-nodes.md#slice) ·
[`Attention`](../06-nodes.md#attention) ·
[`LayerNorm`](../06-nodes.md#layernorm) ·
[`Identity`](../06-nodes.md#identity)

## 체크포인트

체크포인트는 `data/ckpts/act/` 아래 `<조직>/<이름>` 폴더(`ModelConfig::variant`)에 둡니다. 로더가 폴더 안의 `*.safetensors`를 모두 읽습니다.

```
data/
└── ckpts/act/
    └── mair-robosyn/item-assembly/
        └── model.safetensors
```

LeRobot ACT 정책의 HuggingFace 저장소에서 받아 별도 변환 없이 그대로 씁니다. 라이선스는 저장소마다 다릅니다.

데모는 `mair-robosyn/ACT_sim_item_assembly_official`을 씁니다. 이 저장소는 라이선스를 밝히지 않았습니다.

```bash
mkdir -p data/ckpts/act/mair-robosyn/item-assembly
curl -L -o data/ckpts/act/mair-robosyn/item-assembly/model.safetensors \
  https://huggingface.co/mair-robosyn/ACT_sim_item_assembly_official/resolve/main/model.safetensors
```

HuggingFace 저장소의 **Files and versions** 탭에서 직접 내려받거나 위 명령으로 받으면 됩니다.

> 정규화 통계를 별도 파일 `policy_preprocessor_step_3_normalizer_processor.safetensors`로 배포하는 정책(새 버전 LeRobot)은 그 파일도 같은 폴더에 둡니다. 예전 버전은 통계가 `model.safetensors` 안에 있습니다.

로더가 지원하는 범위는 다음과 같습니다.

| 지원 | 지원하지 않음 |
|---|---|
| 카메라 수 제한 없음 (모두 같은 해상도) | 해상도가 다른 카메라 |
| 상태 차원, 행동 차원, 청크 길이 제한 없음 | `ModelConfig`와 다른 구조 (검사하지 않음) |
| `MEAN_STD` 정규화 (`model.safetensors` 또는 step 3 정규화 파일) | `observation.environment_state` |
| | `MIN_MAX`, `IDENTITY` 정규화 |
| | 정규화 앞의 전처리 단계 (예: 상대 행동) |

### 가중치 로딩

체크포인트를 그래프 슬롯에 매핑하는 코드는 [`models/production/act/lerobot-weights.h`](../../../models/production/act/lerobot-weights.h)에 있습니다.
실행하는 쪽은 한 줄이면 됩니다. 경로는 특수화가 조립합니다(`VAI_DATA_DIR "/ckpts/act/" + modelCfg.variant`). 배포명 태그의 원리는 [커스텀 모델 작성의 가중치 로딩 함수 작성](../contributing/models.md#33-가중치-로딩-함수-작성) 참조.

```cpp
loadWeights<"lerobot">(net);
```

## 실행

### 폴더 구조

```
models/production/act/
├── net.h                         모델 정의
├── groups/                       NodeGroup 정의 (EncoderBlock, DecoderBlock)
├── lerobot-weights.h             LeRobot ACT 정책 체크포인트 매핑
├── demo.cpp                      demo-act, 카메라 3대 + 상태로 행동 청크 하나 생성 + 시간 측정
└── CMakeLists.txt

data/
├── ckpts/act/
│   └── <variant>/                ModelConfig::variant (예: mair-robosyn/item-assembly)
│       └── *.safetensors
└── images/                       입력 이미지
```

### 빌드 및 실행

```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo
```

*`demo-act`는 기본으로 빌드됩니다.*

### demo-act

```bash
./build/bin/demo-act
# Windows (Visual Studio, Ninja Multi-Config): build\bin\RelWithDebInfo\demo-act.exe
```

[`demo.cpp`](../../../models/production/act/demo.cpp)는 `mair-robosyn/item-assembly` 정책으로 `data/images/`의 카메라 이미지 3장과 관절 상태(모두 0)를 넣어 실행합니다. forward 시간을 측정해 행동 청크의 앞 5스텝 × 8차원(관절 단위)과 함께 출력합니다.
