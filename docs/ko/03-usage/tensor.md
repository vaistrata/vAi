# 텐서(Tensor)

`Tensor`는 노드 사이를 흐르는 데이터입니다.

대부분의 텐서는 [노드의 슬롯](node.md#슬롯)에 담겨 있고 앞 노드의 출력 슬롯에서
다음 노드의 입력 슬롯으로 건너가며 그래프를 흐릅니다. 이 텐서는 대개 노드가
알아서 만들지만 사용자가 직접 만들어 모델에 넣어 주기도 합니다.

노드의 슬롯은 `Node::slot()`으로, 그 슬롯에 담긴 텐서는
`NodeSlot::getValueRef()`로 얻습니다. `Node::operator[]`는 이 두 호출을 하나로
줄여 슬롯 이름만으로 텐서를 바로 접근합니다
([노드의 값 읽기와 슬롯 얻기](node.md#값-읽기와-슬롯-얻기) 참조).

```cpp
linear.slot("out0")                 // NodeSlot& — 슬롯 자체. 그래프 연결에 쓴다
linear.slot("out0").getValueRef()   // Tensor&   — 그 슬롯에 담긴 텐서
linear["out0"]                      // Tensor&   — 위 한 줄과 같다
```

## 데이터와 핸들

내부는 두 클래스입니다.

```cpp
// 실제 바이트의 주인. 텐서들이 공유한다.
class TensorData
{
    std::vector<char> hostData_;    // 호스트 데이터
    const Dtype dtype_;
    vai::PooledBuffer buffer_;      // 디바이스 버퍼
};

class Tensor
{
    std::shared_ptr<TensorData> data_;    // 복사하면 이 참조만 늘어난다
    std::vector<uint32_t> shape_;
};
```

실제 바이트와 `dtype`은 `TensorData`가 가집니다. 바이트는 호스트 메모리에 있을
수도, GPU 버퍼에 있을 수도, 둘 다에 있을 수도 있습니다. 사용자가 다루는
`Tensor`는 그 `TensorData`를 가리키는 핸들이고 형상은 핸들마다 따로 가집니다.
그래서 복사는 같은 데이터를 가리키는 핸들을 하나 더 만들 뿐이고 두 핸들이
같은 데이터를 서로 다른 형상으로 볼 수 있습니다.

```cpp
Tensor a({M, N});               // fp32, M x N
Tensor b = a;                   // 복제가 아니라 같은 TensorData를 가리키는 또 하나의 핸들
Tensor c = a.reshape({M * N});  // c만 1차원. a는 [M, N] 그대로고 데이터는 여전히 하나
```


## 상태

텐서는 형상, 호스트 데이터, 디바이스 버퍼를 각각 가질 수도 안 가질 수도
있습니다. 그 조합이 곧 텐서의 상태입니다.

| 상태 | 뜻 | 확인 |
|---|---|---|
| 빈 핸들 | 형상도 데이터도 없다. 슬롯의 초기 상태 | `validShape() == false` |
| 형상만 | 형상과 `dtype`은 정해졌고 바이트는 아직 없다 | `validShape() == true` |
| 호스트 데이터 | 바이트가 호스트 메모리에 있다 | `hasHostData() == true` |
| 디바이스 버퍼 | 바이트가 GPU 버퍼에 있다. 노드의 `run()`이 읽고 쓰는 대상 | `gpuBufferBound() == true` |

뒤의 둘은 배타적이지 않습니다. 업로드 직후에는 양쪽에 다 있을 수 있고
`NeuralNet`은 가중치를 올린 뒤 호스트 쪽을 비웁니다.

어느 상태를 어떤 순서로 지나는지는 텐서마다 다릅니다. 네 가지 길을
[대표적인 텐서 흐름 및 생명 주기](#대표적인-텐서-흐름-및-생명-주기)에서 다룹니다.

## 생성

텐서를 생성하는 방법은 세 가지입니다.

### 1. 형상으로

```cpp
Tensor x({M, N});                   // fp32가 기본
Tensor y({H, W}, Dtype::Fp16);
```

형상과 `dtype`만 정해지고 데이터는 아직 없는 상태로 시작합니다. **`dtype`은
생성 후에는 바꿀 수 없습니다**. 데이터에 붙는 속성이라 다른 `dtype`이 필요하면
새 텐서를 만들어야 합니다.

데이터는 나중에 채우고 텐서만 미리 만들어 두는 용도입니다. 입력 데이터,
테이블 데이터, 노드의 출력 슬롯 등이 이렇게 생성합니다. 데이터는 후에
[호스트 데이터 업로드](#호스트-데이터-업로드)의 함수로 직접 채우거나 노드 출력처럼 GPU가
채웁니다.

### 2. `safetensors`에서

가중치를 올릴 때 쓰이는 방법입니다. 이때 `dtype`에 신경을 써야 합니다.

```cpp
SafetensorsParser parser(VAI_DATA_DIR "/ckpts/mnist/jroot/model.safetensors");
Tensor w(parser["fc.weight"]);                  // 파일의 dtype 그대로
Tensor w(parser["fc.weight"], Dtype::Fp16);     // fp16으로 변환해 로드
```

형상은 파일에 기록된 것을 그대로 받고 호스트 데이터까지 채워진 상태로
시작합니다. 두 번째 인자를 주면 로드하면서 `dtype`을 변환합니다. BF16은 아직
지원하지 않아서 지정이 없으면 `fp32`로 로드합니다.

vAi에서는 슬롯마다 텐서의 `dtype`을 명시적으로 지정해야 합니다. 자동으로
캐스팅되지 않고, 캐스팅이 필요하면 캐스팅 노드를 명시적으로 삽입해야 합니다.
노드가 원하는 `dtype`이 있으면 거기에 맞는 텐서를 넣어 주는 것은 사용자
몫입니다. 다른 타입이 들어오면 실행할 때 런타임 오류가 나거나 동작이 미정의
상태가 될 수 있습니다.

가중치의 `dtype`은 텐서가 정합니다. 노드는 자기가 가진 가중치 텐서의 `dtype`을
읽습니다. 가중치는 넷의 `wDefaultDtype`으로, norm 계수는 fp32로 올리면 됩니다.

```cpp
linear["weight"] = Tensor(parser["fc.weight"], net.config().wDefaultDtype);  // 넷의 가중치 dtype
linear["weight"] = Tensor(parser["fc.weight"]);   // 파일의 dtype 그대로
```

형상은 `dtype`처럼 지정하는 것이 아니라 파일에 기록된 형상이 그대로 텐서의
형상이 됩니다. 그 형상이 노드가 원하는 형상과 다를 수 있는데 그럴 때는
슬롯에 대입하기 전에 호스트에서 고칩니다
([호스트 전용 텐서 연산](#호스트-전용-텐서-연산) 참조).

이렇게 만든 텐서는 아직 [호스트 데이터](#상태) 상태입니다. 슬롯에 대입한다고
바로 GPU로 가는 것은 아니고 첫 forward의 준비 단계에서 `NeuralNet`이 디바이스
버퍼에 올립니다([대표적인 텐서 흐름 및 생명 주기의 가중치](#1-가중치) 참조).

### 3. 기존 텐서에서

복사와 대입, `Tensor::reshape`는 텐서만 새로 만들고 데이터는 원본과 공유합니다.
하지만 `Tensor::permute`처럼 메모리 레이아웃을 바꾸는 연산은 데이터를 새로 만듭니다
([호스트 전용 텐서 연산](#호스트-전용-텐서-연산) 참조).

```cpp
Tensor b = a;                    // 새 텐서 — 데이터는 a와 공유
Tensor c = a.reshape({M * N});   // 새 텐서 — 데이터는 a와 공유, 형상만 다르다
Tensor t = a.permute(1, 0);      // 새 텐서 — 재배열된 새 데이터를 소유
```

## 호스트 데이터 업로드

형상만 있는 텐서에 데이터를 올리는 함수는 두 개입니다. `Tensor::setFloats()`는
`std::vector<float>`를 받아 각 원소를 텐서의 `dtype`으로 변환해 올립니다.
`Tensor::setData<T>()`는 임의 타입의 로우 포인터를 받아 변환 없이 바이트를
그대로 복사합니다. 단 `setData()`는 다음을 사용자가 보장해야 합니다.

- `sizeof(T)`가 텐서의 원소 크기와 같아야 합니다.
- 포인터가 텐서의 원소 수만큼 유효한 데이터를 가리켜야 합니다.

```cpp
// 이미지 입력 — float 값을 텐서 dtype으로 변환해 저장
std::vector<float> pixels(28 * 28);       // [0, 1]로 정규화된 픽셀
Tensor img({1, 28, 28}, Dtype::Fp16);
img.setFloats(pixels);                    // fp32 -> fp16 변환

// 토큰 입력 — 변환 없는 memcpy. 정수 id의 바이트가 그대로 들어간다
std::vector<int> tokenIds = tokenizer.encode(prompt);
Tensor tokens({(uint32_t)tokenIds.size()});
tokens.setData(tokenIds.data());          // sizeof(int) == fp32 원소 크기
```

올린 데이터는 아직 호스트에만 있습니다([호스트 데이터](#상태) 상태). 사용자의
몫은 여기까지이고 GPU로 옮기는 일은 forward 때 `NeuralNet`이 알아서 합니다.

## 호스트 전용 텐서 연산

체크포인트에 담긴 텐서의 형상이나 배열이 노드가 원하는 것과 다를 때가
있습니다. 그럴 때는 GPU로 올라가기 전 호스트 데이터 상태에서 아래 연산으로
고칩니다.

네 연산 모두 원본은 그대로 두고 새 텐서를 돌려줍니다. 다만 결과 텐서가
자기만의 새 데이터를 소유하는지, 원본과 데이터를 공유하는지는 연산마다
다릅니다. 표의 마지막 열이 그 구분입니다.

| 연산 | 하는 일 | 데이터 |
|---|---|---|
| `Tensor::permute` | 축 재배열 (예: 전치) | 새 데이터를 소유 |
| `Tensor::split` | 선두 축을 잘라 여러 텐서로 | 각 조각이 새 데이터를 소유 |
| `Tensor::concat` | 선두 축으로 이어붙이기 | 새 데이터를 소유 |
| `Tensor::reshape` | 같은 데이터를 다른 형상으로 (원소 수 불변) | 원본과 공유 |

```cpp
// HF GPT-2의 c_attn: [in, out]으로 저장돼 있어 전치가 필요하다
projQKV["weight"] = Tensor(parser["attn.c_attn.weight"], dt).permute(1, 0);

// 융합된 QKV를 셋으로 가른다
auto [q, k, v] = Tensor(parser["attn.c_attn.weight"], dt).split(D, 2 * D);

// SwiGLU: gate와 up 두 projection을 한 GEMM으로 돌리려 이어붙인다
gateValue["weight"] = Tensor(parser["gate.weight"], dt).concat(Tensor(parser["up.weight"], dt));
```

`permute`, `split`, `concat`은 호스트 데이터가 있어야 동작합니다. 디바이스 버퍼가
붙은 뒤의 텐서를 고치는 것은 노드의 일입니다(예: `Reshape`,
`Permute`).

## 출력 읽기

forward는 비동기입니다. `model(...)`에 출력으로 전달한 텐서에 `Tensor::downloadToHost()`를 부르면 GPU
완료를 기다렸다가 결과를 호스트로 내려받습니다. 내려받기는 첫 호출 한 번만
일어납니다.

`Tensor::getData<T>()`는 `downloadToHost()`를 부른 뒤 호스트 데이터의 포인터를 `const T*`로
재해석해 돌려줍니다. 값을 변환하지는 않으므로 `T`는 실제 담긴 값의 타입이어야
합니다.

```cpp
Tensor out;
net(inDict, {{"nextTokenId", &out}});
uint32_t next = out.getData<uint32_t>()[0];   // 내부에서 downloadToHost()
```

출력 버퍼는 다음 forward가 제자리에서 덮어쓰므로 **결과는 다음 `model()` 호출
전에 읽어야 합니다**. 그 시점을 넘긴 핸들로 `downloadToHost()`를 부르면 이미 훼손된
데이터라서 `assert`에 걸립니다.

출력을 꼭 호스트로 내릴 필요는 없습니다. 출력 텐서를 다른 `NeuralNet`의
입력으로, 또는 오토리그레시브 생성처럼 같은 `NeuralNet`의 다음 입력으로 그대로
넘길 수 있습니다.
디바이스 버퍼를 이미 가진 입력은 그래프가 그 버퍼를 직접 읽으므로 호스트를
거치지 않는 **zero-copy** 연결이 됩니다. 모델 여러 개를 이어 붙일 때 쓰는
방법입니다.

## 대표적인 텐서 흐름 및 생명 주기

같은 상태를 지나더라도 텐서가 무엇이냐에 따라 거치는 길이 다릅니다.

### 1. 가중치

호스트에서 시작해 디바이스로 올라가고 그 뒤로는 고정입니다.

```
safetensors 로드 (호스트 데이터)
  → 호스트 전용 연산 (전치 등, 필요하면)
  → persistent 슬롯에 대입
  → 첫 forward의 준비 단계에서 NeuralNet이 업로드 (디바이스 버퍼)
```

슬롯이 `persistent`이므로 forward가 끝나도 버퍼가 회수되지 않고 GPU에 머물러 있습니다. 따라서 `NeuralNet`이
살아 있는 한 가중치를 다시 올리지 않아도 됩니다.

### 2. 외부 입력

호스트에서 시작하는 것은 같지만 실행마다 값이 새로 채워집니다.

```
형상으로 생성
  → setData / setFloats (호스트 데이터)
  → model() 호출 (입력 노드의 슬롯에 담겨 업로드)
  → 다음 호출 전에 값을 다시 채움
```

디코드 루프처럼 같은 형상이 반복될 때는 텐서를 새로 만들지 않고 같은 핸들에
`setData`로 값만 바꿔 다시 넘기는 것이 전형입니다. 매번 새 텐서를 만들어
넘기면 **커맨드버퍼 재기록이 발생**해 효율이 떨어질 수 있습니다.

### 3. 노드 출력

사용자가 만들거나 넘기는 텐서가 아닙니다. 호스트를 거치지 않고 GPU
안에서 곧바로 생겼다가 다음 노드로 넘어가 쓰이고 나면 사라집니다.

```
prepare()가 형상만 생성
  → NeuralNet이 버퍼 풀에서 버퍼를 붙임 (디바이스 버퍼)
  → GPU가 채움
  → 다음 노드의 입력 슬롯으로 (핸들만 전달)
  → 다 쓰인 버퍼는 풀로 돌아가 재사용
```

### 4. forward 출력

`model(...)`이 사용자가 지정한 출력에 채우는 텐서입니다. 호스트로 내려받아 읽을 수도
있고 그대로 다른 forward의 입력으로 넘길 수도 있습니다.

```
model()이 반환
  → 읽기: downloadToHost() / getData<T>()로 GPU 완료를 기다려 내려받음 (호스트 데이터)
  → 넘기기: 그대로 다른 forward의 입력으로 (호스트를 거치지 않는 zero-copy)
  (어느 쪽이든 다음 forward가 같은 버퍼를 덮어쓰기 전에)
```
