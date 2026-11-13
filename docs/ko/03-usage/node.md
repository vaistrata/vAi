# 노드(Node)

![Linear 노드 셋의 출력이 Attention의 Q, K, V 슬롯으로 이어진 그래프](../../assets/node-slots.svg)

`Node`는 연산 하나를 나타내는 단위입니다. 이름 있는 슬롯으로 `Tensor`를 받아 연산하고 슬롯으로 내보냅니다.
한 노드가 여러 GPU 커널을 기록할 수도 있고, `Identity`·`Flatten`처럼 버퍼를 공유하고
계산 명령을 기록하지 않는 노드도 있습니다.

```cpp
vai::node::Linear linear({.inDim = inDim, .outDim = outDim});
vai::node::Attention attn({
    .headDim = 64,
    .numQHeads = 8,
    .numKVHeads = 2,   // GQA: 생략하면 numQHeads와 같아진다
    .kvCacheSize = 1024,
});
```

각 노드는 생성자에서 자기 슬롯을 정의합니다. 위 그림의 두 노드는 아래와 같이 정의합니다:

```cpp
Linear::Linear(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
    addSlot("weight", NodeSlot::persistent);
    addSlot("partial", NodeSlot::temporary);   // reduce 패스를 탈 때만 할당된다
    addSlot("bias", NodeSlot::persistent);
    if (config_.acceptResidual) addSlot("residual", NodeSlot::input);
}
```

```cpp
Attention::Attention(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("Q", NodeSlot::input);
    addSlot("K", NodeSlot::input);
    addSlot("V", NodeSlot::input);
    addSlot("partialO", NodeSlot::temporary);    // split-K 부분합, kCount > 1일 때만 할당된다
    addSlot("partialML", NodeSlot::temporary);

    // ... Hkv 계산
    if (config_.kvCacheSize > 0)
    {
        addSlot("K_cache", NodeSlot::persistent);
        addSlot("V_cache", NodeSlot::persistent);
        // ... 캐시 텐서 생성
    }
}
```

슬롯 구성은 옵션에 따라 달라지기도 합니다. `Linear`의 `residual`은
`acceptResidual`일 때만 생깁니다. `bias` 슬롯은 항상 있고, 가중치가 채워졌을 때만 적용됩니다.
Attention의 캐시 슬롯은 `kvCacheSize > 0`일 때만 생깁니다. 그림에 없는
`partial`은 연산 내부의 버퍼입니다.그래프의 입출력 연결에는 사용하지 않습니다.

## 1. 슬롯

슬롯은 `Tensor`가 드나드는 포트입니다. 네 종류가 있고 종류마다 값을 넣는 주체가
다릅니다.

| 종류 | 값(텐서)의 출처 | 값의 수명 |
|---|---|---|
| `input` | 앞 노드의 출력 또는 외부 입력 | 실행마다 갱신 |
| `output` | 노드 자신 (`prepare()`에서 만들고 `run()`이 채움) | 실행마다 갱신 |
| `persistent` | 사용자(가중치) 또는 노드(캐시, 상수 테이블) | 네트워크 실행 동안 유지 |
| `temporary` | 노드 자신 | `run()` 동안만 유효 |

### 1.1 `input`

연결 연산자 `-`로 이어진 앞 노드의 출력을 받습니다. 그래프의 시작점이라면
`NeuralNet`이 외부 입력으로 연결시킵니다.

### 1.2 `output`

노드가 `prepare()`에서 dtype을 정하고 결과 형상을 계산해 텐서를 만들어 둡니다. `run()`은 그 버퍼를 채울 GPU 명령을 기록하고, 실행이 끝나면 결과가 저장됩니다.

### 1.3 `persistent`

실행 사이에 값이 유지되는 슬롯입니다. 두 갈래가 있습니다.

**1. 가중치는 사용자가 준비합니다**: `Linear`의 `weight`와 `bias` 같은 학습된
파라미터 값이 올라가는 슬롯입니다. 준비하지 않으면 `Node::prepare()`의 형상 검증에 걸립니다.

```cpp
vai::Checkpoint ckpt(VAI_DATA_DIR "/ckpts/mnist/jroot");
const vai::Dtype wDt = net.config().wDefaultDtype;   // 넷의 가중치 dtype
linear["weight"] = vai::Tensor(ckpt["fc.weight"], wDt);
linear["bias"]   = vai::Tensor(ckpt["fc.bias"], wDt);
```

**2. 나머지는 노드가 스스로 준비합니다**: `Attention`의 `K_cache`, `V_cache`,
`Rope`의 `freqs`가 이에 해당합니다.

```cpp
// Attention 생성자: 노드가 캐시를 직접 만든다.
(*this)["K_cache"] = Tensor({Hkv, config_.kvCacheSize, config_.headDim}, config_.kvCacheDtype);
```

가중치처럼 고정된 값이든 KV 캐시처럼 매 스텝 쌓이는 상태든 버퍼가 실행 사이에
살아 있다는 점이 `persistent`의 공통점입니다. `input`, `output` 버퍼는 실행마다
갱신되고 버퍼 풀에서 재사용됩니다.

### 1.4 `temporary`

연산 내부에서만 쓰는 작업 버퍼입니다. `Linear`의 `partial`처럼 특정 실행 경로에서만
필요할 수 있어, 그 경로를 사용하지 않으면 버퍼를 할당하지 않습니다.

## 2. 슬롯 다루기

### 2.1 슬롯 이름

슬롯은 생성자에서 `Node::addSlot`으로 선언합니다. 사용자가 밖에서 새 이름을 붙일 수 없고, 없는 이름으로 접근하면 예외가 발생합니다.

```cpp
linear.slot("weight");   // OK
linear.slot("scale");    // Linear에 없는 슬롯(예외)
```

어떤 슬롯이 있는지는 [제공 노드들](../06-nodes.md)의 노드별 슬롯 표에
있습니다.

### 2.2 값 읽기와 슬롯 접근

`Node::operator[]`와 `Node::slot()`은 이름이 비슷하지만 돌려주는 것이 다릅니다.

```cpp
linear["weight"];       // Tensor&    값을 읽고 저장한다
linear.slot("out0");    // NodeSlot&  연결에 사용한다
```

연산자 `-`는 내부적으로 `slot()`을 부릅니다. 값을 다룰 때는 `[]`, 그래프를 엮을
때는 `slot()`이라고 보면 됩니다.

## 3. 연결

연산자 `-`는 왼쪽 노드의 `output` 슬롯을 오른쪽 노드의 `input` 슬롯에
잇습니다. 어느 슬롯인지는 `/`로 이름을 붙여 정합니다.

```cpp
conv/"out0" - "in0"/flatten/"out0" - "in0"/linear;
```

### 3.1 슬롯 지정

슬롯 지정 연산자 `/`는 노드 왼쪽에 붙으면 `input` 슬롯, 오른쪽에 붙으면 `output` 슬롯입니다.
데이터가 왼쪽에서 오른쪽으로 흐르는 것과 같은 방향입니다.

```
          "in1" / add1 / "out0"
          └┬──┘          └─┬──┘
          input 슬롯    output 슬롯
```

슬롯 지정 연산자 `/`를 생략하면 노드 앞자리에는 `in0`, 뒷자리에는 `out0` 슬롯이
쓰입니다.

| 표기 | `input` 슬롯 | `output` 슬롯 |
|---|---|---|
| `node` | `in0` | `out0` |
| `"X"/node` | **`X`** | `out0` |
| `node/"Y"` | `in0` | **`Y`** |
| `"X"/node/"Y"` | **`X`** | **`Y`** |

그래서 위의 줄은 이렇게 줄여 쓸 수 있습니다.

```cpp
conv - flatten - linear;
```

노드 앞의 `/`에는 `input` 타입 슬롯만, 노드 뒤의 `/`에는 `output` 타입 슬롯만
올 수 있습니다. `persistent`나 `temporary` 슬롯은 그래프 연결에 쓰이지 않습니다.

### 3.2 슬롯이 여러 개인 경우

입력이 여러 개인 노드에서 슬롯 지정이 필요합니다. 예를 들어 `Add`는 `in0`과 `in1`을 받는데, `-`만으로는
둘 다 `in0`으로 가버리니 한쪽을 지정해야 합니다.

```cpp
input - norm1 - mhAttn - "in0"/add1;   // 어텐션 결과는 add1의 in0으로
input                  - "in1"/add1;   // 원래 입력은 add1의 in1로
```

이것이 잔차 연결입니다. `input`의 출력이 두 곳으로 갈라져 나가는데, 한 출력을
여러 입력에 잇는 것은 허용됩니다.

일반적으로 하나의 입력 슬롯에는 하나의 출력 슬롯만 연결합니다. 예외적으로 Gate의 `then`과 `else` 두 경로의 출력 슬롯은 동일한 입력 슬롯에 연결할 수 있습니다.
이 경우 선택된 경로만 커맨드에 기록되므로, 실행 시에도 해당 경로의 결과만 사용합니다.

```cpp
gate/"then" - relu        - "in0"/merge;
gate/"else" - passthrough - "in0"/merge;
```

슬롯 이름이 아예 다른 노드도 있습니다. `Attention`는 `Q`, `K`, `V`를
받으므로 전부 지정해야 합니다.

```cpp
input - projQ - splitQ - perQ - "Q"/attn - mergeOut - projOut;
input - projK - splitK - perK - "K"/attn;
input - projV - splitV - perV - "V"/attn;
```

## 4. 노드의 라이프사이클

노드의 라이프사이클은 두 단계입니다. 직접 호출하지 않고 `NeuralNet`이 호출합니다.

|                      | 언제            | 하는 일                  |
| -------------------- | ------------- | --------------------- |
| `Node::prepare()`    | 입력 형상이 정해졌을 때 | 형상·dtype을 검증하고 출력·작업 텐서와 파이프라인을 준비한다 |
| `Node::run(cmdBuff)` | 그래프를 기록할 때    | GPU 커맨드를 커맨드 버퍼에 기록한다 |

`Node::run()`은 연산을 실행하는 것이 아니라 명령을 적어두는 것입니다. 실제 계산은
나중에 GPU가 그 명령서를 실행할 때 일어납니다. vAi가 매 추론마다 그래프를 다시
순회하지 않는 이유가 여기 있습니다. 한 번 만들고 기록 조건이 같으면 커맨드 버퍼를 재사용합니다.

입력 형상이 바뀌면 `Node::prepare()`를 다시 호출니다. 형상에 따라 커널이 갈리는 노드는
이때 파이프라인을 다시 고릅니다. 더 자세한 내용은 [커스텀 노드와 커널 작성의 prepare()](../contributing/nodes.md#prepare) 참조.
