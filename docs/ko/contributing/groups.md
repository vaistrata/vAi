# 커스텀 그룹 작성

그룹은 함께 사용하는 `Node`와 연결을 하나의 부품으로 묶어서 사용할 수 있게 해줍니다. 그룹 안에서 `Node`로 구성요소를 만들고 연결한 뒤, 외부에서 사용할 입출력을 명시해서 다른 모델이나 그룹이 이 입출력에 연결해 같은 구성을 재사용할 수 있도록 합니다.

그룹은 `vai::NodeGroup`을 상속하도록 합니다. 앞서 커스텀 노드 작성 문서에서 `Node`는 그래프의 계산 단위라고 했습니다. `Node`를 하나로 묶은 구현체이기 때문에 실제로 계산할 때 실행되는 것은 그룹 안의 `Node`들입니다. 그룹은 이 `Node`들을 쌓아 만든 하나의 작은 모델이라고 생각하면 됩니다. 새 GPU 계산이 필요하면 `Node`와 커널을 새로 만들어야하고 이 때는 [커스텀 노드와 커널 작성](nodes.md)을 참조하면 됩니다.

이 문서에서 커스텀 그룹 구현의 이해를 돕기 위해 따라갈 기본 예제는 `Transformer`의 `Classic` 구성이 사용하는 [`FeedForward`](../../../vai/groups/feed-forward.h)입니다. 기여자가 정한 설정을 받아 두 `Linear` `Node`를 만들고, 내부를 연결한 뒤 그룹의 입출력을 공개하는 순서로 구현 내용을 따라갑니다. 기여자가 정하고 싶은 설정에 따라 일부 `Node`를 추가하거나 생략하는 방법은 [`MultiHeadAttention`](../../../vai/groups/multi-head-attention.h)의 정규화 `Node`로 설명합니다. 같은 블록을 여러 개 생성하고 차례로 연결하는 방법은 [`Transformer`](../../../vai/groups/transformer.h)의 층 구성으로 설명합니다.

## 1. 그룹으로 묶을 범위와 입출력 정하기

먼저 어떤 `Node`를 하나의 그룹으로 묶을지 정합니다. 그런 다음 그룹에 데이터를 넣을 입력과 결과를 받을 출력을 정합니다. 각 입력과 출력에는 이름, 형상, dtype을 명시합니다.

예를 들어 `FeedForward`는 두 `Linear` `Node`와 활성화 계산을 묶어서 사용합니다. 기본 계산은 `fc2(activation(fc1(in0)))`입니다. 이 절에서는 먼저 `acceptResidual = false`인 간단한 구성을 설명하고 이후 추가 입력은 5절에서 다루겠습니다.

| 항목 | `FeedForward` 예시 |
|---|---|
| 입력 | `in0 [..., inDim]` |
| 중간 결과 | `[..., hiddenDim]` |
| 출력 | `out0 [..., outDim]` |
| dtype | 예제의 `Linear` 출력은 엔진의 기본 출력 dtype 사용 |
| 계산 설정 | 입력·중간·출력 차원, 활성화 함수, 편향 사용 여부 |

`...`는 앞쪽의 축을 유지한다는 뜻입니다. 예를 들어 입력이 `[T, inDim]`이면 출력은 `[T, outDim]`입니다. dtype과 차원 크기에 제한이 있으면 내부 `Node`가 지원하는 조건에 맞춰 명시합니다.

그룹은 `Node`뿐 아니라 다른 그룹도 구성요소로 사용할 수 있습니다. 예를 들어 `TransformerLayer`는 `attention`과 `FeedForward`를 각각 하위 그룹으로 사용합니다. 각 하위 그룹은 내부 `Node`를 연결하고, 부모 그룹은 이 하위 그룹들을 다른 구성요소와 연결합니다.

## 2. 설정과 구성 차이 구분하기

### 2.1 생성자의 설정값

그룹의 생성자는 필요한 값을 `Config`로 받도록 작성합니다. 각 값의 의미와 기본값을 명시하고, 해당 값을 사용하는 `Node`에 전달합니다.

다음은 `FeedForward` 안에 정의된 `Config`입니다.

```cpp
struct Config {
    uint32_t inDim;
    uint32_t hiddenDim;
    uint32_t outDim;
    vai::Epilogue activation = vai::Epilogue::None;
    bool hasBias = false;
    bool acceptResidual = false;
};
```

`inDim`, `hiddenDim`, `outDim`은 생성할 `Linear` `Node`의 크기를 정합니다.
`activation`은 첫 `Linear`가 적용할 활성화 함수입니다. 기본값 `None`은 활성화 함수를 적용하지 않습니다.
그룹이 값을 중복해서 정하지 않도록, 공통 값은 한 번 받아 필요한 구성요소에 전달합니다.

구성의 차이는 다음 기준으로 표현합니다.

| 달라지는 내용 | 작성 방법 | `Transformer` 계열 예시 |
|---|---|---|
| 차원이나 반복 횟수 | `Config`의 값으로 받습니다. | `hiddenDim`, `numLayers` |
| 특정 `Node`의 사용 여부 | `std::optional`로 두고 필요한 경우에만 생성합니다. | 어텐션의 선택적 정규화 |
| 멤버 종류나 연결 구조의 큰 차이 | 템플릿 등으로 타입을 나누는 방법을 고려합니다. | `TransformerType`, `QkvLayout` |

깊이나 차원은 생성자가 받는 값입니다. `Node`의 사용 여부는 생성할 때 정하고, 필요한 멤버만 만듭니다. 사용할 부품의 종류나 연결 방식이 달라지면 컴파일할 때 그룹 타입을 선택해서 조정하면 됩니다.

### 2.2 노드·하위 그룹의 타입 선택

같은 연결 구조에서 `Node`나 하위 그룹의 타입만 바꿀 수 있습니다. [`TransformerLayer`](../../../vai/groups/transformer-layer.h)의 `Classic`과 `Default`는 연결 구조를 공유하지만, 정규화 `Node`와 `FeedForward` 그룹의 타입이 다릅니다.

다음은 `vai::group` 네임스페이스에 있는 타입 선택 코드입니다.

```cpp
template<TransformerType layout> struct BranchFfn;
template<> struct BranchFfn<TransformerType::Classic> { using Type = FeedForward; };
template<> struct BranchFfn<TransformerType::Default> { using Type = GatedFeedForward; };
```

첫 줄은 `BranchFfn` 템플릿을 선언합니다. 그리고 뒤의 두 줄은 `Classic`과 `Default`에 사용할 정의를 지정합니다. 이처럼 특정 템플릿 인자에 대한 정의를 따로 작성하는 것을 **템플릿 특수화**라고 합니다.

`Classic`의 `Type`은 `FeedForward`이고, `Default`의 `Type`은 `GatedFeedForward`입니다.
정규화 `Node`를 고르는 `BranchNorm`도 같은 방식으로 `Classic`에는 `LayerNorm`, `Default`에는 `RmsNorm`을 지정합니다. `TransformerLayer`는 선택된 타입으로 멤버를 선언하므로, 구성마다 생성자와 연결 코드를 따로 작성할 필요가 없습니다.

모델이나 부모 그룹에서는 다음과 같이 사용할 레이어 구성을 지정합니다.

```cpp
using ClassicLayer = vai::group::TransformerLayer<
    vai::TransformerType::Classic, vai::QkvLayout::Separate>;
```

`Node`나 하위 그룹의 타입은 템플릿 인자로 선택할 수 있습니다. 이렇게 선택한 타입은 컴파일할 때 정해집니다. 차원이나 반복 횟수처럼 값만 달라지는 설정은 그룹을 생성할 때 `Config`로 전달합니다.

### 2.3 연결 구조별 그룹 특수화

필요한 `Node`의 수나 연결 경로가 다르면 그룹 템플릿을 특수화할 수 있습니다. 각 특수화에 멤버, 생성자, 연결 코드를 따로 작성합니다.

다음은 [`MultiHeadAttention`](../../../vai/groups/multi-head-attention.h)의 두 특수화에서 일부 멤버만 발췌한 코드입니다.

```cpp
template<QkvLayout layout> struct MultiHeadAttention;

template<>
struct MultiHeadAttention<QkvLayout::Separate> : public vai::NodeGroup
{
    using Config = MultiHeadAttentionConfig;

    vai::node::Linear toQ, toK, toV;
    // ... 나머지 멤버와 생성자, 연결
};

template<>
struct MultiHeadAttention<QkvLayout::Fused> : public vai::NodeGroup
{
    using Config = MultiHeadAttentionConfig;

    vai::node::Linear toQKV;
    // ... 나머지 멤버와 생성자, 연결
};
```

모델이나 부모 그룹은 템플릿 인자로 사용할 구현을 선택합니다. 선택한 구현의 생성자가 해당 구성의 `Node`를 만들고 연결합니다.

여러 구현이 같은 의미의 설정을 받으면 설정 구조체를 그룹 밖에 정의하고 함께 사용합니다. 예제의 두 특수화는 `using Config = MultiHeadAttentionConfig;`로 같은 설정 타입을 사용합니다. 따라서 내부 구현이 달라도 모델이나 부모 그룹은 같은 설정 타입으로 생성자에 값을 전달할 수 있습니다. 다른 구현과 공유하지 않는 설정은 `FeedForward::Config`처럼 그룹 안에 둘 수 있습니다.

## 3. 구성요소를 멤버로 생성하기

### 3.1 노드 멤버 생성과 설정 전달

그룹은 `struct`로 정의하고 `vai::NodeGroup`을 상속합니다. 필요한 `Node`와 하위 그룹은 기본적으로 객체 자체를 멤버로 선언합니다. 그룹을 생성하면 이 멤버들도 함께 생성됩니다. 그리고 그룹 생성자의 초기화 목록에서 각 멤버에 설정을 전달합니다.다음은 `FeedForward`의 멤버와 생성자에서 발췌한 코드입니다. `FeedForward`를 `struct`로 선언하고 초기화 목록으로 생성자의 `:` 뒤에 멤버 이름과 해당 멤버를 생성할 설정을 적었습니다. 예제에서는 `fc1`과 `fc2`에 각각 `Linear` 설정을 전달합니다. 이렇게 생성한 `Node`들을 생성자 본문에서 연결합니다.

```cpp
struct FeedForward : public vai::NodeGroup
{
    vai::node::Linear fc1;
    vai::node::Linear fc2;

    // ... 2절의 Config

    FeedForward(Config config)
    : fc1({
        .inDim = config.inDim,
        .outDim = config.hiddenDim,
        .hasBias = config.hasBias,
        .epilogue = config.activation})
    , fc2({
        .inDim = config.hiddenDim,
        .outDim = config.outDim,
        .hasBias = config.hasBias,
        .acceptResidual = config.acceptResidual})
    {
        // ... 4절의 연결과 5절의 슬롯 공개
    }
};
```

첫 `Node`의 출력 차원과 둘째 `Node`의 입력 차원에 같은 `hiddenDim`을 전달하는데 이처럼 연결되는 두 구성요소의 형상이 맞도록 설정합니다. 실제 입력을 받는 `Node`의 `prepare()`에서도 필요한 조건을 검사합니다. `Node` 설정은 초기화 목록에서 읽을 수 있게 작성합니다.

### 3.2 타입별 하위 그룹 생성

2절에서 선택한 하위 그룹의 타입으로 멤버를 선언합니다. 다음은 `TransformerLayer`가 선택한 타입으로 `ffn` 멤버를 선언하는 코드입니다.

```cpp
typename BranchFfn<layout>::Type ffn;
```

`BranchFfn<layout>::Type`은 2절의 선택 규칙으로 정해진 하위 그룹의 타입입니다. `typename`은 이 `Type`을 타입 이름으로 해석하도록 하는 C++ 키워드입니다. `ffn`은 이 타입으로 생성할 멤버의 이름입니다. 위 규칙으로 선언하시면 됩니다.

멤버를 생성할 때는 선택한 타입에 맞는 설정을 전달합니다. 다음은 생성자에서 `ffn`을 생성하는 부분입니다. 다른 멤버의 초기화는 생략했습니다.

```cpp
TransformerLayer(Config config)
: ffn(ffnConfig(config))
{
    // ... 내부 연결과 슬롯 공개
}
```

`ffnConfig(config)`는 부모 그룹의 설정을 받아 하위 그룹에 전달할 설정을 만드는 방식을 보여줍니다. 바깥의 `ffn(...)`은 그 설정으로 `ffn` 멤버를 생성합니다. 타입마다 설정 구조체가 달라도, 이렇게 설정을 만드는 함수를 두면 부모 그룹의 생성 코드를 공유할 수 있습니다.

정리하면, 부모 그룹은 하위 그룹에 전달할 설정값을 정합니다. 그리고 하위 그룹 내부의 `Node` 생성과 연결은 해당 하위 그룹이 담당합니다.

### 3.3 `optional` 멤버의 조건부 생성

설정에 따라 추가하거나 생략할 `Node`와 하위 그룹은 `std::optional` 멤버로 둘 수 있습니다.  그룹 생성자에서 설정을 확인하고, 필요한 경우에만 해당 멤버를 생성합니다. 다음은 [`MultiHeadAttention`](../../../vai/groups/multi-head-attention.h)에서 필요한 경우에만 `Node`를 생성하는 코드입니다. 멤버 선언과 생성자 실제 본문에서 발췌했습니다.

```cpp
std::optional<vai::node::RmsNorm> normQ, normK;

// ... 생성자 본문
if (config.qkNormEps > 0.0f)
{
    normQ.emplace(vai::node::RmsNorm::Config{.eps = config.qkNormEps});
    normK.emplace(vai::node::RmsNorm::Config{.eps = config.qkNormEps});
}
```

`emplace()`는 전달받은 설정으로 `optional` 안에 `Node`를 직접 생성합니다. 조건을 만족하지 않으면 `optional`은 `Node`가 없는 빈 상태로 남습니다. 그래프를 연결할 때 엔진의 `-` 문법은 중간에 있는 빈 `optional`을 건너뛰고 앞뒤 `Node`나 그룹을 직접 연결합니다. 연결 예시는 4절에서 설명합니다.

슬롯은 자신이 속한 `Node`를 참조하고, 연결된 슬롯의 주소를 보관합니다. 따라서 `Node`와 그룹은 보관할 위치에 직접 생성하고, 사용하는 동안 이동하거나 복사하지 않습니다. 연결한 `optional`도 사용하는 동안 비우거나 안의 `Node`를 다시 생성하지 않습니다.

### 3.4 반복 그룹 생성과 주소 유지

반복되는 블록은 벡터에 담을 수 있습니다. `Transformer`는 `std::vector<Layer> layers`를 멤버로 가집니다. 여기서 `Layer`는 선택한 `TransformerLayer` 타입의 별칭입니다. 다음은 `Transformer` 생성자에서 블록 생성 코드입니다. 층별 설정을 적용하는 부분은 생략했습니다.

```cpp
VAI_ASSERT(config.numLayers >= 1);

layers.reserve(config.numLayers);
for (uint32_t i = 0; i < config.numLayers; ++i)
{
    // ... 이 층에 적용할 설정 계산
    layers.emplace_back(config.layerCfg);
}
```

`reserve()`로 필요한 전체 공간을 먼저 확보하고, `emplace_back()`으로 그 위치에 그룹을 만듭니다. 이후 원소가 이동하거나 벡터가 재할당되는 작업은 피합니다. 주소를 유지하는 것은 엔진 동작에 필요한 조건입니다. 더 참고할 다른 소유 방식은 [컨테이너 사용 시 주의할 점](../03-usage/node-group.md#4-컨테이너-사용-시-주의할-점)을 확인하시면 됩니다.

반복 수와 자식별 설정은 부모 그룹이 정합니다. 예를 들어 마지막 층에만 적용할 옵션은 부모가 해당 층을 생성하기 전에 설정하면 됩니다.

## 4. 내부 경로 연결하기

3절에서 생성한 `Node`와 하위 그룹을 그룹 생성자 본문에서 연결합니다. 각 연결은 한 `Node`나 그룹의 출력을 다음 `Node`나 그룹의 입력으로 전달할 경로를 정합니다.

### 4.1 기본 입출력 연결

`Node`와 하위 그룹 모두 `-` 문법으로 연결합니다. 슬롯 이름을 생략하면 왼쪽의 `out0`과 오른쪽의 `in0`을 연결합니다. 다음은 `FeedForward`의 내부 연결입니다.

```cpp
fc1 - fc2;
```

위 코드처럼 `fc1`의 `out0`을 `fc2`의 `in0`에 간단하게 연결할 수 있습니다. 그룹을 연결할 때는 그룹이 외부에 공개한 슬롯을 사용합니다. 슬롯 공개 방법은 5절에서 설명합니다. 하위 그룹의 타입이 달라도 같은 이름과 역할의 슬롯을 제공하면 연결 코드를 공유할 수 있습니다.

### 4.2 반복 그룹의 순차 연결

벡터에 담은 반복 그룹은 반복문을 사용해서 앞 원소의 출력을 다음 원소의 입력에 연결할 수 있습니다. 다음은 [`Transformer`](../../../vai/groups/transformer.h)에서 생성한 그룹들을 차례로 연결하는 부분입니다. 3절의 `layers` 벡터에 모든 그룹을 생성한 뒤 이 반복문으로 연결합니다.

```cpp
for (size_t i = 1; i < layers.size(); ++i)
    layers[i - 1] - layers[i];
```

각 연결은 `layers[i - 1]`의 `out0`과 `layers[i]`의 `in0`을 잇습니다.

### 4.3 이름으로 슬롯 선택

특정 슬롯에 연결하려면 `/`로 이름을 지정합니다. 이름을 `Node`나 그룹 앞에 쓰면 입력 슬롯을, 뒤에 쓰면 출력 슬롯을 선택합니다. 다음 표에서 `a`와 `b`는 `Node` 또는 하위 그룹입니다.

| 연결 코드 | 왼쪽의 출력 슬롯 | 오른쪽의 입력 슬롯 |
|---|---|---|
| `a - b` | `a`의 `out0` | `b`의 `in0` |
| `a - "in1"/b` | `a`의 `out0` | `b`의 `in1` |
| `a/"out1" - b` | `a`의 `out1` | `b`의 `in0` |

슬롯 이름은 해당 `Node`나 그룹이 실제로 제공하는 이름을 사용합니다.

### 4.4 출력 분기와 `optional` 연결

하나의 출력값을 여러 계산에 사용하려면 같은 출력 슬롯을 각 입력 슬롯에 연결합니다. 다음은 [`MultiHeadAttention`](../../../vai/groups/multi-head-attention.h)에서 세 경로가 시작되는 연결만 남긴 코드입니다. 각 경로에서 이어지는 계산은 생략했습니다.

```cpp
input - toQ;
input - toK;
input - toV;
```

`input`은 입력값을 그대로 출력하는 `Identity` `Node`의 멤버 이름입니다. 세 줄 모두 같은 `input`의 `out0`을 사용합니다. 이 출력을 `toQ`, `toK`, `toV`의 `in0`에 각각 연결해 세 경로로 나눕니다. 3절의 `optional` 멤버도 같은 연결식에 넣을 수 있습니다. 다음은 같은 구현에서 `normQ` 앞뒤의 연결만 발췌한 코드입니다.

```cpp
splitQ - normQ - perQ;
```

`normQ`에 `Node`가 있으면 `splitQ`의 출력이 `normQ`를 거쳐 `perQ`로 전달되도록 연결합니다. 비어 있으면 `splitQ`의 `out0`을 `perQ`의 `in0`에 직접 연결합니다. 이 연결은 생성자에서 그래프를 연결할 때 정해지므로, 설정 조합마다 연결식을 따로 작성할 필요가 없습니다.

### 4.5 형상·dtype 조건과 계산 순서

앞 `Node`나 그룹의 출력 형상과 dtype을 뒤 `Node`나 그룹이 지원하는지 확인해야 합니다. `-` 문법 자체는 형상이나 dtype을 변환하지 않습니다. 변환이 필요하면 해당 계산을 수행하는 `Node`를 경로에 넣습니다.

또한, `Node`의 옵션으로 활성화나 잔차 합산을 함께 처리할 때는 계산 순서를 확인해야 합니다. 예를 들어 활성화 함수 `f`를 적용한 뒤 추가 입력 `r`을 더하는 계산은 `f(x) + r`입니다. 먼저 `r`을 더하는 `f(x + r)`로 순서를 바꾸면 결과가 달라질 수 있습니다. 원래 계산 순서가 유지되도록 옵션을 정하거나, 필요한 계산을 별도 `Node`로 연결합니다.

## 5. 외부에서 사용할 슬롯 공개하기

4절에서 내부 경로를 연결했으면, 모델이나 부모 그룹이 연결할 입력과 출력을 공개합니다. 그룹 생성자에서 `defineSlot()`으로 내부 `Node`나 하위 그룹의 슬롯에 외부 이름을 붙입니다.

### 5.1 내부 슬롯 공개

`defineSlot()`의 첫 인자는 그룹 외부에서 사용할 이름입니다. 둘째 인자는 그 이름으로 접근할 기존 내부 슬롯입니다. 이 함수는 새 슬롯이나 `Tensor`를 만들지 않습니다. 다음은 [`FeedForward`](../../../vai/groups/feed-forward.h) 생성자의 슬롯 공개 부분입니다.

```cpp
defineSlot("in0", fc1.slot("in0"));
if (config.acceptResidual)
    defineSlot("residual", fc2.slot("residual"));
defineSlot("out0", fc2.slot("out0"));
```

`fc1.slot("in0")`은 `fc1`의 기존 입력 슬롯을 가져옵니다. `defineSlot("in0", ...)`은 그 슬롯을 그룹의 `in0`이라는 이름으로 공개합니다. 예제에서 공개하는 슬롯은 다음과 같습니다.

| 그룹의 슬롯 이름 | 실제 연결되는 내부 슬롯 | 공개 조건 |
|---|---|---|
| `in0` | `fc1`의 `in0` | 항상 공개 |
| `out0` | `fc2`의 `out0` | 항상 공개 |
| `residual` | `fc2`의 `residual` | `acceptResidual = true` |

하나의 외부 이름에는 하나의 내부 슬롯을 등록합니다. 같은 이름을 다시 등록해도 연결 대상이 추가되지는 않습니다. 입력을 여러 경로로 나누려면 4절처럼 내부에서 하나의 출력을 여러 입력에 연결합니다.

### 5.2 조건부 입력의 공개와 연결

설정에 따라 추가되는 슬롯은 해당 조건에서만 공개합니다. 예제에서는 `acceptResidual = true`로 그룹을 생성하면 `residual` 입력이 추가됩니다. 이 입력으로 받은 값은 `fc2`의 결과에 더합니다. 모델이나 부모 그룹은 공개된 추가 입력에도 값을 전달할 경로를 연결해야 합니다.

다음은 외부 연결을 설명하기 위한 작성 예시입니다. `ffn`은 `acceptResidual = true`로 생성한 그룹이고, `residualSource`는 더할 값을 출력하는 `Node`입니다.

```cpp
residualSource - "residual"/ffn;
```

`acceptResidual = false`이면 이 입력은 공개되지 않습니다. 공개하는 조건과 함께 입력의 형상과 dtype 조건도 명시해야 합니다.  이 조건은 연결되는 내부 `Node`가 요구하는 조건을 따릅니다. 예제의 `residual`은 그룹 출력과 같은 dtype을 사용합니다. 허용하는 형상은 [`Linear` 구현](../../../vai/nodes/linear.h)을 참조합니다.

### 5.3 하위 그룹의 슬롯 공개

하위 그룹의 입력이나 출력을 공개할 때도 `slot()`으로 해당 슬롯을 가져와 `defineSlot()`에 전달합니다. 다음은 3절의 `ffn` 하위 그룹 출력을 부모 그룹의 `out0`으로 공개하는 코드입니다.

```cpp
defineSlot("out0", ffn.slot("out0"));
```

부모 그룹의 `out0`에 연결하면 하위 그룹 `ffn`의 출력에 연결됩니다. 여러 그룹 구현을 같은 방식으로 사용하려면 공개 슬롯의 이름과 역할을 맞춥니다. 연결할 입력과 출력의 형상 및 dtype 조건도 호환되어야 합니다.

## 6. 빌드와 작성 규칙 확인하기

### 6.1 그룹 헤더 포함과 빌드

그룹을 사용하는 C++ 소스 파일이나 모델·부모 그룹의 헤더에서 그룹 헤더를 포함합니다. 예를 들어 `FeedForward`를 사용하려면 다음 헤더를 포함합니다.

```cpp
#include "vai/groups/feed-forward.h"
```

이 헤더를 사용하는 C++ 소스를 빌드하면 필요한 그룹 구현도 함께 컴파일됩니다. 기존 `Node`와 그룹만 조합했다면 그룹 전용 커널을 새로 추가할 필요는 없습니다. 사용한 `Node`의 커널은 해당 `Node`의 기존 빌드 설정을 따릅니다. 실행 타겟을 추가하는 방법은 [커스텀 모델 작성](models.md#51-실행-타겟-등록)를, 빌드 환경과 명령은 [빠른 실행](../02-quickstart.md)을 참조하시면 됩니다.

### 6.2 파일 배치 그리고 코드와 주석 작성 규칙

| 용도 | 파일 위치 | 네임스페이스 |
|---|---|---|
| 여러 모델에서 사용하는 그룹 | `vai/groups/<이름>.h` | `vai::group` |
| 특정 모델에서 사용하는 그룹 | `models/<모델명>/groups/<이름>.h` | 해당 모델의 네임스페이스 |

주소 유지와 올바른 슬롯 연결은 엔진 동작에 필요한 조건입니다. 값 멤버, `Config` 중심 생성자, 다음 코드 형식은 저장소의 작성 관례입니다.

| 항목 | 작성 규칙 |
|---|---|
| 선언 | `struct 그룹명 : public vai::NodeGroup`으로 선언합니다. |
| 이름 | 타입은 `PascalCase`, 함수와 변수는 `camelCase`로 씁니다. |
| 초기화 | `Node` 설정은 필드 이름을 지정해 초기화합니다. 필요한 파생값은 사용하는 멤버보다 먼저 선언합니다. |
| 공통 설정 | 여러 구현이 같은 설정을 받으면 공통 타입을 두고 `using Config = ...;`로 사용합니다. |
| 지원 범위 | 지원하지 않는 템플릿 선택값이 컴파일 단계에서 드러나게 합니다. |
| 코드 형식 | 함수의 여는 중괄호는 다음 줄에 둡니다. [공통 코드 컨벤션](../index.md#3-공통-코드-컨벤션)을 따릅니다. |
| 헤더 | 헤더 가드와 사용하는 `Node`·그룹의 헤더 포함문을 둡니다. 클래스 정의 밖의 일반 함수 정의에는 `inline`을 붙입니다. |
| 설명 주석 | 계산식, 입력과 출력, 조건부 슬롯, 지원 조건을 적습니다. |

**외부 상태 전달**

외부 버퍼나 상태가 필요한 `Node`를 포함하면, 그룹에 이를 내부 `Node`로 전달하는 메서드를 둘 수 있습니다. 해당 메서드는 모델이나 부모 그룹의 코드에서 필요한 시점에 호출합니다. 상태 관리의 호출 순서는 [모델의 실행 메서드](models.md#42-실행-메서드의-상태-관리)를 참조하면 됩니다.

**타입별 설정을 만드는 도우미**

선택한 부품마다 설정 타입이 다르면 `if constexpr`로 해당 설정을 만드는 함수를 둘 수 있습니다. 다음은 필드별 값을 생략한 `TransformerLayer::ffnConfig()`입니다.

```cpp
static auto ffnConfig(const Config& config)
{
    if constexpr (layout == TransformerType::Classic)
        return FeedForward::Config{/* ... */};
    else
        return GatedFeedForward::Config{/* ... */};
}
```

`if constexpr`는 선택값에 해당하는 분기만 컴파일합니다. 따라서 `Classic`은 `FeedForward::Config`, `Default`는 `GatedFeedForward::Config`를 반환할 수 있습니다. 이 함수에는 설정값을 정하는 코드만 두고, 구성요소의 생성과 연결은 그룹에 둡니다.

**지원하는 선택값과 초기화 순서**

`TransformerLayer`의 공통 구현은 `static_assert`로 `Classic`과 `Default`를 받도록 제한합니다. 다른 구성은 해당 특수화가 처리합니다. `MultiHeadAttention`처럼 기본 템플릿은 선언만 두고, 지원할 선택값에 대해서만 그룹을 정의할 수도 있습니다. 템플릿 특수화와 이 검사 방식은 구현을 나누는 작성 방법이며, 엔진이 요구하는 형식은 아닙니다.

여러 멤버가 사용하는 파생값은 사용하는 멤버보다 먼저 선언하고 초기화합니다. C++ 멤버는 초기화 목록의 나열 순서가 아니라 멤버 선언 순서로 초기화됩니다.

**반복 블록의 설정 변경 순서**

층별 설정을 바꾸는 콜백은 적용 순서와 변경값의 유지 범위를 명시합니다. `Transformer`의 `perLayer`는 같은 `config.layerCfg`를 수정하므로 변경값이 다음 층에도 남습니다. 부모가 보장할 조건은 콜백 뒤에 확정하고 자식을 만듭니다. 다음은 `FeedForward`의 설명 주석에서 발췌한 예시입니다.

```cpp
/////////////////////////////////////////////////////////////////////////////////////////
// FeedForward - two-layer MLP, the activation fused into the first projection
// Input: in0 [..., inDim], optional residual [..., outDim]
// Output: out0 [..., outDim] = fc2(activation(fc1(in0)))
//
// ... 선택 설정과 계산 설명
/////////////////////////////////////////////////////////////////////////////////////////
```
