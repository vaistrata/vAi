# 노드그룹(NodeGroup)

`NodeGroup`은 노드 여러 개를 감싸 하나처럼 쓰는 서브그래프입니다. 트랜스포머
블록처럼 반복되는 구조를 한 단위로 묶을 때 씁니다.

먼저 [`vai/groups`](../../../vai/groups/)에 재사용 가능한 그룹이 있는지 확인합니다.

```cpp
struct FeedForward : public vai::NodeGroup {
    vai::node::Linear fc1, fc2;                 // 1

    // ... Config 정의

    FeedForward(Config config)
    : fc1({ .inDim = config.inDim, .outDim = config.hiddenDim, .epilogue = config.activation })
    , fc2({ .inDim = config.hiddenDim, .outDim = config.outDim, .acceptResidual = config.acceptResidual })
    {
        fc1 - fc2;                            // 2

        defineSlot("in0", fc1.slot("in0"));   // 3
        if (config.acceptResidual)
            defineSlot("residual", fc2.slot("residual"));
        defineSlot("out0", fc2.slot("out0"));
    }
};
```

1. 멤버로 노드를 선언하고,
2. 생성자에서 내부를 잇고,
3. 밖에 노출할 슬롯을 정합니다.

`NodeGroup::defineSlot`은 새 슬롯을 만드는 것이 아니라 내부 `Node`드의 슬롯에 외부 이름을
붙이는 것입니다. 밖에서 그룹의 `in0`에 연결하면 실제로는 `fc1`의 `in0` 슬롯에
잇습니다.

`NodeGroup`에는 `prepare()`도 `run()`도 없습니다. 실행되는 것은 안에 든 `Node`들이고,
`NodeGroup`은 여러 `Node`를 하나의 서브그래프로 묶어, 반복되는 구조를 재사용할 수 있게 합니다.

## 1. 연결

`Node`와 똑같이 `-`로 잇습니다. `NodeGroup`과 `Node`를 섞어도 됩니다.

```cpp
norm - ffn - head;          // 노드 - 그룹 - 노드
ffn1 - ffn2;                // 그룹 - 그룹
```

`/`로 슬롯을 지정하는 규칙도 같습니다. 그룹 앞이면 들어가는 슬롯, 뒤면 나가는
슬롯이고, 쓸 수 있는 이름은 `defineSlot`으로 노출한 것뿐입니다. `Node`와 마찬가지로
`/`를 생략하면 `in0`과 `out0`으로 간주합니다.

## 2. 중첩

`NodeGroup` 안에 다른 `NodeGroup`을 둘 수도 있습니다. Llama 3.2의 TransformerLayer는 `MultiHeadAttention`과
`GatedFeedForward`를 멤버로 갖는 `NodeGroup`입니다. 두 멤버 역시 여러 `Node`로 구성된 `NodeGroup`입니다.

```cpp
template<TransformerType layout, QkvLayout qkv>
struct TransformerLayer : public vai::NodeGroup {
    vai::node::Identity input;
    typename BranchNorm<layout>::Type norm1;
    MultiHeadAttention<qkv> mhAttn;
    std::optional<vai::node::Slice> lastRow;
    std::optional<vai::node::Gate> gate;
    typename BranchNorm<layout>::Type norm2;
    typename BranchFfn<layout>::Type ffn;

    TransformerLayer(...)
    {
        input - norm1 -            mhAttn - lastRow - gate/"then" - norm2 -            ffn;
        input         - "residual"/mhAttn - lastRow - gate/"then"         - "residual"/ffn;

        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", ffn.slot("out0"));
    }
};
```

Llama 3에서는 `norm1`·`norm2`가 `RmsNorm`, `ffn`이 `GatedFeedForward`입니다.
첫째 줄이 정규화·Attention·FFN으로 이어지는 주 경로이고, 둘째 줄이 잔차 경로입니다.

## 3. Identity

위 예에서 `input`은 아무 연산도 하지 않는 `Identity`입니다. 입력이 여러
갈래로 갈라질 때 그 분기점을 만들어 줍니다. 잔차 연결처럼 같은 입력을 두 곳으로
보내야 할 때 씁니다.

## 4. 컨테이너 사용 시 주의할 점

`Node`도 `NodeGroup`도 컨테이너에 담을 수 있습니다. 다만 `-`로 이은 배선은 슬롯이
서로의 주소를 갖고 있기 때문에 담긴 원소는 복사·이동하지 않고 주소를 유지해야 합니다.

```cpp
layers.reserve(config.numLayers);
for (uint32_t i = 0; i < config.numLayers; ++i)
{
    // ... 레이어별 설정
    layers.emplace_back(config.layerCfg);
}

for (size_t i = 1; i < layers.size(); ++i)
    layers[i - 1] - layers[i];
```

`std::vector`에 자리가 부족하면 기존 원소를 새 메모리로 옮깁니다.
`push_back(Layer(...))`처럼 임시 노드를 넣는 것도 복사·이동을 일으킵니다.
이때 슬롯의 부모 참조나 연결이 원래 객체를 가리킨 채 남아 실행 시 잘못된
객체에 접근할 수 있습니다. 컴파일은 통과하지만 그래프 실행 시 에러가 발생합니다.

원소를 제자리에 묶어 두는 것은 작성자 몫이고, 방법은 두 가지입니다.

**1. `reserve` + `emplace_back`**: 필요한 개수만큼 자리를 미리 잡아 재할당을
막고 컨테이너 안에서 원소를 직접 만들어 복사를 막습니다. 위 예가 이 방식입니다.

**2. `std::unique_ptr`**: 원소를 힙에 두고 컨테이너는 포인터만 담습니다. 벡터가
포인터를 옮겨도 가리키는 객체는 움직이지 않습니다.

```cpp
std::vector<std::unique_ptr<Layer>> layers;
layers.push_back(std::make_unique<Layer>(layerCfg));
```

vAi는 2번 방식도 허용하지만 1번을 권장합니다.

그룹을 추가할 때는 기존 그룹을 재사용 가능한지 확인하세요.

`std::deque`, `std::list`, `std::map` 등 다른 STL 컨테이너도 마찬가지로 담긴
원소가 만들어진 자리에 머물러야 합니다. 어느 연산에서 원소가 옮겨지는지는
컨테이너마다 다르지만 그것은 표준 C++ 쪽 규칙이라 여기서는 다루지 않습니다.
