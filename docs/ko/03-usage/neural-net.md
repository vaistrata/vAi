# 신경망(NeuralNet)

`NeuralNet`은 그래프 전체입니다. 만드는 방식은 [노드그룹](node-group.md)과 거의 같습니다. 노드를 멤버로 선언하고 생성자에서 `-`로 잇습니다.

차이는 그래프의 입출력을 정의하는 방법입니다. `NodeGroup`은 `NodeGroup::defineSlot()`으로 안쪽 슬롯에 이름을 붙여 밖으로 내보내지만, `NeuralNet`은 `NeuralNet::input()`과 `NeuralNet::output()`이라는 경계에 직접 잇습니다.

```cpp
struct Block : public vai::NodeGroup {       // 노드그룹
    defineSlot("in0", fc1.slot("in0"));      // 안쪽 슬롯에 이름을 붙여 내보낸다
};

struct Net : public vai::NeuralNet {       // 신경망
    input("in0") - fc1;                      // 입력 경계에 직접 잇는다
};
```

`input()`과 `output()`은 그래프가 바깥과 만나는 유일한 지점입니다. 여기에 이어진 것만 호스트에서 값을 넣거나 꺼낼 수 있습니다.

## 생성

```cpp
struct Net : public vai::NeuralNet {
    vai::node::TokenEmbedding embedding;
    std::vector<TransformerBlock> layers;
    vai::node::Linear lmHead;

    Net(const vai::NeuralNetConfig& netCfg, ModelConfig modelCfg)
    : NeuralNet(netCfg)
    , embedding({ /* ... */ })
    , lmHead({ modelCfg.embedDim, modelCfg.vocabSize })
    {
        layers.reserve(modelCfg.numLayers);
        for (uint32_t i = 0; i < modelCfg.numLayers; ++i)
            layers.emplace_back(/* ... */);

        input("inTokenIds") - embedding - layers[0];

        for (size_t i = 1; i < layers.size(); ++i)
            layers[i - 1] - layers[i];

        layers.back() - lmHead - output("nextTokenId");
    }
};
```

`std::vector`와 같은 컨테이너를 쓸 때 지켜야 할 것은 [노드그룹의 컨테이너 사용 시 주의할 점](node-group.md#컨테이너-사용-시-주의할-점)과 동일합니다.

## 경계 이름

`input()`과 `output()`은 이름으로 구분하며, 생략하면 각각 `"in0"`과 `"out0"`입니다.

```cpp
input()              // == input("in0")
input("inTokenIds")  // "inTokenIds"라는 입력 경계를 만든다
```

경계는 처음 부를 때 만들어집니다. 미리 선언할 필요가 없으며, 같은 이름을 다시 부르면 같은 경계를 돌려줍니다. 입력이나 출력이 여러 개라면 서로 다른 이름을 주면 됩니다.

여기서 정한 이름이 곧 실행할 때 쓰는 키입니다.

## 실행

`NeuralNet::operator()`에 입력 텐서와 결과를 받을 출력 텐서의 주소를 이름별로 넘깁니다.

```cpp
using TensorDict = std::map<std::string, Tensor>;
using TensorRefDict = std::map<std::string, Tensor*>;

void NeuralNet::operator()(TensorDict inputs, TensorRefDict outputs, bool syncPrev = true);
```

입력과 출력 모두 키는 경계 이름입니다. 넘기는 `TensorDict`의 키는 `input()`으로 만든 입력 경계에, 출력 `TensorRefDict`의 키는 `output()`으로 만든 출력 경계에 대응합니다.

```cpp
namespace gpt2 = model::production::gpt2;

gpt2::Net net({}, gpt2::ModelConfig::_124M());

Tensor tokens = Tensor(/* ... */);

vai::TensorDict in = {{"inTokenIds", tokens}};
Tensor result;
net(in, {{"nextTokenId", &result}});
```

입력 딕셔너리를 호출 자리에서 만들 수도 있습니다.

```cpp
net({{"inTokenIds", tokens}}, {{"nextTokenId", &result}});
```

호출은 **비동기**입니다. `model(...)`이 반환되었다고 GPU 계산이 끝난 것은 아닙니다. 결과 텐서에 `Tensor::downloadToHost()` 또는 `Tensor::getData<T>()`를 호출하는 시점에 필요하면 완료를 기다린 뒤 데이터를 가져옵니다
([텐서의 출력 읽기](tensor.md#출력-읽기) 참조).

호출 한 번의 내부 과정은 [실행 파이프라인](execution-pipeline.md)에서 단계별로 다룹니다.

넘기는 이름은 `input()`으로 만든 경계와 정확히 일치해야 합니다. 입력 딕셔너리를 비우거나 존재하지 않는 이름을 추가하면 오류가 발생합니다. 여러 입력 중 일부를 생략하면 해당 입력에서 출발하는 가지가 실행에서 제외됩니다.

```cpp
net({{"inTokenIds", tokens}}, {{"nextTokenId", &result}});                  // OK
net({}, {{"nextTokenId", &result}});                                        // X: 입력이 빠졌다
net({{"inTokenIds", t}, {"extra", e}}, {{"nextTokenId", &result}});         // X: 없는 경계
```

## `onForwardBegin`

GPU에 값을 전달하는 기본 경로는 `NeuralNet::operator()`의 입력 텐서입니다. 하지만 그래프를 따라 흐르지는 않으면서 실행마다 달라져야 하는 값도 있습니다.

* KV 캐시가 지금까지 처리한 토큰 수: 어텐션이 캐시의 어느 위치부터 이어 쓸지 결정합니다.
* 이번 샘플링에 사용할 난수: `TopKSampling`가 역 CDF로 토큰을 뽑을 때 사용합니다.

토큰이나 로짓처럼 노드 사이를 흐르는 데이터가 아니라, 특정 커널이 실행할 때 참조하는 작은 상태값입니다.

### 버퍼 준비

호스트가 직접 쓸 수 있는 별도의 버퍼에 담고, 필요한 노드가 그 버퍼를 가리키도록 합니다.

```cpp
kvCacheCounter = device().createBuffer({
    .size = sizeof(uint32_t),
    .usage = eva::BUFFER_USAGE::UNIFORM_BUFFER,
    .reqMemProps = eva::MEMORY_PROPERTY::HOST_VISIBLE |
                   eva::MEMORY_PROPERTY::HOST_COHERENT
});

embedding.setKvCacheCounter(kvCacheCounter);

for (auto& layer : layers)
    layer.mhAttn.attn.setKvCacheCounter(kvCacheCounter);
```

`HOST_VISIBLE`이므로 CPU는 `map()`으로 값을 직접 쓸 수 있습니다. 노드에는 이 버퍼를 한 번 연결해 두고 실행마다 버퍼 안의 값만 바꿉니다.

여기에는 동기화가 필요합니다. GPU가 버퍼를 읽는 동안 CPU가 값을 덮어쓰면 같은 실행에서 서로 다른 값을 읽을 수 있습니다. 게다가 `NeuralNet::operator()`는 GPU를 기다리지 않고 반환하므로 호출이 끝났다고 GPU 실행까지 끝난 것은 아닙니다.

### 값 갱신

값을 아무 때나 바꾸면 안 됩니다. 이전 실행이 끝난 뒤, 이번 실행이 시작되기 전에 갱신해야 합니다. 그 자리가 `NeuralNet::onForwardBegin()`입니다.

```cpp
void onForwardBegin(const vai::TensorDict& inputs) override
{
    // 이번 실행 이전까지 처리한 토큰 수
    *(uint32_t*)kvCacheCounter.map() = tokensSoFar_;

    tokensSoFar_ += (uint32_t)inputs.at("inTokenIds").shape()[0];

    topkSampling.setUniform(
        (float)std::rand() / ((float)RAND_MAX + 1.0f));
}
```

실행 순서에서 훅의 위치는 다음과 같습니다.

```text
이전 실행 완료 대기
  → onForwardBegin()        ← 여기서 값 갱신
  → 이번 실행 준비
  → GPU에 커맨드버퍼 제출
```

이 위치 때문에 두 가지가 보장됩니다.

* **훅 이전**: 직전 실행이 이미 끝났으므로 GPU와 충돌하지 않고 버퍼를 수정할 수 있습니다.
* **훅 이후**: 이번 실행의 준비와 제출이 아직 시작되지 않았으므로 여기서 바꾼 값이 이번 실행에 반영됩니다.

### 왜 이렇게 하는가

이런 값을 푸시 상수처럼 GPU 커맨드에 직접 넣을 수도 있습니다. CUDA로 치면 커널 호출 인자와 비슷한 방식입니다. 하지만 커맨드에 기록된 값이 바뀌면 커맨드버퍼도 다시 기록해야 합니다.

반대로 커맨드에는 버퍼를 가리키도록 해 두고 실제 값만 버퍼 안에서 바꾸면, 커맨드버퍼는 그대로 재사용할 수 있습니다. 실행마다 바뀌는 것은 버퍼 안의 몇 바이트뿐입니다.

이것이 `onForwardBegin()`의 핵심 역할입니다. 커맨드버퍼를 다시 기록하지 않고도 GPU에 매번 새로운 값을 전달합니다.

이렇게 실행마다 바뀌는 값을 커맨드에서 분리하면 KV 캐시 카운터나 샘플링 난수가 계속 달라져도 커맨드버퍼를 다시 기록할 필요가 없습니다. vAi의 [고성능](../index.md#3-고성능)을 가능하게 하는 부분입니다.

커맨드버퍼가 실제로 언제 다시 기록되는지는 [실행 파이프라인의 5. 재기록 판정](execution-pipeline.md#5-재기록-판정)에서 다룹니다.

### 실행 밖에서 같은 값을 수정할 때

KV 캐시를 비우는 것처럼 `onForwardBegin()` 밖에서 같은 버퍼를 수정해야 할 때도 있습니다. 이전 실행이 아직 그 값을 읽고 있을 수 있으므로 `NeuralNet::syncPrevForward()`로 먼저 완료를 기다립니다.

```cpp
void resetCache()
{
    syncPrevForward();      // GPU가 아직 카운터를 읽고 있을 수 있다

    tokensSoFar_ = 0;
    *(uint32_t*)kvCacheCounter.map() = 0;
}
```

이후에는 GPU가 해당 버퍼를 더 이상 사용하지 않으므로 안전하게 값을 수정할 수 있습니다.
