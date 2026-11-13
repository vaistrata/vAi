# 빠른 실행

vAi를 빌드하고 MNIST 분류기 예제를 돌려봅니다.

## 설치

### 요구사항

| | |
|---|---|
| GPU | Vulkan conformant (NVIDIA / AMD / Intel) |
| Vulkan SDK | 1.3이상, glslc 포함 |
| CMake | 3.16 이상 |
| 컴파일러 | C++23 |

### 클론

eva(Vulkan 래퍼)가 서브모듈이므로 함께 받아야 합니다.

```bash
git clone --recursive https://github.com/vaistrata/vAi.git
cd vAi
```

이미 vAi 레포를 받았다면:

```bash
git submodule update --init --recursive
```

### 빌드

```bash
cmake -B build
cmake --build build
```

빌드 옵션:

| 옵션 | 기본 | 내용 |
|---|---|---|
| `VAI_BUILD_MODEL_DEMO` | `ON` | 모델별 데모 (`demo-gpt2`, `demo-yolox` 등) |
| `VAI_BUILD_EXAMPLES` | `OFF` | 모델 밖의 예제 (`demo-mnist`, `chat-llama3`, `chat-gemma3`) |
| `VAI_BUILD_TESTS` | `OFF` | 커널 단위 테스트, A/B |
| `VAI_BUILD_BENCH` | `OFF` | llama.cpp, ncnn 등 비교 프레임워크 벤치 |

## 간단 예제

MNIST 분류기는 모델 정의부터 가중치 로딩, 실행까지 [`examples/mnist/main.cpp`](../../examples/mnist/main.cpp) 한 파일에 들어 있습니다. 아래에서는 이 파일을 위에서부터 순서대로 읽습니다. 체크포인트는 [`data/ckpts/mnist/jroot/`](../../data/ckpts/mnist/jroot/)에, 입력 이미지는 [`data/images/`](../../data/images/)에 있습니다.

### 노드 잇기

반복 단위를 먼저 만들어 둡니다.

```cpp
struct ConvBlock : public vai::NodeGroup {
    vai::node::Convolution conv;
    vai::node::MaxPool2d maxpool;

    ConvBlock(uint32_t inChannels, uint32_t outChannels, uint32_t kernelSize)
    : conv({
        .inChannels = inChannels,
        .outChannels = outChannels,
        .kernelSize = kernelSize,
        .stride = 1,
        .epilogue = vai::Epilogue::Relu
    })
    , maxpool({
        .kernelSize = 2,
        .stride = 2
    })
    {
        conv - maxpool;
        defineSlot("in0", conv.slot("in0"));
        defineSlot("out0", maxpool.slot("out0"));
    }
};
```



노드의 설정은 `Config` 구조체로 넘깁니다. ReLU를 에필로그로 주면 활성 노드를 따로 두지 않고 `conv` 커널 안에서 끝냅니다.



`ConvBlock`의 `in0`과 `out0`은 새로 만든 슬롯이 아니라 `conv`의 `in0`과 `maxpool`의 `out0`에 붙인 다른 이름입니다.
### 블록 반복하기

같은 블록을 여러 번 쌓을 때는 벡터에 담아 반복문으로 잇습니다. 노드그룹도 노드와 똑같이 `-`로 이으면 됩니다.

```cpp
struct ConvSequence : public vai::NodeGroup {
    std::vector<ConvBlock> blocks;

    ConvSequence(std::vector<uint32_t> channels, uint32_t kernelSize)
    {
        const uint32_t nBlocks = channels.size() - 1;

        blocks.reserve(nBlocks);
        for (uint32_t i = 0; i < nBlocks; ++i)
        {
            blocks.emplace_back(channels[i], channels[i + 1], kernelSize);
            if (i > 0)
                blocks[i - 1] - blocks[i];
        }

        defineSlot("in0", blocks[0].slot("in0"));
        defineSlot("out0", blocks[nBlocks - 1].slot("out0"));
    }
};
```

`channels`는 블록 경계의 채널 수라서, `{1, 32, 64}`는 1→32, 32→64, 총 두 블록이 됩니다.

`reserve`와 `emplace_back`은 값이 재할당되는 것을 방지합니다([노드그룹의 컨테이너 사용 시 주의할 점](03-usage/node-group.md#컨테이너-사용-시-주의할-점) 참조).

### 모델

`vai::NeuralNet`을 상속하고, `input()`과 `output()`으로 그래프의 입구와 출구를 만듭니다.

```cpp
struct Net : public vai::NeuralNet {
    ConvSequence convX2;
    vai::node::Flatten flatten;
    vai::node::Linear linear;

    Net(const vai::NeuralNetConfig& netCfg)
    : NeuralNet(netCfg)
    , convX2({1, 32, 64}, 3)
    , flatten()
    , linear({7 * 7 * 64, 10})
    {
        input("in0") - convX2 - flatten - linear - output("out0");
    }
};
```

멤버를 모두 `-`로 이으면 Mnist 모델이 완성됩니다. 28x28 입력이 블록 두 개를 지나며 7x7로 줄고, `flatten`을 거쳐 10개 로짓이 나옵니다.

### 가중치 로딩

가중치 로딩은 `loadWeights<배포명>(model)`를 특수화해 만듭니다. 배포명으로 이 MNIST 모델을 학습시킨 기여자의 이름을 씁니다.

```cpp
template<>
void loadWeights<"jroooot", mnist::Net>(mnist::Net& net)
{
    std::string path = VAI_DATA_DIR "/ckpts/mnist/jroot/model.safetensors";
    auto parser = vai::SafetensorsParser(path.c_str());

    // 가중치는 넷의 wDefaultDtype으로 읽는다.
    const vai::Dtype wDt = net.config().wDefaultDtype;
    auto load = [&](auto& node, const std::string& wKey, const std::string& bKey)
    {
        node["weight"] = vai::Tensor(parser[wKey], wDt);
        node["bias"]   = vai::Tensor(parser[bKey], wDt);
    };

    load(net.convX2.blocks[0].conv, "conv1.weight", "conv1.bias");
    load(net.convX2.blocks[1].conv, "conv2.weight", "conv2.bias");
    load(net.linear, "fc.weight", "fc.bias");
}
```

`VAI_DATA_DIR`은 빌드 시스템이 박아주는 저장소의 `data/` 경로입니다. 체크포인트 키 이름과 디스크 경로를 아는 것은 이 함수뿐이고 모델 정의는 둘 다 모릅니다. 제공 모델들은 이 특수화를 `models/<impl>/<모델>/<배포명>-weights.h`라는 별도 파일로 두는데, 그 구조는 [제공 모델의 가중치 로딩](04-models/README.md#가중치-로딩)을 참조하세요.

### 실행

```cpp
int main()
{
    // 28x28 흑백 이미지를 읽어 [0, 1]로 정규화
    std::vector<float> inputData = vai::utils::ImageProcessor(VAI_DATA_DIR "/images/0.png", 1)
        .normalize(0.0f, 1.0f)
        .chw();

    // 모델 생성. defaultDtype이 파라미터의 기본 정밀도다.
    mnist::Net net({
        .defaultDtype = Dtype::Fp32,
    });

    // 체크포인트에서 가중치 로드 ("jroooot"는 학습시킨 기여자)
    loadWeights<"jroooot">(net);

    Tensor input = Tensor({1, 28, 28}, Dtype::Fp32).setFloats(inputData);
    Tensor output;

    // 모델을 돌려 로짓을 받는다
    net({{"in0", input}}, {{"out0", &output}});

    // 네트워크 출력은 fp32다.
    const float* data = output.getData();

    // 클래스별 로짓 출력
    printf("\n=== Output logits ===\n");
    for (int i = 0; i < 10; ++i)
        printf("data[%d] = %f\n", i, data[i]);

    return 0;
}
```

`model(...)`은 텐서를 이름으로 주고받습니다. 앞에서 `input("in0")`과 `output("out0")`으로 정한 이름이 그 키입니다. 한 번 호출할 때 무슨 일이 일어나는지는 [실행 파이프라인](03-usage/execution-pipeline.md) 참조.

출력은 이렇습니다. 입력이 `0.png`라 0번 클래스의 로짓이 압도적으로 큽니다.

```
=== Output logits ===
data[0] = 23.026661
data[1] = -24.376106
data[2] = -1.872803
data[3] = -15.056368
data[4] = -14.481293
data[5] = -13.030799
data[6] = -0.130337
data[7] = -14.873937
data[8] = 1.507278
data[9] = -2.433280
```
