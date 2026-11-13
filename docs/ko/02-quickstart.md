# 빠른 실행

vAi를 빌드하고 MNIST 분류기 예제를 돌려봅니다.

MNIST 분류기는 모델 정의부터 가중치 로딩, 실행까지 [`examples/mnist/main.cpp`](../../examples/mnist/main.cpp) 한 파일에 들어 있습니다. 체크포인트는 [`data/ckpts/mnist/jroot/`](../../data/ckpts/mnist/jroot/)에, 입력 이미지는 [`data/images/`](../../data/images/)에 있습니다.

## 1. 설치

### 1.1 요구사항

| | |
|---|---|
| GPU | Vulkan conformant (NVIDIA / AMD / Intel / Qualcomm) |
| Vulkan SDK | 1.3 이상, glslc 포함 |
| CMake | 3.20 이상 |
| 컴파일러 | C++23 (gcc 14 이상) |

### 1.2 클론

eva(Vulkan 래퍼)가 서브모듈이므로 함께 받아야 합니다.

```bash
git clone --recursive https://github.com/vaistrata/vAi.git
cd vAi
git submodule update --init --recursive
```

### 1.3 빌드

```bash
cmake -S . -B build -DVAI_BUILD_EXAMPLES=ON
cmake --build build --target demo-mnist
```

빌드 옵션:

| 옵션 | 기본 | 내용 |
|---|---|---|
| `VAI_BUILD_MODEL_DEMO` | `ON` | 모델별 데모 (`demo-gpt2`, `demo-yolox` 등) |
| `VAI_BUILD_EXAMPLES` | `OFF` | 모델 밖의 예제 (`demo-mnist`, `chat-llama3`, `chat-gemma3`) |
| `VAI_BUILD_BENCH` | `OFF` | llama.cpp, ncnn 등 비교 프레임워크 벤치 |
| `VAI_BUILD_PYTHON` | `OFF` | Python 바인딩 |

### 1.4 실행

```bash
./build/bin/demo-mnist
```

### 1.5 결과

출력 예시는 다음과 같습니다. 입력이 `0.png`이므로 10개 로짓 중 0번 클래스의 값이 가장 큰지 확인합니다.

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
