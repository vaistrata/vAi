# EVA 객체

EVA는 vAi의 GPU 연산에 필요한 데이터 저장, 연산 준비, 명령 기록 및 제출을 관리하는 C++ API입니다. 런타임은 `eva::Buffer`, `eva::ComputePipeline` 등의 객체를 사용하여 입력과 실행 리소스를 준비하고 GPU에 연산 명령을 제출합니다.

[EVA (Easy Vulkan Api)](https://github.com/vaistrata/eva)는 Vulkan의 고급 기능을 쉽게 사용할 수 있도록 설계된 API입니다.

이 문서는 **EVA 객체의 역할과 사용 방법, 내부 Vulkan 객체의 구조**를 설명합니다. 자세한 Vulkan 객체의 상세 정의, 사용 방법 및 동기화 규칙은 [Vulkan Specification](https://docs.vulkan.org/spec/latest/)을 참조하십시오.

이 문서에서 **Host**는 CPU 측, **Device**는 GPU 측을 뜻합니다. **커널**은 GPU에서 실행할 연산 코드이며, 여기서는 Compute Shader로 작성합니다.

추론 호출의 단계는 [실행 파이프라인](03-usage/execution-pipeline.md), 그래프 구성은 [NeuralNet](03-usage/neural-net.md), 데이터 공유와 버퍼 관리는 [Tensor](03-usage/tensor.md)를 참조하십시오.

## 1. 주요 객체와 역할

vAi 런타임은 Buffer에 데이터를 저장하고 Compute Pipeline으로 연산을 준비합니다. Descriptor Set으로 연산에 사용할 데이터를 연결한 뒤, 실행 명령을 Command Buffer에 기록하여 Queue에 제출합니다. Timeline Semaphore는 제출한 작업의 완료 지점을 추적합니다.

| EVA 객체 | 주요 기능 | vAi에서의 역할 |
|---|---|---|
| `eva::Buffer` | 데이터 저장소 관리 | 입력·출력·가중치 및 전송용 저장 공간 |
| `eva::ComputePipeline` | 연산 실행 상태 구성 | Compute Shader와 특수화 설정을 반영한 실행 객체 |
| `eva::DescriptorSet` | 리소스 연결 정보 관리 | Shader의 입력·출력 위치에 실제 버퍼 범위 연결 |
| `eva::CommandBuffer` | GPU 명령 기록 | 복사, 연산 실행, 동기화 명령 기록 |
| `eva::Queue` | GPU 작업 제출 | Command Buffer를 GPU에서 실행하도록 제출 |
| `eva::TimelineSemaphore` | 작업 완료 지점 추적 | 증가하는 값으로 추론 제출별 완료 지점 구분 |

이 밖에 Barrier는 작업 간 의존성을 지정하는 명령이고, Push Constant와 Specialization Constant는 Shader에 값을 전달하는 방법입니다. Staging Buffer와 Readback Buffer는 별도의 EVA 클래스가 아니라 `eva::Buffer`의 용도를 나타냅니다.

## 2. `eva::Buffer`

`eva::Buffer`는 GPU 연산 및 데이터 전송에 사용하는 바이트 단위의 저장 공간을 관리하는 객체입니다. vAi는 입력·출력·가중치를 이 버퍼에 저장합니다. CPU의 직접 접근 가능 여부는 생성할 때 지정한 메모리 속성에 따라 결정됩니다.

### 2.1 생성과 메모리 접근

`eva::Device::createBuffer({size, usage, reqMemProps})`로 크기, 용도, 필수 메모리 속성을 지정합니다.

| 설정 또는 메서드 | 의미 |
|---|---|
| `BUFFER_USAGE::STORAGE_BUFFER` | Shader가 데이터 버퍼로 사용 |
| `BUFFER_USAGE::TRANSFER_SRC`, `TRANSFER_DST` | 각각 복사의 원본과 대상으로 사용 |
| `MEMORY_PROPERTY::DEVICE_LOCAL` | Device에 로컬인 메모리 요청 |
| `MEMORY_PROPERTY::HOST_VISIBLE` | CPU 주소 공간에 매핑할 수 있는 메모리 요청 |
| `map()`, `unmap()` | CPU 접근용 포인터 획득 및 매핑 해제 |
| `flush()`, `invalidate()` | 각각 Host 쓰기와 Device 쓰기에 대한 캐시 관리 |
| `buffer(offset, size)` | 버퍼와 바이트 범위를 묶은 `eva::BufferRange` 생성 |

`DEVICE_LOCAL`과 `HOST_VISIBLE`은 서로 다른 속성입니다. GPU에 따라 두 속성을 모두 가진 메모리도 있으므로, Device 메모리와 Host 접근 가능 메모리를 항상 별개의 물리 메모리라고 가정하지 않습니다.

CPU가 `map()`으로 접근하려면 `HOST_VISIBLE` 속성이 필요합니다. 메모리가 `HOST_COHERENT`이면 EVA는 `flush()`와 `invalidate()`의 내부 Vulkan 호출을 생략합니다. 이 속성이 GPU 작업 완료까지 보장하는 것은 아니므로, CPU와 GPU의 접근 시점은 별도로 동기화해야 합니다.

vAi의 `Tensor::buffer()`는 버퍼와 `byteOffset()`을 담은 범위를 반환합니다. 이 범위의 기본 크기인 `EVA_WHOLE_SIZE`와 Tensor의 실제 데이터 크기인 `numBytes()`는 구분해야 합니다. 버퍼 재사용과 반환은 `BufferPool` 및 `PooledBuffer`가 담당합니다.

### 2.2 내부 Vulkan 객체

EVA 내부에서는 **버퍼 자체인 `VkBuffer`**와 **실제 저장 공간인 `VkDeviceMemory`**를 함께 관리합니다. `createBuffer()`는 Vulkan에서 필요한 버퍼 생성, 메모리 타입 선택, 메모리 할당과 연결을 묶어서 수행합니다. 현재 구현은 각 버퍼에 전용 메모리를 할당하고 Offset 0에 연결합니다.

`eva::Buffer`는 포인터 대신 핸들로 관리하며, `map()`은 데이터 복사 없이 CPU 접근 주소를 반환합니다.

## 3. 명령 기록과 제출

`eva::CommandBuffer`는 GPU가 수행할 명령을 기록하는 객체이고, `eva::Queue`는 기록된 명령을 GPU에 제출하는 객체입니다. 명령의 기록, 제출 및 실행은 서로 다른 과정입니다.

### 3.1 `eva::CommandBuffer`

`eva::CommandBuffer`에는 데이터 복사, Pipeline과 Descriptor Set 바인딩, 커널 실행 및 Barrier 명령을 기록합니다. 각 메서드는 Command Buffer에 명령을 추가하며, 호출 시점에 GPU 연산을 완료하지 않습니다.

| 주요 메서드 | 기록하거나 수행하는 작업 |
|---|---|
| `begin()`, `end()` | 명령 기록 시작과 종료 |
| `reset()` | 재기록을 위한 초기화 |
| `copyBuffer()` | 버퍼 사이의 데이터 복사 기록 |
| `bindPipeline()` | 사용할 연산 Pipeline 선택 |
| `bindDescSets()` | Shader가 접근할 리소스 연결 선택 |
| `setPushConstants()` | 작은 파라미터 값 기록 |
| `dispatch()` | Workgroup 개수를 지정하여 커널 실행 기록 |
| `dispatch2()` | 요청한 Thread 수를 Workgroup 크기로 나누어 올림한 뒤 실행 기록 |
| `barrier()` | 선행·후속 작업 간 의존성 기록 |

Workgroup은 함께 실행되는 Shader Invocation의 묶음입니다. `dispatch2(N)`은 `N`개 원소를 처리하는 데 필요한 그룹 수를 계산합니다. 올림으로 추가 실행되는 Invocation에 대해서는 Shader에서 범위를 검사해야 합니다.

내부에서는 `VkCommandBuffer`에 Vulkan 명령을 저장합니다. `eva::CommandPool`은 내부의 `VkCommandPool`을 통해 Command Buffer를 할당합니다. `Device::newCommandBuffer()`는 지정한 Queue 타입에 맞는 기본 Pool을 이용해 Command Buffer를 생성합니다.

vAi의 `NeuralNet::recordCommands()`는 유효한 추론 명령을 재사용하고, 입력 Shape나 버퍼 등 재기록 조건이 바뀌면 이전 작업을 기다린 뒤 다시 기록합니다.

### 3.2 `eva::Queue`

`eva::Queue`는 기록이 끝난 Command Buffer를 GPU에서 실행하도록 제출하는 객체입니다. 제출은 작업 완료와 다르므로, CPU에서 결과를 사용하려면 필요한 완료 지점을 기다려야 합니다.

- `Device::queue(type, index)`는 지정한 Queue를 선택합니다.
- `Queue::submit()`은 Command Buffer와 필요한 Semaphore 대기·신호 정보를 제출합니다.
- `Queue::waitIdle()`은 해당 Queue의 작업이 모두 끝날 때까지 Host를 대기시킵니다.
- `CommandBuffer::submit().wait()`는 마지막으로 제출한 Queue 전체의 완료를 기다립니다.

내부에서는 `VkQueue`를 참조합니다. Queue Family는 지원하는 작업 종류 등을 기준으로 Queue를 분류하는 단위이며, Command Buffer는 호환되는 Queue Family에 제출해야 합니다. EVA는 일반 제출에 `vkQueueSubmit()`, Semaphore 정보를 담은 제출 배치에 `vkQueueSubmit2()`를 사용합니다. EVA에서는 필요한 Barrier와 Semaphore 의존성을 명시적으로 구성합니다.

## 4. 연산과 리소스 연결

`eva::ComputePipeline`은 실행할 연산을 정의하고, `eva::DescriptorSet`은 해당 연산이 사용할 버퍼와 범위를 연결합니다. 두 객체를 Command Buffer에 바인딩한 뒤 Dispatch 명령을 기록합니다. 바인딩은 이후 명령에서 사용할 객체를 지정하는 과정입니다.

### 4.1 `eva::ComputePipeline`

`eva::ComputePipeline`은 Compute Shader와 특수화 설정 등을 반영한 GPU 연산 실행 객체입니다. vAi의 Node는 연산에 맞는 Pipeline을 준비하고, `bindPipeline(pipe)`로 사용할 연산을 선택합니다. Pipeline을 만드는 것만으로 연산이 실행되지는 않습니다.

`Device::createComputePipeline()`은 SPIR-V 형태의 Shader 코드로 Pipeline을 생성합니다. SPIR-V는 Vulkan이 입력으로 받는 Shader 중간 표현입니다. EVA는 Shader의 리소스 선언을 분석하는 Reflection을 통해 Layout과 Workgroup 크기 정보를 얻습니다.

내부에서 사용하는 Vulkan 객체와 역할은 다음과 같습니다.

| EVA 인터페이스 | 내부 Vulkan 객체 | 역할 |
|---|---|---|
| `eva::ComputePipeline` | `VkPipeline` | 특수화 설정 등을 반영한 연산 실행 상태 |
| `eva::ShaderModule` | `VkShaderModule` | SPIR-V Shader 코드 |
| `eva::PipelineLayout` | `VkPipelineLayout` | Descriptor Set Layout과 Push Constant 범위 정의 |

SPIR-V 바이트로 생성하는 경로에서는 EVA가 임시 Shader Module을 만들고 Pipeline 생성 후 해제합니다. Pipeline Layout을 생략하면 Reflection으로 구성하고, 명시적으로 전달하면 해당 Layout을 사용합니다.

EVA에서는 Pipeline 준비와 실행 제출을 별도로 수행합니다.

### 4.2 `eva::DescriptorSet`

`eva::DescriptorSet`은 Shader가 사용할 버퍼와 바이트 범위를 연결하는 객체입니다. Shader에 선언된 **Set은 리소스 묶음 번호**, **Binding은 그 묶음 안의 연결 위치**입니다.

예를 들어 Activation Shader의 Set 0에서 Binding 0은 출력, Binding 1은 입력입니다. `descSet.write({outBuf, inBuf})`는 이 위치에 실제 버퍼 범위를 연결하고, `cmdBuff.bindDescSets({descSet})`는 실행에 사용할 Set을 지정합니다. 데이터 자체를 복사하는 동작은 아닙니다.

내부의 `VkDescriptorSet`은 리소스 연결 정보를 보관합니다. `eva::DescriptorSetLayout`은 `VkDescriptorSetLayout`을 통해 각 Binding의 리소스 타입과 개수를 정의합니다. `eva::DescriptorPool`은 내부의 `VkDescriptorPool`에서 Set을 할당합니다. `DescriptorSet::write()`는 Layout에 맞춰 `vkUpdateDescriptorSets()`를 호출합니다.

Descriptor Set은 버퍼의 수명을 관리하지 않으므로, GPU가 사용하는 동안 Set과 연결한 버퍼를 유효하게 유지해야 합니다.

EVA에서는 연결 정보를 별도의 Set으로 구성합니다.

### 4.3 Activation의 명령 기록

다음은 `Activation::run()`에서 연산 명령을 기록하는 코드입니다. `pipe`와 `descSet`은 `prepare()`에서 준비하며, `cmdBuff`는 `NeuralNet`이 전달합니다.

```cpp
auto outBuf = (*this)["out0"].buffer();
auto inBuf  = config_.inPlace ? eva::Buffer() : (*this)["in0"].buffer();

descSet.write({ outBuf, inBuf });

uint32_t N = (*this)["out0"].numElements();
struct {
    uint32_t N;
} constants = { N };

cmdBuff
    .bindPipeline(pipe)
    .bindDescSets({descSet})
    .setPushConstants(0, sizeof(constants), &constants)
    .dispatch2(N)
    .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
```

위 예제는 Pipeline과 Descriptor Set 바인딩, 원소 수 전달, Dispatch 및 후속 읽기를 위한 Barrier를 기록합니다. GPU는 이후 Queue에 제출된 명령을 실행합니다. In-place 모드에서는 출력 버퍼를 입력으로도 사용하며, 빈 입력 Descriptor를 기록하려면 Device의 `nullDescriptor` 기능이 필요합니다.

## 5. 동기화

Barrier는 작업 간 실행 및 메모리 접근 의존성을 지정합니다. Timeline Semaphore는 제출한 작업의 완료 지점을 값으로 구분하며, 제출 간 의존성과 Host의 완료 대기에 사용합니다.

### 5.1 Barrier

`barrier()`는 선행 작업과 후속 작업의 실행 및 메모리 접근 의존성을 기록합니다. 예를 들어 앞선 커널이 출력 버퍼에 쓴 결과를 다음 커널이 읽는다면, 그 쓰기 결과가 다음 읽기에서 보이도록 연결해야 합니다.

```cpp
cmdBuff.barrier(
    eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ
);
```

여기서 `COMPUTE_WRITE`는 선행 Compute 쓰기, `outBuf`는 대상 버퍼 범위, `COMPUTE_READ`는 후속 Compute 읽기입니다. EVA의 `/` 연산자는 이 정보를 묶는 표현입니다. CPU가 GPU 완료를 기다리는 명령은 아닙니다.

내부에서는 `vkCmdPipelineBarrier2()`를 기록합니다. 버퍼를 지정하면 `VkBufferMemoryBarrier2`에 Stage, Access 및 버퍼 범위를 설정합니다. Stage는 의존성을 적용할 실행 단계를 나타내고, Access는 연결할 읽기·쓰기 접근을 나타냅니다. Barrier는 별도로 생성해 보관하는 Vulkan 객체 핸들이 아닙니다.

다른 Queue 사이의 의존성은 Semaphore 등과 함께 구성해야 합니다. 자세한 동기화 규칙은 [Vulkan Specification의 Synchronization](https://docs.vulkan.org/spec/latest/chapters/synchronization.html)을 참조하십시오.

### 5.2 `eva::TimelineSemaphore`

`eva::TimelineSemaphore`는 증가하는 64비트 부호 없는 카운터로 GPU 작업의 완료 지점을 추적하는 객체입니다. 예를 들어 제출 A가 끝나면 1, 제출 B가 끝나면 2를 Signal하도록 구성하면, Host는 값 2에 대한 Wait로 해당 완료 지점을 기다릴 수 있습니다. Wait는 카운터가 요청 값 이상이면 충족됩니다.

| EVA 인터페이스 | 역할 |
|---|---|
| `Device::createTimelineSemaphore(0)` | 초기값 0인 Timeline Semaphore 생성 |
| `timeline(value, stage)` | 제출에 사용할 Semaphore 값과 실행 단계 지정 |
| `value()` | 현재 카운터 조회 |
| `wait(value, timeout)` | Host에서 지정 값 이상이 될 때까지 대기 |
| `signal(value)` | Host에서 카운터 Signal |

vAi는 `NeuralNet`마다 Timeline Semaphore를 만들고, 추론마다 증가하는 `submitCount_`를 완료 값으로 사용합니다.

```cpp
device_.queue() << cmdBuffer_ / timeline_(++submitCount_);
```

위 예제는 Command Buffer를 제출하면서 해당 Timeline 값을 Signal하도록 지정합니다. 기본 Stage는 `ALL_COMMANDS`입니다. `NeuralNet::sync()`는 마지막 제출 값을 기다리고, 입력 전송용 버퍼를 재사용할 때도 완료 값을 사용합니다. 제출 번호는 `NeuralNet`이 관리하며, Timeline Semaphore는 지정한 값의 Signal과 Wait를 처리합니다.

내부 객체는 Timeline 타입으로 생성한 `VkSemaphore`입니다. EVA는 `vkGetSemaphoreCounterValue()`와 `vkWaitSemaphores()` 등으로 조회와 대기를 수행합니다. Binary Semaphore가 Signal 상태를 사용하는 것과 달리, Timeline Semaphore는 하나의 객체에서 여러 완료 지점을 값으로 구분합니다. 자세한 동작은 [Vulkan의 Semaphore 규칙](https://docs.vulkan.org/spec/latest/chapters/synchronization.html#synchronization-semaphores)을 참조하십시오.

완료 값에 도달했다는 사실만으로 모든 후속 접근의 메모리 의존성이 구성되는 것은 아니므로, Signal·Wait 범위와 필요한 Barrier도 함께 확인해야 합니다.

## 6. Host와 Device 사이의 데이터 전송

Staging Buffer와 Readback Buffer는 Host와 Device 사이의 데이터 전송에 사용하는 `eva::Buffer`입니다. 두 이름은 각각 업로드와 다운로드에 사용하는 버퍼의 용도를 나타냅니다.

### 6.1 입력과 가중치 업로드

Staging Buffer는 CPU에서 준비한 데이터를 Device 연산용 버퍼로 옮기는 중간 저장 공간입니다. vAi의 입력 업로드는 다음 순서로 구성합니다.

1. CPU가 쓸 수 있는 `HOST_VISIBLE | HOST_COHERENT` Staging Buffer를 준비합니다.
2. `map()`으로 얻은 포인터에 Host 데이터를 복사합니다.
3. `copyBuffer()`로 Staging Buffer에서 Tensor의 Device 버퍼로 복사를 기록합니다.
4. Transfer 쓰기 결과를 Compute 읽기에 연결하는 Barrier를 기록하고 Queue에 제출합니다.
5. 해당 GPU 작업이 끝난 뒤 Staging Buffer를 재사용합니다.

Staging Buffer와 Device 버퍼는 모두 내부적으로 `VkBuffer`와 `VkDeviceMemory`를 사용합니다. Staging Buffer에는 `TRANSFER_SRC`, 복사 대상에는 `TRANSFER_DST` 용도가 필요합니다. 가중치도 전송용 버퍼를 이용해 업로드하지만, 명령 구성과 재사용 시점은 입력 업로드와 별도 경로에서 관리합니다.

### 6.2 결과 다운로드

Readback Buffer는 CPU에서 사용할 Device 연산 결과를 복사하는 저장 공간입니다. vAi의 `Tensor::downloadToHost()`는 Device 버퍼에서 Readback Buffer로 복사를 제출하고, 완료를 기다린 뒤 매핑한 데이터를 Host 저장소에 복사합니다.

복사의 원본에는 `TRANSFER_SRC`, Readback Buffer에는 `TRANSFER_DST` 용도가 필요합니다. 현재 Readback은 `HOST_VISIBLE | HOST_CACHED` 메모리를 요청하고, CPU에서 읽기 전에 `invalidate()`를 호출합니다. 실제 메모리가 Coherent이면 내부 캐시 무효화 호출은 생략합니다.

**GPU 작업의 완료와 CPU에서 데이터를 읽을 수 있는 메모리 가시성은 함께 확인해야 합니다.** 복사 전후의 접근 의존성, 완료 대기, 캐시 관리는 각각의 역할이 있습니다. 실제 Readback 구현의 동기화와 Device 선택은 [Tensor 문서](03-usage/tensor.md) 및 [tensor.cpp](../../vai/engine/tensor.cpp)를 함께 확인하십시오.

## 7. Shader 파라미터

Push Constant는 명령 기록 시 작은 값을 Shader에 전달하고, Specialization Constant는 Pipeline 생성 시 Shader 상수를 지정합니다. 값의 설정 시점에 따라 변경 방법도 달라집니다.

| 구분 | Push Constant | Specialization Constant | Uniform Buffer |
|---|---|---|---|
| EVA 인터페이스 | `CommandBuffer::setPushConstants()` | `eva::SpecializationConstant`, `eva::constant_id<ID>()` | `eva::Buffer`와 `DescriptorSet::write()` |
| 설정 시점 | 명령 기록 시 | Pipeline 생성 시 | Shader가 읽을 버퍼 데이터를 준비할 때 |
| 변경 방법 | 새 값을 명령에 기록 | 다른 값으로 생성한 Pipeline 사용 | 버퍼 내용 갱신 및 접근 동기화 |
| Activation 예 | 원소 수 `N` | In-place 여부, 입출력 Dtype, Epilogue | 해당 Node에서는 사용하지 않음 |

### 7.1 Push Constant

`setPushConstants()`는 원소 수나 Offset처럼 작은 값을 명령에 담아 Shader에 전달합니다. Activation은 `N`을 전달하여 Shader가 처리 범위를 판단하게 합니다. 기록된 Command Buffer를 다시 제출하면 같은 `N`을 사용하므로, 값이 바뀌면 그 값을 기록하는 명령도 갱신해야 합니다.

내부에서는 `vkCmdPushConstants()`를 기록합니다. 값의 Offset, 크기 및 Shader Stage는 `VkPipelineLayout`에 선언한 범위와 일치해야 합니다. 값을 전달하기 위한 별도의 Buffer와 Descriptor Set은 필요하지 않습니다.

### 7.2 Specialization Constant

`eva::SpecializationConstant`는 Shader에 선언한 상수의 값을 Pipeline 생성 시 지정하기 위한 데이터입니다. Shader는 ID로 상수를 선언하고, C++에서는 `eva::constant_id<ID>(value)`로 해당 ID에 값을 연결합니다.

다음은 `Activation::prepare()`의 Pipeline 요청입니다.

```cpp
pipe = pipelinePool().request("activation", {
    eva::constant_id<0>(config_.inPlace ? 1u : 0u),
    eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),
    eva::constant_id<SPEC_ID_TYPE_I>(iDtype),
    eva::constant_id<SPEC_ID_EPILOGUE>((uint32_t) config_.epilogue)
});
```

이 설정은 In-place 여부, 입출력 자료형, 후처리 연산을 결정합니다. 실행마다 전달하는 원소 수 `N`과 달리 Pipeline 생성 시 고정되므로, 설정을 바꾸려면 그 설정으로 생성한 Pipeline을 사용합니다. vAi의 `PipelinePool`은 같은 설정의 Pipeline을 재사용합니다.

내부에서는 ID별 값과 Offset·크기를 `VkSpecializationInfo`로 구성하고, Shader Stage의 생성 정보에 연결하여 `VkPipeline`을 생성합니다. `VkSpecializationInfo`는 독립적인 Vulkan 객체 핸들이 아니라 Pipeline 생성에 전달하는 정보 구조체입니다. 상세 사용 규칙은 [Vulkan Specification의 Pipelines](https://docs.vulkan.org/spec/latest/chapters/pipelines.html)를 참조하십시오.

## 8. 추론 실행과 리소스 수명

런타임은 입력과 연산 리소스를 준비하고, GPU 명령을 기록·제출한 뒤 필요한 시점에 작업 완료를 기다립니다. 다음 표는 Host 입력으로 Activation을 실행할 때 각 단계에서 사용하는 EVA 객체를 나타냅니다.

| 단계 | 담당 코드 | 사용하는 EVA 객체와 기능 |
|---|---|---|
| 입력 준비 | `NeuralNet::prepareInputs()` | `Buffer`에 입력을 업로드하고 복사 및 Barrier 제출 |
| 연산 준비 | `Activation::prepare()` | `ComputePipeline` 요청 및 `DescriptorSet` 할당 |
| 명령 기록 | `NeuralNet::recordCommands()`, `Activation::run()` | 출력 버퍼 준비, Descriptor 연결, 바인딩, Push Constant, Dispatch, Barrier 기록 |
| 실행 제출 | `NeuralNet::operator()()` | `Queue`에 Command Buffer와 Timeline Signal 값 제출 |
| 완료 대기 | `NeuralNet::sync()` | `TimelineSemaphore`의 마지막 제출 값 대기 |
| Host 결과 확보 | `Tensor::downloadToHost()` | 별도의 Readback 복사 제출 및 대기 후 Host 저장소로 복사 |

최초 추론에서는 연산 리소스를 준비하고 GPU 명령을 기록합니다. 이후 재기록 조건이 바뀌지 않으면 기존 추론 명령을 재사용합니다. `sync()`의 추론 완료 대기와 `downloadToHost()`의 결과 복사는 별개의 동작입니다.

**EVA 핸들의 수명과 내부 GPU 리소스의 수명은 구분해야 합니다.** 예를 들어 `eva::Buffer`를 복사해도 데이터가 복제되는 것은 아니며, 같은 내부 구현을 참조합니다. 지역 핸들이 범위를 벗어나도 리소스가 유지되지만, `destroy()`로 구현을 제거하면 복사해 둔 핸들도 함께 무효화됩니다.

`eva::Device`는 자신이 생성한 리소스를 관리하고, 정리 시 Device의 작업 완료를 기다린 뒤 남은 리소스를 해제합니다. Command Buffer와 Descriptor Set은 각각 Pool에서 관리합니다. 개별 리소스를 해제하거나 재사용할 때는 해당 리소스를 사용하는 GPU 작업이 끝났는지 호출자가 확인해야 합니다.
