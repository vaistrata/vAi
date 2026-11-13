# 프로파일링

vAi는 각 노드의 **GPU 실행 시간**을 측정할 수 있습니다. 노드의 `run()` 앞뒤에 타임스탬프를 기록하고, forward가 끝난 뒤 그 값을 읽어 누적합니다.

## 프로파일링 활성화

프로파일링은 노드에 태그를 하나라도 달면 활성화됩니다. 반대로 태그를 하나도 달지 않으면 프로파일링으로 인한 추가 비용도 없습니다.

```cpp
net.decoder.layers[i].mhAttn.attn.setProfileTag("Attn");
```


## 같은 태그는 합쳐진다

같은 태그를 단 노드들의 시간은 하나로 합산됩니다.

```cpp
for (uint32_t i = 0; i < net.modelCfg.languageCfg.numLayers; ++i)
{
    net.decoder.layers[i].norm1.setProfileTag("LayerNorm");
    net.decoder.layers[i].norm2.setProfileTag("LayerNorm");
}
```

블록이 12개라면 총 24개 노드의 시간이 `LayerNorm` 한 줄에 모입니다.

태그에 `.`을 넣으면 계층으로 묶을 수 있습니다.

```cpp
b.mhAttn.projQKV.setProfileTag("Linear.[768x2304]");
b.mhAttn.projOut.setProfileTag("Linear.[768x768]");
```

이 경우 `Linear` 아래에 형상별 시간이 나뉘며, 부모에는 그 아래 항목들의 시간이 합산됩니다.

## 여러 forward는 평균으로 누적된다

forward가 끝날 때마다 측정값이 누적되고, `printProfile()`은 지금까지 기록된 forward의 평균을 출력합니다.

```cpp
for (int i = 0; i < 5; ++i)   // 워밍업
    net(inDict);

net.resetProfileRecords();   // 워밍업 기록을 버린다

for (int i = 0; i < 10; ++i)
    net(inDict);

net.printProfile();          // 10회 평균
```

초기 실행에는 셰이더 컴파일이나 버퍼 준비 같은 일회성 비용이 섞일 수 있으므로, 워밍업 뒤에 측정하는 편이 좋습니다. `resetProfileRecords()`는 지금까지 누적된 기록을 모두 지우고 다시 측정할 때 사용합니다.

## 출력과 기록 리셋

```text
=== Per-node GPU profile (avg over 10 iters) ===
  ├── Conv1x1bn                 :  28.1411 ms  (15.05%)  ██████
  │   ├── [1024x256@80]         :   3.5974 ms  ( 1.92%)  █
  │   └── [1280x512@40]         :   4.7626 ms  ( 2.55%)  █
  ├── Conv3x3s1                 : 129.6934 ms  (69.36%)  ████████████████████████████
  OTHER (untagged)              :   6.1002 ms  ( 3.25%)
  TOTAL (full forward)          : 187.5394 ms
```

| 항목      | 의미                     |
| ------- | ---------------------- |
| `ms`    | forward 한 번당 평균 GPU 시간 |
| `%`     | `TOTAL` 대비 비율          |
| `TOTAL` | 전체 forward의 GPU 시간     |
| `OTHER` | 태그가 붙지 않은 구간의 시간       |

`OTHER`가 크다면 아직 태그를 달지 않은 구간에서 많은 시간이 쓰이고 있다는 뜻입니다.

`printProfile()`은 출력한 뒤 누적 기록을 초기화하는 것이 기본 동작입니다. 기록을 유지한 채 중간 결과만 확인하려면 `printProfile(false)`를 사용합니다.

기본 동작을 이용하면 prefill과 decode처럼 같은 태그 구성의 두 구간을 간단히 나눠 측정할 수 있습니다.

```cpp
net(inDict);
net.printProfile();          // prefill 출력 후 기록 초기화

for (int i = 0; i < maxNewTokens; ++i)
    net(inDict);

net.printProfile();          // decode 구간만 출력
```

첫 번째 출력과 함께 기록이 초기화되므로, 두 번째 출력에는 decode에서 새로 누적된 기록만 포함됩니다.

## 태그 변경

측정 도중 태그를 바꿔도 됩니다. 변경된 태그는 다음 forward에서 커맨드버퍼를 다시 기록하면서 반영됩니다.

전체 프로파일을 확인한 뒤 특정 병목만 자세히 보고 싶다면 기존 태그를 지우고 원하는 노드에 다시 달 수 있습니다.

```cpp
net.clearProfileTags();

net.decoder.layers[0].ffn.fc1.setProfileTag("fc1");
net.decoder.layers[0].ffn.fc2.setProfileTag("fc2");
```

특정 노드의 태그만 제거하려면 빈 문자열을 줍니다.

```cpp
net.lmHead.setProfileTag("");
```

태그 구성이 바뀌면 기존 기록은 초기화됩니다. 측정 대상이 서로 다른 값을 하나의 평균에 섞지 않기 위해서입니다.

## 값으로 읽기

출력하지 않고 특정 태그의 평균 GPU 시간만 숫자로 읽을 수도 있습니다.

```cpp
double ms = net.getGpuTimeMs("Linear.[768x2304]");
```

`getGpuTimeMs()`는 `.`을 계층으로 해석하지 않고 **태그 전체를 하나의 이름으로 찾습니다**. 따라서 계층형 태그를 사용했다면 `"Linear"`이 아니라 `"Linear.[768x2304]"`처럼 정확한 전체 태그를 넘겨야 합니다.

해당 태그가 없거나 아직 기록된 forward가 없다면 `0.0`을 반환합니다.

## 알아둘 점

`printProfile()`, `getGpuTimeMs()`, `resetProfileRecords()`은 GPU 타임스탬프를 읽기 위해 내부에서 이전 forward의 완료를 기다립니다. 측정 루프 안에서 매번 호출하면 실행이 직렬화될 수 있으므로 가능하면 루프 밖에서 호출하세요.

프로파일링 중에는 태그가 붙은 노드의 앞뒤에 타임스탬프 명령이 추가됩니다. 측정 단위는 노드까지이며, 하나의 노드가 여러 커널을 디스패치하더라도 전체가 하나의 구간으로 측정됩니다.

## API 정리

| API                                         | 역할                      |
| ------------------------------------------- | ----------------------- |
| `Node::setProfileTag(tag)`                  | 태그 지정. 빈 문자열이면 제거       |
| `NeuralNet::clearProfileTags()`             | 모든 태그 제거                |
| `NeuralNet::resetProfileRecords()`          | 누적 기록 초기화               |
| `NeuralNet::printProfile(withResetRecords)` | 평균 출력. 기본값은 출력 후 기록 초기화 |
| `NeuralNet::getGpuTimeMs(tag)`              | 특정 태그의 평균 GPU 시간 반환     |
