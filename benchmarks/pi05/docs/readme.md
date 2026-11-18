# pi0.5 벤치마크 실행 가이드 (AMD: vAi vs MIGraphX)

AMD PC(Linux)에서 pi0.5 LIBERO 정책 한 번 예측(prefill 1회 + denoise 10 step)의 지연 시간을 **vAi(Vulkan)** 와 **MIGraphX(ROCm)** 로 같은 조건에서 재고 비교하는 방법을 설명합니다.
NVIDIA 쪽 비교 대상(vla.cpp, TensorRT Edge-LLM)과 벤치 전반의 설명은 [../README.md](../README.md)에 있습니다.

## 0. 전체 흐름 요약

```
data/ckpts/pi05/lerobot/libero/  (LeRobot FP32 체크포인트)
   │
   ├─ scripts/reference.py ──▶ data/benchmarks/pi05/libero/{reference,noise,images}_2_50_10.bin ─┐
   │     (LeRobot PyTorch FP32, 정답 액션 + 입력 noise + 전처리된 이미지)                           │
   │                                                                                             ├─▶ bench-pi05-vai --workload migraphx
   ├─ (vAi는 model.safetensors 를 직접 읽음) ─────────────────────────────────────────────────────┤
   │                                                                                             │
   └─ scripts/export-onnx.py ─▶ data/benchmarks/pi05/libero/{prefill,step}.fp16.onnx ... ────────┴─▶ bench-pi05-migraphx
         (FP16 ONNX 번들, 상수 폴딩 기본)
```

이미 준비가 끝난 PC에서 처음부터 끝까지 돌리는 명령은 다음과 같습니다. 각 단계의 자세한 내용은 아래 절에 있습니다.

```bash
cd <repo root>
source ~/.cache/vai/venv-pi05-export/bin/activate                    # §3 Python 환경

python benchmarks/pi05/scripts/reference.py --cameras 2 --chunk 50 --steps 10   # §4 reference .bin
./benchmarks/scripts/setup-migraphx.sh                                          # §5 MIGraphX 2.18 + ROCm 10.1 (최초 1회)
python benchmarks/pi05/scripts/export-onnx.py --cxx g++-14                      # §6 FP16 ONNX 번들

cmake -B build -DCMAKE_BUILD_TYPE=Release -DVAI_BUILD_BENCH=ON \
      -DCMAKE_CXX_COMPILER=g++-14 -DMIGRAPHX_GEMM_PROVIDER=mlir                 # §7 빌드
cmake --build build --target bench-pi05-vai bench-pi05-migraphx

./benchmarks/pi05/scripts/run-amd.sh                                            # §8 실행 (GPU 클럭 high 고정)
```

## 1. 무엇을 비교하나

두 벤치 모두 [`common/workload.h`](../common/workload.h)의 `migraphx` 워크로드(`vsMIGraphX`)를 그대로 씁니다.

| 항목 | 값 |
|---|---|
| 체크포인트 | LeRobot LIBERO: [`lerobot/pi05_libero_base`](https://huggingface.co/lerobot/pi05_libero_base) (revision `5fcf06fc42b6717e22ab885fd381a096574b12c7`), FP32 `model.safetensors` 14.5 GB |
| 정책(프롬프트) | LeRobot 자체 정책: 이산화된 state가 프롬프트에 들어감 |
| 카메라 | 2대 (`data/images/cam_high.jpg`, `cam_left_wrist.jpg`), 224×224 |
| 토큰 | 이미지 512 (카메라당 256) + 텍스트 프롬프트 154. 벤치 표에 `512 + 154`로 나옵니다 |
| 액션 | chunk 50 × dim 32, denoise 10 step (Euler) |
| 관측값 | [`common/bench-inputs.h`](../../common/bench-inputs.h): task 문장, state(0.1 … 0.6, 32까지 0 패딩), noise seed 0 |

측정 방식([`bench-utils.h`](../../common/bench-utils.h)의 `runVLABench`):

- 시간 측정 없이 1회 예측해서 reference와 비교합니다. 기준을 넘으면 지연 시간을 출력하지 않고 exit 1로 끝납니다.
- 이어서 warmup 3회, 측정 20회를 합니다. **mean ± SD**와 min(ms)을 출력하고, SD가 평균의 10%를 넘으면 경고합니다.
- 측정 1회에는 입력 업로드(H2D), prefill, 10 step decode, 액션 readback(D2H)까지 들어갑니다.
- MIGraphX의 그래프 컴파일 시간은 측정에 들어가지 않습니다 (`compile: N s`로 따로 출력).

## 2. 검증된 환경

2026-11-18 기준으로 아래 환경에서 측정했습니다.

| 항목 | 버전 |
|---|---|
| OS | Ubuntu 24.04.4 LTS (x86_64) |
| GPU | AMD Radeon 8060S (Strix Halo, **gfx1151**, APU) |
| Vulkan 드라이버 (vAi) | Mesa RADV 25.2.8 (`vulkaninfo --summary`) |
| 시스템 ROCm | 7.2.1 (`/opt/rocm`): MIGraphX 벤치 **빌드 시 HIP 헤더용으로만** 사용 |
| MIGraphX 런타임 | **MIGraphX 2.18.0 + ROCm 10.1.0 런타임** (`~/.cache/vai/migraphx-2.18.0`) |
| C++ 컴파일러 | GCC 14 (`g++-14`). vAi 소스와 export 헬퍼 모두 C++23 필요 |
| Python | 3.12.3 (uv venv) |

## 3. 사전 준비

### 3.1 입력 파일

모든 경로는 레포의 `data/` 아래입니다. 빌드할 때 `VAI_DATA_DIR`로 절대 경로가 박히므로 벤치는 어느 디렉터리에서 실행해도 됩니다.

| 파일 | 용도 | 받는 법 |
|---|---|---|
| `data/ckpts/pi05/lerobot/libero/model.safetensors` | 모든 단계 | 아래 `hf download` |
| `data/ckpts/pi05/lerobot/libero/config.json` | reference.py, export-onnx.py | 위와 같음 |
| `data/ckpts/pi05/lerobot/libero/policy_preprocessor.json`, `policy_postprocessor.json` | reference.py (LeRobot 전처리) | 위와 같음 |
| `data/tokenizers/paligemma/tokenizer.json` | vAi 벤치, MIGraphX 벤치, export-onnx.py | `google/paligemma-3b-pt-224` (gated, [../README.md](../README.md#model-preparation) 참고) |
| `data/images/cam_high.jpg`, `cam_left_wrist.jpg` (`cam_right_wrist.jpg`) | 모든 단계 | 데모와 같은 이미지 |

```bash
# 체크포인트 (huggingface_hub 1.x 의 hf CLI)
hf download lerobot/pi05_libero_base \
    --revision 5fcf06fc42b6717e22ab885fd381a096574b12c7 \
    --local-dir data/ckpts/pi05/lerobot/libero
```

> [!NOTE]
> `reference.py`는 LeRobot 전처리기(`policy_preprocessor.json`의 `tokenizer_processor`)를 통해 토크나이저 `google/paligemma-3b-pt-224`를 **Hugging Face Hub에서** 불러옵니다. gated 레포이므로 모델 페이지에서 라이선스에 동의하고 `hf auth login`(또는 `HF_TOKEN`)을 해 두어야 합니다. 이미 `~/.cache/huggingface/hub`에 받아져 있으면 다시 받지 않습니다.

### 3.2 Python 환경 (reference.py, export-onnx.py 공용)

두 스크립트는 같은 venv에서 돌립니다. 그래야 reference와 ONNX 번들이 같은 LeRobot 코드에서 나옵니다. 현재 PC의 venv는 `~/.cache/vai/venv-pi05-export`(uv로 만든 Python 3.12.3)입니다.

| 패키지 | 버전 | 비고 |
|---|---|---|
| Python | 3.12 | lerobot 0.6.0이 `>=3.12` 요구 |
| **lerobot** | **0.6.0** | export-onnx.py가 버전을 정확히 검사 |
| **transformers** | **5.5.4** | export-onnx.py가 검사 (lerobot 0.6.0의 `transformers-dep`: `>=5.4,<5.6`) |
| **torch** | **2.8.0** (`+cpu`) | export-onnx.py가 검사 (`+cpu` 같은 local tag는 무시) |
| torchvision | 0.23.0 (`+cpu`) | torch 2.8.0과 짝 |
| **onnx** | **1.19.0** | export-onnx.py가 검사. export에만 필요 |
| numpy | 2.2.6 | lerobot 0.6.0: `>=2.0,<2.3` |
| pillow | 12.3.0 | reference.py 이미지 로딩 |
| safetensors | 0.8.0 | |
| huggingface_hub | 1.33.0 | `hf` CLI |
| tokenizers | 0.22.2 | |

굵게 표시한 4개는 [`export-onnx.py`](../scripts/export-onnx.py)가 시작할 때 정확히 일치하는지 검사하고, 다르면 `RuntimeError: <pkg> <ver> required`로 멈춥니다. `reference.py`는 검사하지 않지만 같은 버전을 써야 합니다.

같은 환경을 새로 만들 때:

```bash
uv venv ~/.cache/vai/venv-pi05-export --python 3.12
source ~/.cache/vai/venv-pi05-export/bin/activate
# CPU 빌드 torch 먼저: 그러지 않으면 CUDA 빌드(수 GB)가 설치됩니다
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install lerobot==0.6.0 transformers==5.5.4 onnx==1.19.0 numpy==2.2.6 pillow
uv pip list | grep -iE '^(lerobot|transformers|torch|onnx) '    # 위 표와 같은지 확인
```

> [!TIP]
> CPU 빌드 torch를 쓰는 것이 기본입니다. `reference.py`는 `torch.cuda.is_available()`가 참이면 GPU(ROCm torch 포함)에서 돕니다. reference는 FP32라 CPU에서 돌려도 충분하고, 환경이 바뀌어도 결과가 같게 나옵니다. 두 스크립트 모두 FP32 체크포인트(14.5 GB)를 메모리에 올리므로 RAM이 넉넉해야 합니다.

## 4. reference .bin 만들기 (`reference.py`)

[`scripts/reference.py`](../scripts/reference.py)는 LeRobot의 FP32 PyTorch 정책(`PI05Policy`)으로 정답 액션을 만듭니다. 전처리는 체크포인트의 `policy_preprocessor.json`과 `PI05Policy._preprocess_images`를 그대로 쓰고, noise는 seed 0의 torch RNG에서 뽑습니다.

```bash
source ~/.cache/vai/venv-pi05-export/bin/activate
python benchmarks/pi05/scripts/reference.py --cameras 2 --chunk 50 --steps 10
# 기본 정책이 lerobot 이므로 --policy 는 생략 (MIGraphX, vla.cpp 워크로드)
```

생성 파일 (`data/benchmarks/pi05/libero/`, little-endian FP32, 이름 규칙은 `<kind>_<cameras>_<chunk>_<steps>.bin`):

| 파일 | 내용 | 크기 |
|---|---|---|
| `reference_2_50_10.bin` | 모델이 낸 정규화된 액션 `[50, 32]` (정답) | 6,400 B |
| `noise_2_50_10.bin` | 그 액션이 시작한 noise `[50, 32]` | 6,400 B |
| `images_2_50_10.bin` | 모델이 실제로 본 픽셀 `[2, 3, 224, 224]`, CHW, `[-1, 1]` | 1,204,224 B |

- 벤치는 세 파일을 **모두** 읽습니다. vAi의 `sampleNoise`는 torch RNG와 같은 값을 내지 않고, vAi의 letterbox도 LeRobot의 `resize_with_pad`와 다릅니다. 같은 noise와 같은 픽셀을 넣어야 **이미지 전처리를 뺀 모델만** 비교할 수 있습니다.
- 파일이 없으면 벤치는 noise와 이미지를 직접 만들고 `accuracy: unverified`를 출력합니다. 크기가 다른 파일이 있으면 다른 설정에서 만든 것으로 보고 에러로 끝납니다.
- 워크로드(카메라, chunk, step)를 바꾸면 reference도 새로 만들어야 합니다. 참고로 TensorRT 워크로드는 `--cameras 2 --chunk 10 --steps 10 --policy openpi`입니다.

## 5. MIGraphX 설치 (`setup-migraphx.sh`)

### 5.1 왜 ROCm 10.1 런타임인가

벤치가 고정한 MIGraphX는 **2.18**입니다 ([`../CMakeLists.txt`](../CMakeLists.txt)가 `find_package(migraphx 2.18...<2.19)`로 찾고, 다른 릴리스면 벤치를 빌드하지 않음). 2.18은 **ROCm 10.1.0**과 함께 배포되는 릴리스이고, 시스템 ROCm 7.2.1 위에서 돌리면 많이 느립니다 (§10 참고). 그래서 [`setup-migraphx.sh`](../../scripts/setup-migraphx.sh)는 다음을 합니다.

- AMD 저장소의 Ubuntu 24.04 패키지 `amdrocm10-migraphx`, `amdrocm10-migraphx-dev` 2.18.0-1을 받아 **apt 없이, root 없이** 풉니다 (SHA256 검증).
- GPU별 ROCm 10.1.0 tarball(`therock-dist-linux-<gfx>-10.1.0.tar.gz`, 약 2 GB)에서 MIGraphX가 실제로 읽는 런타임(HIP, hiprtc, comgr, HSA, MIOpen, rocBLAS, hipBLASLt와 커널 라이브러리)만 골라 같은 `lib/`에 풉니다.
- 설치 위치는 `${XDG_CACHE_HOME:-~/.cache}/vai/migraphx-2.18.0`이고, `~/.cmake/packages/migraphx/vai-2.18.0`에 CMake user package registry로 등록합니다. 그래서 CMake에 플래그를 따로 줄 필요가 없습니다.
- 벤치 바이너리의 RUNPATH가 이 `lib/`를 가리키므로, 시스템에 깔린 ROCm 버전과 상관없이 **실행 시에는 ROCm 10.1 라이브러리만** 읽습니다. 시스템 ROCm(`/opt/rocm`)은 빌드할 때 HIP 헤더만 제공합니다.

### 5.2 설치

필요한 것: Linux x86_64, `curl`, `dpkg-deb`, `tar`, `sha256sum`, amdgpu 커널 드라이버, 시스템 ROCm(HIP 헤더용).

```bash
./benchmarks/scripts/setup-migraphx.sh                 # 다운로드 약 2.1 GB, 설치 약 1.6 GB. GPU는 자동 감지 (gfx1151)
./benchmarks/scripts/setup-migraphx.sh --arch gfx1151  # GPU 지정
./benchmarks/scripts/setup-migraphx.sh --force         # 재설치
./benchmarks/scripts/setup-migraphx.sh --uninstall     # 설치본과 registry 항목 제거
```

- 설치 후 `~/.cache/vai/migraphx-2.18.0/.vai-pin`에 `2.18.0-1 rocm-10.1.0 gfx1151`이 적힙니다.
- tarball SHA256은 gfx1151 것만 고정되어 있습니다. 다른 GPU는 검증 없이 설치되고 경고가 나옵니다.
- 다른 곳에 설치한 2.18을 쓰려면 configure할 때 `-Dmigraphx_DIR=<prefix>/lib/cmake/migraphx`를 줍니다.

빌드한 뒤 실제로 ROCm 10.1을 읽는지 확인하는 방법:

```bash
ldd build/bin/bench-pi05-migraphx | grep -E 'amdhip|migraphx_c'
# libmigraphx_c.so.3 => ~/.cache/vai/migraphx-2.18.0/lib/libmigraphx_c.so.3
# libamdhip64.so.7  => ~/.cache/vai/migraphx-2.18.0/lib/libamdhip64.so.7
```

## 6. MIGraphX용 ONNX 번들 export (`export-onnx.py`)

[`scripts/export-onnx.py`](../scripts/export-onnx.py)는 LIBERO 체크포인트를 MIGraphX 벤치가 읽는 **v5 split 번들**로 내보냅니다. prefill 그래프는 예측마다 1회 돌아 K/V cache를 채우고, step 그래프는 Euler step 1회 분량으로 10번 재사용됩니다. GPU 없이 CPU에서 돕니다.

```bash
source ~/.cache/vai/venv-pi05-export/bin/activate
python benchmarks/pi05/scripts/export-onnx.py --cxx g++-14
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--cxx` | `$CXX` 또는 `c++` | 입력 준비용 C++ 헬퍼를 빌드할 컴파일러. C++23 explicit object parameter가 필요하므로 Ubuntu 24.04 기본 g++ 13으로는 안 되고 **`g++-14`(또는 clang++ 18 이상)** 를 줘야 합니다 |
| `--constant-folding` / `--no-constant-folding` | 켜짐 | torch가 상수 서브그래프를 접어 weight 전치 등을 export 때 미리 계산합니다. **MLIR GEMM 경로(기본)는 폴딩된 번들에서 가장 빠르고**, hipBLASLt 경로는 폴딩하지 않은 번들에서 더 빠릅니다 |
| `--output DIR` | `data/benchmarks/pi05/libero` | 번들 위치 |
| `--checkpoint`, `--revision`, `--local-files-only`, `--data-root` | | 체크포인트와 데이터 위치 지정 |

내부 동작:

1. 레포의 `vai/utils/image-processor.cpp`, `tokenizer.cpp`로 임시 C++ 헬퍼를 빌드합니다. 이 헬퍼가 벤치와 똑같은 이미지, 토큰, noise를 만듭니다.
2. FP32 원본 `sample_actions`와 split(prefill + step×10)이 일치하는지 확인합니다 (`rtol 1e-5, atol 2e-5`).
3. 파라미터를 FP16으로 바꾸고 CPU에서 FP16 split을 한 번 돌려 FP32 대비 max abs 오차를 출력합니다.
4. ONNX opset 18로 export, 엄격한 ONNX 검사, 공통 부분식(CSE) 정리를 합니다.

생성 파일 (`data/benchmarks/pi05/libero/`):

| 파일 | 내용 |
|---|---|
| `prefill.fp16.onnx` + `prefill.fp16.weights.bin` | 이미지, 토큰 → `cache_mask`, `K0..K17`, `V0..V17` (FP16) |
| `step.fp16.onnx` + `step.fp16.weights.bin` | `x_t`, `timestep`, cache → `x_next` (Euler update 포함) |
| `metadata.json` | 패키지 버전, 프롬프트, `constant_folding`, FP16 CPU 오차, 그래프 검증 결과 |
| `config.txt` | `5 2 10 50 32 18 666` (번들 버전, 카메라, step, chunk, action dim, layer, prefix 길이). 마지막에 쓰이므로 이 파일이 있으면 export가 끝난 것입니다 |

- 같은 위치에 번들이 이미 있으면 덮어쓰지 않고 `FileExistsError`로 멈춥니다. 다시 export할 때는 기존 6개 파일을 지우거나 `--output`으로 다른 폴더를 지정하세요 (reference `.bin` 3개는 지울 필요 없습니다).
- 번들은 export할 때의 shape(카메라 2, chunk 50, step 10, 프롬프트 154 토큰)로 고정됩니다. 프롬프트나 워크로드가 바뀌면 다시 export해야 합니다.

## 7. 빌드

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVAI_BUILD_BENCH=ON \
      -DCMAKE_CXX_COMPILER=g++-14 \
      -DMIGRAPHX_GEMM_PROVIDER=mlir          # 기본값. hipblaslt 도 가능 (§9.5)
cmake --build build --target bench-pi05-vai bench-pi05-migraphx
```

configure 로그에서 다음 줄이 보여야 MIGraphX 벤치가 빌드됩니다.

```
-- [vAi-bench] MIGraphX found: ~/.cache/vai/migraphx-2.18.0/lib/cmake/migraphx (2.18), GEMM through mlir
-- [vAi-bench] bench-pi05-migraphx runs on the ROCm runtime in ~/.cache/vai/migraphx-2.18.0/lib
```

- `MIGraphX X.Y ... != pinned 2.18 (rocm-10.1): skipping bench-pi05-migraphx`: 시스템 ROCm의 다른 MIGraphX를 찾은 경우입니다. `setup-migraphx.sh`를 실행하거나 `-Dmigraphx_DIR`을 지정하세요.
- `MIGraphX or HIP not found`: 2.18이 설치되지 않았거나 HIP(`/opt/rocm` 또는 `$ROCM_PATH`)가 없는 경우입니다.
- 결과물: `build/bin/bench-pi05-vai`, `build/bin/bench-pi05-migraphx`. MIGraphX를 나중에 설치했다면 configure를 다시 하세요.

## 8. 실행

### 8.1 권장: `run-amd.sh`

[`scripts/run-amd.sh`](../scripts/run-amd.sh)는 `bench-pi05-vai --workload migraphx`와 `bench-pi05-migraphx`를 차례로 실행합니다.

```bash
./benchmarks/pi05/scripts/run-amd.sh
./benchmarks/pi05/scripts/run-amd.sh -- --atol 0.2 --rtol 0.1   # -- 뒤의 인자는 두 벤치에 모두 전달 (MIGraphX는 기본 허용 오차를 넘음, §9.6)
BUILD_DIR=build-other AMD_CARD=/sys/class/drm/card1 ./benchmarks/pi05/scripts/run-amd.sh
```

- amdgpu 카드의 `power_dpm_force_performance_level`을 `high`로 고정합니다 (필요할 때만 `sudo`). 스크립트가 어떻게 끝나든 원래 값으로 되돌립니다.
- APU는 CPU와 전력을 나눠 쓰므로, 벤치가 도는 동안 sclk와 전력(hwmon)을 0.2초마다 샘플링하고, GPU busy가 50% 이상인 구간의 범위를 출력합니다 (`clock under load: sclk ...-... MHz, power mean ... W`).
- vAi 벤치는 Vulkan 장치가 여러 개면(RADV + llvmpipe) stdin으로 하나를 고르라고 묻습니다. 스크립트는 stdin을 닫고 실행하므로 첫 번째 장치(RADV GPU)가 선택됩니다.

### 8.2 수동 실행

```bash
# GPU 클럭 고정 (benchmarks/SETUP.md#gpu-clock)
echo high | sudo tee /sys/class/drm/card<N>/device/power_dpm_force_performance_level

build/bin/bench-pi05-vai --workload migraphx     # 장치 선택 프롬프트에서 RADV(0) 선택, 또는 </dev/null
build/bin/bench-pi05-migraphx                    # 기본 번들: data/benchmarks/pi05/libero
build/bin/bench-pi05-migraphx /path/to/bundle    # 다른 번들

echo auto | sudo tee /sys/class/drm/card<N>/device/power_dpm_force_performance_level
```

### 8.3 옵션

| 옵션 | vAi | MIGraphX | 설명 |
|---|:-:|:-:|---|
| `--workload <name>` | ✓ | | `migraphx`(기본), `trt`, `vlacpp`. AMD 비교에는 `migraphx` |
| `[bundle-dir]` | | ✓ | 다른 ONNX 번들 폴더 |
| `--atol N`, `--rtol N` | ✓ | ✓ | reference 비교 허용 오차 (§9.6) |
| `--showActions` | ✓ | ✓ | 표 다음에 액션 앞 5개(8 dim)를 출력 |
| `--help` | ✓ | ✓ | |

### 8.4 출력 읽기

순서대로 다음 줄이 나옵니다. 숫자는 실행할 때마다 다릅니다.

- `device:`, `model:`: 사용한 GPU, 런타임 버전 (MIGraphX는 `MIGraphX 2.18, HIP <ver>`)
- (MIGraphX) `precision:`, `bundle: ... (constants folded)`, `compile: N s`, `gemm: MIGRAPHX_DISABLE_MLIR=..., MIGRAPHX_SET_GEMM_PROVIDER=...`
- `workload: migraphx (pi0.5 libero, lerobot policy)`, `input: 2 cameras x 224x224, 154 prompt tokens, ...`
- `images:`, `noise:`: reference.py 파일을 읽었는지 (`preprocessed here` 또는 `sampled`가 나오면 파일이 없는 것)
- `accuracy: pass|FAIL, max abs error ..., K of 1600 values outside atol .. + rtol ..`
- 결과 표: `image + text | framework | latency (ms) mean ± SD | min (ms)`

결과를 기록할 때는 [../README.md#results](../README.md#results)의 표 형식(OS, Vulkan 드라이버, ROCm 버전, GPU 클럭, vAi 커밋)을 따릅니다.

## 9. 정밀도

기준은 **vAi의 정밀도가 MIGraphX보다 높아야 한다**는 것입니다 (§10: max abs error vAi 0.00116, MIGraphX 0.00221). 아래 내용은 2026-11-18 기준이고, vAi는 커밋 `8552ebe`의 타워별 정밀도(`TowerPrecision`)를 따릅니다.

- **vAi:** 코드와 셰이더를 직접 읽어 확인했습니다. 장치 조건(device ID 0x1586, CU 40개, subgroup 64, `VK_KHR_cooperative_matrix`, `shaderFloat16`)은 `vulkaninfo`와 `rocminfo`로 확인했습니다.
- **MIGraphX:** export한 ONNX 그래프를 shape inference로 집계했습니다. 컴파일 결과는 `migraphx-driver compile --gpu`, `MIGRAPHX_TRACE_COMPILE=1`(pass별 IR), `MIGRAPHX_MLIR_DUMP`(MLIR 모듈)로 확인했습니다.

### 9.1 요약

| | reference.py | vAi | MIGraphX |
|---|---|---|---|
| 가중치 저장 | FP32 | FP16 (norm 계수, condition table, RoPE 주파수는 FP32) | FP16 (RMSNorm 계수 `1+w`만 FP32 상수) |
| 블록 branch 활성값 (norm, Q/K/V, RoPE, attention, FFN hidden) | FP32 | **FP16** (`branchFp16`, 세 타워 모두) | FP16 |
| residual stream | FP32 | vision, language **FP32**, action expert FP16 (`residualFp16`) | FP16 |
| 그 밖의 활성값 (블록 밖 norm, projector, velocity) | FP32 | FP32 (`oDefaultDtype`) | FP16 |
| GEMM, attention의 행렬곱 입력 | FP32 | FP16 (FP32 입력은 커널 안에서 FP16으로 반올림) | FP16 |
| GEMM 누산 | FP32 | FP32 (coopmat accumulator) | 확인 불가 (rocMLIR 내부) |
| GEMM 후처리 (bias, residual, GELU) | FP32 | FP32로 계산한 뒤 출력 dtype으로 저장 | FP16 |
| softmax | FP32 | FP32, P는 FP16 | FP32 (MIGraphX가 변환), P는 FP16 |
| norm (LayerNorm, RMSNorm) | FP32 | 통계 FP32, 블록 안 norm 출력 FP16 | 통계는 FP32, 출력은 FP16 |
| K/V cache | FP32 | FP16 | FP16 |
| timestep, condition | FP32 | 호스트에서 double/FP32로 미리 계산 | timestep을 FP16으로 반올림한 뒤 GPU에서 계산 |
| velocity, Euler update | FP32 | FP32 | velocity는 FP16, update는 FP32 |
| 실행 장치 | CPU (CPU 빌드 torch) | Vulkan (RADV) | HIP (ROCm 10.1) |

두 벤치 모두 reference.py가 쓴 `images_*.bin`, `noise_*.bin`(FP32)을 입력으로 받고, 액션을 FP32로 돌려줍니다.

### 9.2 vAi 상세

설정은 [`vai/bench.cpp`](../vai/bench.cpp)의 `wDefaultDtype = Fp16`, `oDefaultDtype = Fp32`, `RuntimeConfig::kvCacheDtype = Fp16`(기본값, [`production/pi05/net.h:75`](../../../models/production/pi05/net.h#L75)), 그리고 [`experimental/pi05/net.h`](../../../models/experimental/pi05/net.h)의 타워별 `TowerPrecision`입니다.

| 타워 | `branchFp16` | `residualFp16` | `accF32` |
|---|---|---|---|
| vision (SigLIP) | on | off | on (AMD에서는 효과 없음) |
| language (Gemma prefill) | on | off | off |
| action expert | on | **on** | off |

- `branchFp16`: 블록 안에서 residual stream에서 갈라져 나가는 branch(norm 2개, Q/K/V, RoPE, attention, FFN hidden과 게이팅)의 출력이 FP16입니다.
- `residualFp16`: residual stream(residual을 더하는 출력 projection, action expert의 gated residual)과 타워 입력(`actionProj`)도 FP16입니다.
- `accF32`: Linear 누산을 FP32로 바꿉니다. AMD는 원래 FP32로 누산하므로 이 벤치에서는 바뀌는 것이 없습니다.

**공통 규칙**

- **노드 출력:** 위 스위치가 FP16으로 정한 노드 외에는 모두 `oDefaultDtype`(FP32)을 씁니다 (예: [`linear.h:2293`](../../../vai/nodes/linear.h#L2293)). 블록 밖의 노드(patch embedding, post_layernorm, projector, token embedding, 최종 norm, velocityHead)는 FP32입니다. Linear split-K partial은 출력 dtype으로 저장되고, attention split-K partial은 FP32입니다.
- **가중치 변환:** 로드할 때 호스트에서 FP32 → FP16으로 바꿉니다 ([`tensor.cpp:46-69`](../../../vai/engine/tensor.cpp#L46-L69)). round-to-nearest-even이고, **FP16 subnormal(|w| < 2⁻¹⁴ ≈ 6.1e-5)은 0으로 flush**됩니다 (`tensor.cpp:66`). torch의 `.to(float16)`는 subnormal을 남기므로 MIGraphX 번들과 이 부분이 다릅니다. GPU 업로드는 변환 없는 memcpy입니다.
- **GEMM과 attention 입력:** FP16 활성값은 그대로, FP32 활성값은 커널 안에서 읽을 때 FP16으로 반올림해서 cooperative matrix(f16×f16)에 넣습니다. 누산은 FP32입니다. 그 외 연산(norm, RoPE, GELU, gating, embedding scale, bias, residual, Euler)은 FP32로 계산한 뒤 출력 dtype으로 저장합니다.

**가중치 저장 dtype** ([`experimental/pi05/lerobot-weights.h`](../../../models/experimental/pi05/lerobot-weights.h))

| 가중치 | 저장 | 비고 |
|---|---|---|
| 모든 Linear weight (SigLIP q/k/v/o/fc1/fc2, projector, Gemma q/k/v/o/gate+up/down, actionProj, velocityHead) | FP16 | `loadLin`, `loadLinNoBias` (136-146) |
| Linear bias (SigLIP, projector, `action_in_proj`, `action_out_proj`; Gemma는 bias 없음) | FP16 | 읽을 때 FP32로 넓혀서 FP32 후처리에 더함 |
| patch embedding 커널 [1152, 3·14·14] | FP16 | Conv가 아니라 patchify(Slice/Reshape/Permute) + Linear |
| patch bias + position embedding [256, 1152] | FP16 | FP32로 더한 뒤 한 번만 FP16으로 반올림 |
| token embedding [257152, 2048] | FP16 | 읽은 뒤 FP32로 `×√2048` |
| SigLIP LayerNorm scale, shift | **FP32** | `normDt = Fp32` (137) |
| Gemma RMSNorm scale | **FP32** | `+1`을 호스트에서 FP32로 더해 둠 (153-161) |
| condition table [10 step, 110 행, 1024] | **FP32** | 아래 action decoder 참고 |
| RoPE 주파수 | **FP32** | 노드가 호스트에서 생성 ([`rope.h`](../../../vai/nodes/rope.h)) |

**단계별** (출력 열이 GPU 메모리에 저장되는 dtype, "→FP16"은 FP32 입력을 커널이 읽을 때 반올림한다는 뜻)

| 단계 | 연산 | 입력 | 계산 / 누산 | 출력 |
|---|---|---|---|---|
| Vision (SigLIP 27 block, 2 카메라 × 256 patch) | patchify | FP32 이미지 | 복사 | FP32 |
| | patch Linear + bias(position 포함) | FP32 →FP16 | f16 coopmat, FP32 누산, bias FP32 | FP32 |
| | LayerNorm (block마다 2개) | FP32 | 평균/분산 FP32 | **FP16** |
| | q/k/v Linear | FP16 | FP32 누산, bias FP32 | **FP16** |
| | attention (head dim 72) | Q/K/V FP16 | 아래 attention 참고 | **FP16** |
| | o Linear + residual | FP16, residual FP32 | FP32 누산, residual FP32 | FP32 |
| | fc1 + GELU(tanh) | FP16 | FP32 누산, GELU FP32 | **FP16** |
| | fc2 + residual | FP16, residual FP32 | FP32 누산 (split-K partial FP32, reduce FP32) | FP32 |
| | post_layernorm | FP32 | 평균/분산 FP32 | FP32 |
| Projector, prompt | projector Linear | FP32 →FP16 | FP32 누산, bias FP32 | FP32 |
| | token embedding | 정수 id | FP16 표를 FP32로 읽고 `×√2048` (FP32) | FP32 |
| | 카메라 + prompt concat | FP32 | 복사 | FP32 |
| Prefill (Gemma 2B 18 block, 666 토큰) | RMSNorm | FP32 | 제곱합, `inversesqrt` FP32 | **FP16** |
| | q/k/v Linear | FP16 | FP32 누산 | **FP16** |
| | RoPE (Q, K) | FP16 | 각도와 sin/cos, 회전 FP32 | **FP16** |
| | attention + KV cache 쓰기 | Q/K/V FP16 | K/V를 FP16 그대로 cache에 append | **FP16** |
| | o Linear + residual | FP16, residual FP32 | FP32 누산, residual FP32 | FP32 |
| | gate+up Linear, GELU 게이팅 | FP16 | FP32 누산, 게이팅 FP32 | **FP16** |
| | down Linear + residual | FP16, residual FP32 | FP32 누산, residual FP32 | FP32 |
| Denoise step (action expert 18 block, 50 행, 10회) | actionProj Linear | `x_t` FP32 →FP16 | FP32 누산, bias FP32 | **FP16** |
| | condition (adaRMS scale/shift/gate) | — | step별 FP32 행을 table에서 복사 | FP32 |
| | adaRMS norm | FP16 | FP32 | **FP16** |
| | q/k/v/o, gate+up, down Linear | FP16 | FP32 누산, 게이팅 FP32 | **FP16** |
| | attention (prefix cache + 자기 50행) | Q FP16, prefix K/V는 FP16 cache, 자기 K/V FP16 | 아래 attention 참고 | **FP16** |
| | gated residual (`g·branch + x`) | g FP32, branch, x FP16 | FP32 `fma` | **FP16** |
| | 최종 adaRMS norm | FP16 | FP32 | FP32 |
| Velocity, Euler | velocityHead Linear (residual = `x_t`, scale = −1/10) | FP32 →FP16 | FP32 누산, split-K reduce에서 `−0.1·(Σ + bias) + x_t`를 FP32로 | FP32 (같은 텐서에 덮어씀) |

**condition table.** condition은 step에만 의존하므로 로드할 때 호스트에서 10 step 전부 미리 계산합니다 ([`lerobot-weights.h:49-127`](../../../models/experimental/pi05/lerobot-weights.h#L49-L127)). t = 1 − k/10과 sinusoid는 double로 계산한 뒤 FP32로 저장하고, time MLP와 adaRMS dense 37개는 FP32(누산 FP32)로 계산합니다. GPU는 step마다 FP32 행을 복사만 합니다.

**KV cache.** 레이어마다 `K_cache`, `V_cache`가 [1, 1018, 256] FP16입니다 ([`attention.h:974-975`](../../../vai/nodes/attention.h#L974-L975)). 1018은 카메라 3 × 256 + prompt 200 + chunk 50입니다. attention 커널이 K/V를 `f16vec4(v)`로 써서 append합니다 ([`attention-fa-cm.comp:183-184`](../../../vai/shaders/attention-fa-cm.comp#L183-L184)). `branchFp16`로 K/V가 이미 FP16이라 이때 새로 반올림되는 값은 없습니다. prefill에서 prefix는 자기 K/V의 FP16 사본을 보고, denoise step은 prefix를 FP16 cache에서 직접 읽습니다. 행렬곱에 들어가는 K/V는 모두 FP16입니다.

**Linear 커널.** Strix Halo(0x1586)에서는 `linear-cm-amd` 셰이더가 선택됩니다. 입력, 가중치, 출력이 모두 FP16인 Linear(세 타워의 q/k/v, vision fc1, Gemma와 action expert의 gate+up, action expert의 o/down)는 FP16 I/O용 타일(`fp16Io`, [`linear.h:2454`](../../../vai/nodes/linear.h#L2454))을 고를 수 있습니다. residual을 FP32로 더하는 vision, language의 o/fc2/down과 블록 밖 Linear는 FP32 입력이나 출력이 있어 이 경로를 쓰지 않습니다.

- AMD에서는 `setAccDtype`으로 따로 정하지 않으면 누산 dtype이 FP32입니다: `accDtype = ... (amd ? Dtype::Fp32 : wDtype)` ([`linear.h:2320`](../../../vai/nodes/linear.h#L2320)). 이 벤치는 FP16 누산을 쓰지 않습니다.
- 입력 반올림: FP32 입력이면 `readI8`, `readI`가 `f16mat2x4` / `float16_t`로 바꿉니다 ([`linear-cm-amd.comp:167-169`](../../../vai/shaders/linear-cm-amd.comp#L167-L169)).
- 행렬 타입: A, B는 `coopmat<float16_t>`, accumulator는 `coopmat<float>`입니다 ([`linear-cm-amd.comp:126-128`](../../../vai/shaders/linear-cm-amd.comp#L126-L128)).
- 출력: FP32 accumulator에 bias, residual, GELU, scale을 FP32로 적용한 뒤 출력 dtype으로 저장합니다. split-K partial은 출력 dtype으로 저장되고 ([`linear.h:2698`](../../../vai/nodes/linear.h#L2698)), `linear-reduce`가 FP32로 읽어 합칩니다.
- coopmat이 없는 장치의 fallback(`linear-cm`, `linear-tile`)도 FP16 입력에 FP32 누산이라 정밀도 등급은 같습니다.

**Attention 커널** (`attention-fa-cm`, vision·prefix·action 공통)

- Q를 FP16으로 반올림해 shared memory에 넣습니다.
- S = K·Qᵀ는 f16 coopmat, **FP32 누산**입니다 (`// S is fp32 fixed`, [`attention-fa-cm.comp:252`](../../../vai/shaders/attention-fa-cm.comp#L252)).
- softmax는 FP32, log2 단위입니다. scale `1/√d`도 FP32입니다 (761). row max, 보정 계수 `exp2(m_prev − m)`, 합 `l`도 모두 FP32입니다.
- **P는 FP16으로 반올림**해서 PV 행렬곱에 넣습니다 (`s_p[...] = f16vec4(p4)`, 805). max에 +3 offset을 주기 때문에 P ≤ 1/8입니다.
- PV는 f16 coopmat, FP32 누산입니다. 출력 `acc · (1/l)`은 FP32이고, split-K partial과 reduce도 FP32입니다.

### 9.3 MIGraphX 상세

[`export-onnx.py`](../scripts/export-onnx.py)는 FP32에서 split이 맞는지 확인한 뒤 모든 파라미터를 `.to(float16)`으로 바꾸고, 상수 폴딩한 FP16 그래프를 opset 18로 export합니다. MIGraphX는 이 그래프를 그대로 컴파일합니다(`quantize_fp16` 등 추가 변환 없음). 그래프에 적힌 dtype과 컴파일 후 실제 커널이 다른 곳이 있어서 둘을 나눠 적었습니다.

**저장**

- prefill 가중치 FP16 28.1억 개(5.6 GB), step 가중치 FP16 4.3억 개(0.86 GB).
- FP32 상수는 prefill의 Gemma RMSNorm 계수 35개 [2048]뿐입니다. 폴딩 때 FP16 파라미터를 FP32로 바꿔 `1+w`를 계산한 값이라, 값 자체는 FP16으로 반올림된 상태입니다.
- 번들 입출력: `images`, `x_t`, `timestep`, 출력 `x_next`는 FP32이고, K/V cache(`K0..V17`)는 FP16입니다.

**단계별**

| 단계 | 연산 | ONNX 그래프 | 컴파일 후 (MIGraphX 2.18, MLIR) |
|---|---|---|---|
| 입력 | images | FP32 → Cast FP16 | `convert_kernel` FP32 → FP16 |
| Vision (SigLIP 27 block × 카메라 2개, 카메라별로 따로 계산) | patch embedding | Conv FP16 | `mlir_slice_convolution_transpose` FP16 |
| | position embedding | Add FP16 | FP16 |
| | LayerNorm | `LayerNormalization` FP16 | `..._convert_mul_reduce_sum_..._rsqrt_..._convert_kernel`: FP32로 바꿔 평균/분산을 구하고 FP16으로 출력 |
| | q/k/v/o, fc1, fc2 | MatMul + Add FP16 | `mlir_*dot*` FP16 입출력, bias FP16 |
| | GELU(tanh) | Tanh 등 FP16 | fc1 MLIR 커널의 epilogue로 융합, FP16 |
| | attention scale + softmax | Mul, **Softmax FP16** | `convert_mul_reduce_max_sub_exp_reduce_sum_div_convert_kernel`: **FP32** 계산, P는 FP16 |
| | residual | Add FP16 | FP16 |
| Projector, prompt | projector | MatMul + Add FP16 | MLIR FP16 |
| | token embedding × √2048 | Gather FP16, Mul FP16 | FP16 |
| Prefill (Gemma 2B 18 block) | RMSNorm | Cast FP32 → x², mean, +eps, sqrt, div, `×(1+w)` FP32 → Cast FP16 | `convert_mul_mul_reduce_sum_add_sqrt_div_mul_convert_kernel`, FP32 계산, FP16 출력 |
| | q/k/v/o, gate+up, down | MatMul FP16 | MLIR FP16 |
| | RoPE | inv_freq·position, cos/sin **FP32** → Cast FP16, 적용(`q·cos + rot(q)·sin`) **FP16** | `add_convert_mul_cos_sin_kernel` FP32, 적용 커널 FP16 |
| | attention mask | Where FP16 (FP16 최저값) | FP32로 변환해 softmax 커널에서 더함 |
| | attention softmax | Softmax FP16 | FP32 계산, P FP16 |
| | GELU 게이팅, residual | FP16 | FP16 |
| | K/V cache 출력 | FP16 | FP16 |
| Denoise step (action expert 18 block, 10회) | `x_t` | FP32 → Cast FP16 | MLIR 커널 안에서 convert |
| | timestep | FP32 → **Cast FP16** → Cast f64 → sin/cos f64 → Cast FP16 | timestep은 FP16으로 반올림되고, sin/cos는 `eliminate_data_type` pass가 f64를 FP32로 낮춰 FP32로 계산 |
| | time MLP (Gemm, swish) | FP16 | MLIR FP16 (매 step GPU에서 계산) |
| | adaRMS | dense Gemm FP16 → norm 통계 FP32 → `x̂·(1+scale)+shift` FP32 → Cast FP16 | 통계, modulation FP32, 출력 FP16 |
| | gated residual | Mul + Add FP16 | o / down projection MLIR 커널(`mlir_dot_slice_unsqueeze_slice_mul_add`)의 epilogue로 융합, FP16 |
| | attention (prefix cache + 50행) | QKᵀ FP16, Softmax FP16, PV FP16 | QKᵀ, PV MLIR FP16, softmax FP32 |
| Velocity, Euler | action_out_proj | MatMul + Add **FP16** | `mlir_dot_add_convert_mul_add`: dot + bias FP16 |
| | Euler update | Cast FP32, `x_t + (−0.1)·v` **FP32** | 같은 MLIR 커널 안에서 FP32로 convert한 뒤 mul, add |

**softmax가 FP32가 되는 이유.** 모델 코드는 `softmax(..., dtype=float32)`를 요청합니다. 그런데 torch의 ONNX exporter는 Softmax를 FP16 입력에 그대로 적용하고, FP32 Cast를 그 *뒤*에 붙입니다. 그래서 그래프만 보면 softmax가 FP16이고, 뒤에 의미 없는 FP16→FP32→FP16 왕복 Cast가 남습니다.

MIGraphX에서는 이렇게 처리됩니다.
1. `eliminate_convert` 등이 이 왕복 Cast를 지웁니다.
2. `rewrite_reduce` pass가 FP16 softmax를 FP32 연산(scale, mask, max, sub, exp, sum, div)으로 풀고, PV dot 앞에서 FP16으로 되돌립니다. pass별 IR에서 확인했습니다.

[`export-onnx.py`](../scripts/export-onnx.py) 주석의 "softmax는 FP32로 남는다"는 그래프 기준으로는 맞지 않고, 실제 실행 기준으로는 MIGraphX 덕분에 맞는 셈입니다.

**GEMM 누산.** MLIR 커널의 입력과 출력은 FP16입니다. 덤프한 모듈에서도 `migraphx.dot ... f16 x f16 -> f16`로 나옵니다. 내부 누산 dtype은 rocMLIR가 정하고 덤프에 드러나지 않아서 **확인하지 못했습니다**. hipBLASLt 경로(`MIGRAPHX_DISABLE_MLIR=1`)도 마찬가지입니다.

### 9.4 vAi와 MIGraphX의 정밀도 차이

같은 FP16 가중치와 FP16 K/V cache를 쓰면서도 vAi의 오차가 더 작은 것은 아래 지점들 때문입니다.

| 지점 | vAi | MIGraphX |
|---|---|---|
| residual stream | vision, language FP32, action expert FP16 | FP16. 블록마다 residual add가 FP16으로 반올림됨 |
| 블록 밖 활성값 (patch embedding, post_layernorm, projector, prompt embedding, 최종 norm) | FP32 | FP16 |
| GEMM 후처리 (bias, residual, GELU) | FP32 accumulator에 FP32로 적용한 뒤 출력 dtype으로 한 번 반올림 | FP16 출력, FP16 후처리 |
| RoPE 적용 | sin/cos, 곱셈 모두 FP32, 결과만 FP16으로 저장 | sin/cos를 FP16으로 반올림한 뒤 FP16으로 곱함 |
| timestep | double로 계산 (t = 1, 0.9, …, 0.1 그대로) | FP16으로 반올림 (예: 0.9 → 0.89990, 0.1 → 0.09998) |
| condition (time MLP, adaRMS dense) | 호스트 FP32 | GPU FP16 GEMM |
| velocity | FP32 accumulator에서 FP32로 Euler까지 | FP16으로 반올림한 뒤 FP32 Euler |
| 가중치 FP16 subnormal | 0으로 flush | 유지 |

두 엔진이 같은 부분도 있습니다. GEMM 입력과 attention의 Q, K, V, P는 둘 다 FP16이고, K/V cache도 둘 다 FP16입니다. 블록 안 branch(norm 출력, Q/K/V, attention 출력, FFN hidden)도 둘 다 FP16으로 저장합니다. softmax와 norm 통계는 둘 다 FP32로 계산합니다.

**확인하지 못한 것**
- MIGraphX MLIR / hipBLASLt GEMM의 내부 누산 dtype
- vAi 셰이더에서 GPU가 FP32 → FP16 변환에 쓰는 반올림 방식. float-controls를 지정하지 않아 드라이버가 정합니다(RADV/ACO는 RTE로 예상). 호스트 변환만 RTE + subnormal flush로 확인했습니다.
- 큰 인자에서 RADV의 `sin`/`cos`/`tanh`/`exp2` 정확도. RoPE 각도는 최대 약 700 rad입니다 (position ≤ 715, 주파수 ≤ 1).

### 9.5 MIGraphX GEMM 경로

GEMM 백엔드는 configure할 때 `-DMIGRAPHX_GEMM_PROVIDER`로 정합니다 ([`migraphx/net.h`](../migraphx/net.h)의 `compileWithFlags`).

| `MIGRAPHX_GEMM_PROVIDER` | 설정되는 환경 변수 | 맞는 번들 |
|---|---|---|
| `mlir` (기본) | `MIGRAPHX_DISABLE_MLIR=0`, `MIGRAPHX_SET_GEMM_PROVIDER=hipblaslt` | 상수 폴딩된 번들 (export 기본값) |
| `hipblaslt` | `MIGRAPHX_DISABLE_MLIR=1`, `MIGRAPHX_SET_GEMM_PROVIDER=hipblaslt` | `--no-constant-folding` 번들 |

- 두 경우 모두 MLIR에 들어가지 않은 dot은 hipBLASLt로 갑니다.
- 환경 변수를 직접 주면 그 값이 우선합니다. 다시 빌드하지 않고 바꿔 보려면:
  `MIGRAPHX_DISABLE_MLIR=1 build/bin/bench-pi05-migraphx <unfolded-bundle>`
- 벤치는 번들의 `metadata.json`에서 폴딩 여부를 읽어 출력하고, 폴딩되지 않은 번들이면 MLIR에서 느리다는 `note:`를 출력합니다.
- GEMM 경로끼리 비교할 때는 **같은 번들**로 하세요. 번들의 폴딩 여부에 따라 어느 경로가 빠른지가 바뀝니다.

### 9.6 정확도 검사 (`--atol`, `--rtol`)

시간을 재기 전에 1회 예측해서 `reference_2_50_10.bin`의 1,600개 값과 `numpy.allclose` 방식으로 비교합니다.

```
|out - ref| <= atol + rtol * |ref|     (모든 값, NaN은 실패)
```

기본값은 `atol 0.002`, `rtol 0.001`입니다 ([`bench-utils.h`](../../common/bench-utils.h)의 `VLABenchOptions`). MIGraphX는 이 기준을 넘는 값이 있어(§10) 지연 시간을 출력하지 않으므로, 결과표의 두 벤치는 `run-amd.sh -- --atol 0.2 --rtol 0.1`로 잽니다. 결과를 기록할 때는 사용한 허용 오차를 함께 적으세요. `max abs error`는 허용 오차와 관계없이 항상 출력되므로 프레임워크 간 정확도 비교에 쓸 수 있습니다.

## 10. 참고 측정값 (2026-11-18)

공식 결과표는 [../README.md#results](../README.md#results)에 있습니다. 아래는 같은 날 측정한 값입니다.

Radeon 8060S (gfx1151), 클럭 `high`, 단위 ms. vAi 정밀도별 비교 (`TowerPrecision`은 커밋 `8552ebe`, 활성값 FP32는 `e979c71`, 활성값 FP16은 그 부모 `7438add`):

| vAi 설정 | latency (ms) | max abs error | strict (`--atol 0.002 --rtol 0.001`) |
|---|---:|---:|---|
| 가중치 FP16, **`TowerPrecision`** (현재) | 151.73 ± 3.98 (`run-amd.sh` 3회 중앙값, 범위 149.3 – 151.8) | **0.00116** | pass (0 / 1600) |
| 가중치 FP16, 활성값 FP32 | 182.45 ± 4.95 (`run-amd.sh` 3회 중앙값, 범위 179.0 – 183.3) | 0.000981 | pass (0 / 1600) |
| 가중치 FP16, 활성값 FP16 | 149.08 ± 0.50 (`run-amd.sh` 3회 중앙값, 범위 148.5 – 150.5) | 0.00239 | 측정 안 함 |
| MIGraphX (FP16 번들, MLIR, 폴딩) | 195.33 ± 4.17 (`run-amd.sh` 3회 중앙값, 범위 194.5 – 196.3) | 0.00221 (이전 측정 0.00221 – 0.00225) | **FAIL** (1 / 1600, 이전 측정 2 / 1600) |

`TowerPrecision`과 활성값 FP32를 같은 세션에서 vAi 벤치만 3회씩 돌려 비교하면 149.05 – 150.35 ms(0.00116)와 178.24 – 182.41 ms(0.000981)입니다. 정밀도 변경으로 32 ms가 줄고 오차는 18% 늘었지만, 여전히 MIGraphX의 절반 정도입니다.

MIGraphX 런타임, 번들, GEMM 경로별 비교 (LIBERO, 카메라 2, 154 토큰, chunk 50, 10 step, FP16 번들 v5, 단위 ms, mean):

| MIGraphX 런타임 | 번들 | hipBLASLt | MLIR |
|---|---|---:|---:|
| 2.18 on 시스템 ROCm 7.2.1 | (이전 번들) | ~287 | ~317 |
| 2.18 on ROCm 10.1 | 폴딩 안 함 | 208.7 | 293.7 |
| 2.18 on ROCm 10.1 | **상수 폴딩 (기본)** | 236.4 | **194.3** |

이 표를 측정할 때 vAi(활성값 FP16)는 147.9 ms, MIGraphX(MLIR + 폴딩)는 194.5 ms였습니다 (커밋 `7438add`, master의 AMD Linear 최적화 `706943a`, `a7819ed` 병합 이후).

## 11. 문제 해결

| 증상 | 원인과 조치 |
|---|---|
| `RuntimeError: lerobot 0.6.0 required; found ...` | export-onnx.py 버전 검사. §3.2 버전으로 venv 맞추기 |
| `C++ compiler 'c++' failed the C++23 explicit object parameter probe` | `--cxx g++-14` |
| `FileExistsError: an ONNX bundle already exists` | 기존 번들 6개 파일 삭제 또는 `--output NEW_DIR` |
| reference.py에서 Hugging Face 401/403 | `google/paligemma-3b-pt-224` 라이선스 동의, `hf auth login` |
| `error: ... holds N bytes, expected M` | 다른 워크로드에서 만든 reference 파일. §4 명령으로 다시 생성 |
| `accuracy: unverified, no .../reference_2_50_10.bin` | reference.py를 실행하지 않음. 이때는 지연 시간만 나오고 정확도는 검증되지 않음 |
| `error: reference mismatch; latency results suppressed` | 허용 오차 초과. `max abs error`와 사용한 `--atol/--rtol` 확인 |
| `v1..v4 pi05 bundle (FP32 graph) is unsupported` | 예전 FP32 번들. export-onnx.py로 다시 export |
| `the prompt has N tokens, the bundle was exported for M` | 프롬프트(task, state, 토크나이저)가 export 때와 다름. 다시 export |
| CMake: `skipping bench-pi05-migraphx` | §7 참고. MIGraphX 2.18(ROCm 10.1) 설치와 registry 확인 |
| MIGraphX가 예상보다 느림 | `ldd`로 ROCm 10.1 라이브러리를 읽는지, `bundle: (constants folded)`와 `gemm:` 줄이 기대한 조합인지, `clock under load` 범위 확인 |
