# 기여 절차와 자동 검사

이 문서는 수정한 코드가 `master`에 들어가기까지의 절차를 설명합니다.
처음 한 번 준비할 것, 커밋과 PR에서 실행되는 검사, 검사가 실패했을 때 고치는 방법을 순서대로 다룹니다.

코드 자체의 작성 방법은 [커스텀 모델 작성](05-contributing-models.md), [커스텀 노드와 커널 작성](07-contributing-nodes.md), [커스텀 그룹 작성](08-contributing-groups.md)을 참조하면 됩니다.

## 1. 처음 한 번 준비하기

### 1.1 빌드와 커밋 훅

저장소는 `--recursive`로 받고, [빠른 실행](02-quickstart.md)의 방법으로 빌드합니다.
그다음 커밋 훅을 설치합니다. 훅은 커밋할 때 바뀐 부분을 검사합니다.

```bash
pip install pre-commit
pre-commit install
```

### 1.2 Python 패키지와 테스트

Python 바인딩이나 `Node`를 수정한다면 패키지를 설치하고 테스트를 실행합니다.
`torch`는 테스트의 기준 구현으로만 사용합니다. CPU 빌드를 먼저 설치하지 않으면 수 GB의 CUDA 빌드를 받게 됩니다.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test]"
python -m pytest python/tests
```

### 1.3 clang-tidy 훅 켜기

C++ 코드를 수정한다면 clang-tidy 훅도 켭니다. 이 훅은 켜기 전까지 아무 동작도 하지 않습니다.
clang-tidy는 컴파일 정보(`compile_commands.json`)가 있어야 하는데, Visual Studio 빌드에는 이 파일이 없기 때문입니다.

```bash
python .github/scripts/clang-tidy-staged.py --setup
```

이 명령은 `build-tidy/`에 Ninja로 CMake 설정만 실행하고, 빌드는 하지 않습니다.
Windows에서는 C++ 도구가 설치된 최신 Visual Studio를 찾아 그 환경에서 실행하므로, 개발자 명령 프롬프트를 열 필요가 없습니다.
훅을 끄려면 `build-tidy/` 폴더를 지우면 됩니다.

## 2. 커밋할 때 실행되는 검사

| 훅 | 검사 범위 | 문제가 있을 때 |
|---|---|---|
| clang-format | 스테이징한 C++ 줄 | 해당 줄을 고치고 커밋을 중단합니다. 고쳐진 내용을 다시 스테이징해 커밋합니다. |
| ruff format | 스테이징한 Python 파일 | 위와 같습니다. |
| ruff check | 스테이징한 Python 파일 | 문제를 알리고 커밋을 중단합니다. |
| clang-tidy | `vai/`, `models/`, `examples/`에서 스테이징한 C++ 줄 | 문제를 알리고 커밋을 중단합니다. [1.3](#13-clang-tidy-훅-켜기)을 실행하기 전에는 동작하지 않습니다. |

훅은 문제를 일찍 알기 위한 장치입니다. PR에서 같은 검사를 다시 실행하므로, 훅을 설치하지 않아도 결과는 같고 알게 되는 시점만 늦어집니다.

## 3. PR에서 실행되는 검사

### 3.1 검사 목록

| 검사 | 실행 조건 | 실패하는 경우 |
|---|---|---|
| `format` | 모든 PR | PR이 바꾼 C++ 줄이 clang-format과 다릅니다. PR이 바꾼 Python 파일이 ruff format과 다릅니다. 저장소 전체에서 ruff check가 문제를 찾았습니다. |
| `tidy` | `python/`, `docs/`, `*.md` 밖의 파일을 바꾼 PR | `vai/`, `models/`, `examples/`에서 PR이 바꾼 C++ 줄에 clang-tidy가 문제를 찾았습니다. |
| `vai-linux`, `vai-windows` | `python/`, `docs/`, `*.md` 밖의 파일을 바꾼 PR | C++ 빌드가 실패했습니다. (GCC 14, MSVC.) |
| `pyvai-linux` | `docs/`, `*.md` 밖의 파일을 바꾼 PR | Python 패키지 빌드가 실패했거나 `python/tests`가 실패했습니다. |
| `docs` | `docs/`, `mkdocs.yml`, `vai/nodes/`를 바꾼 PR | [제공 노드들](06-nodes.md)이 `Node` 헤더와 다르거나, 문서 사이트 빌드가 실패했습니다. |

### 3.2 바꾼 줄만 검사

C++ 검사는 파일 전체가 아니라 PR이 추가하거나 바꾼 줄만 확인합니다.
저장소의 코드가 검사 도구보다 먼저 작성되었기 때문에, 이미 있는 코드의 문제는 PR에서 고칠 대상이 아닙니다.
손대지 않은 줄에서 나온 문제로 PR이 실패하지 않습니다.

### 3.3 포크에서 올린 첫 PR

포크에서 처음 올린 PR은 관리자가 실행을 승인한 뒤에 검사가 시작됩니다.
그때까지는 검사가 대기 중으로 표시됩니다.

## 4. 실패한 검사 고치기

검사 도구의 버전은 [CI 설정](../../.github/workflows/ci.yml)에 고정되어 있습니다. 세 도구 모두 `pip install`로 설치할 수 있습니다.

| 도구 | 버전 |
|---|---|
| clang-format | 21.1.8 |
| clang-tidy | 22.1.8 |
| ruff | 0.17.4 |

### 4.1 clang-format

브랜치에서 바꾼 줄만 포맷하고, 결과를 커밋합니다.

```bash
git clang-format $(git merge-base origin/master HEAD)
```

### 4.2 ruff

```bash
ruff format path/to/file.py
ruff check .          # --fix를 붙이면 안전한 수정을 적용합니다
```

### 4.3 clang-tidy

검사 기록에 위치, 검사 이름이 나옵니다. 이름 규칙에 걸렸다면 바꿀 이름도 함께 나옵니다.

```
vai/nodes/slice.h:164:8: error: invalid case style for struct 'probe_struct' [readability-identifier-naming]
  164 | struct probe_struct { int x; };
      |        ^~~~~~~~~~~~
      |        ProbeStruct
```

PR을 올리기 전에 확인하려면 [1.3](#13-clang-tidy-훅-켜기)의 훅을 켜거나, 편집기가 `.clang-tidy`를 읽게 합니다.
Visual Studio와 clangd가 이 파일을 읽어 편집 중인 코드에 표시합니다.

Windows에서는 `git diff`의 출력을 `clang-tidy-diff`에 직접 넘기지 않습니다.
git은 경로에 `/`를 쓰고 clang-tidy는 `\`를 써서, 문제가 있어도 통과로 나옵니다. 훅의 스크립트는 경로를 바꿔서 넘깁니다.

### 4.4 노드 문서

[제공 노드들](06-nodes.md)은 `Node` 헤더의 설명 주석에서 생성합니다.
설명 주석, 슬롯, `Config`를 바꿨다면 문서를 다시 생성하고, 생성된 두 파일을 함께 커밋합니다. 이 문서는 직접 고치지 않습니다.

```bash
python docs/tools/gen-node-docs.py
```

## 5. 자동 검사가 확인하는 규칙

규칙의 기준은 저장소 루트의 설정 파일입니다. 이 절은 설정 파일의 내용을 요약한 것입니다.

### 5.1 C++ 형식

[`.clang-format`](../../.clang-format)이 확인합니다.

| 항목 | 규칙 |
|---|---|
| 들여쓰기와 줄 길이 | 공백 4칸을 쓰고 탭은 쓰지 않습니다. 한 줄은 120자까지입니다. |
| 여는 중괄호 | [공통 코드 컨벤션](index.md#3-공통-코드-컨벤션)의 3.2를 따릅니다. |
| 한 줄에 쓸 수 있는 것 | 클래스 안의 짧은 함수, 본문이 빈 함수, 짧은 람다, 짧은 `enum`입니다. `if`와 `for`의 본문은 항상 다음 줄에 둡니다. |
| 포인터와 참조 | `*`와 `&`는 타입에 붙입니다. (예: `const char* path`, `Node& node`.) |
| 포함문과 주석 | 포함문의 순서와 주석의 줄바꿈은 작성한 그대로 둡니다. |
| 제외 대상 | `external/`, `vai/shaders/`, `vai/utils/`의 외부 헤더(`stb_image.h`, `stb_image_write.h`, `tiny-json.h`)는 포맷하지 않습니다. |

### 5.2 C++ 이름

[`.clang-tidy`](../../.clang-tidy)가 확인합니다. 규칙과 예외는 [공통 코드 컨벤션](index.md#3-공통-코드-컨벤션)의 3.1에 있습니다.

상수 앞에 `k`를 붙이지 않는 규칙은 도구가 확인하지 못합니다. `kBeginOfText`도 형태는 `camelCase`이기 때문입니다. 이 규칙은 리뷰에서 확인합니다.

### 5.3 C++ 실수 검사

[`.clang-tidy`](../../.clang-tidy)가 `bugprone-*`, `performance-*`, `clang-analyzer-*` 검사를 실행합니다.
취향에 가까운 검사 네 개(`narrowing-conversions`, `exception-escape`, `easily-swappable-parameters`, `enum-size`)는 제외했습니다.

검사 결과가 해당 줄에 맞지 않으면, 그 줄에서 그 검사 하나만 끄고 이유를 적습니다.

```cpp
const float pad = float(M) / float(CEIL_DIV(M, b) * b);   // NOLINT(bugprone-integer-division): 올림 나눗셈이 의도입니다
```

검사 이름이 없는 `// NOLINT`는 그 줄의 모든 검사를 끄므로 쓰지 않습니다.

### 5.4 Python

[`pyproject.toml`](../../pyproject.toml)의 설정을 따릅니다.

| 항목 | 규칙 |
|---|---|
| 형식 | ruff format을 따릅니다. 한 줄은 100자까지입니다. |
| 검사 | ruff check의 `F`(pyflakes), `E4`·`E7`·`E9`(오류에 가까운 pycodestyle 규칙), `B`(bugbear)를 실행합니다. |
| 예외 | 결과가 해당 줄에 맞지 않으면 `# noqa: <규칙>`을 붙이고 이유를 적습니다. |

## 6. 노드를 추가할 때 함께 넣는 것

`Node`와 커널의 작성 방법은 [커스텀 노드와 커널 작성](07-contributing-nodes.md)에 있습니다.
이 절은 `Node`를 추가하는 PR에 함께 들어가야 하는 나머지 파일을 다룹니다.

| 순서 | 파일 | 위치 |
|---|---|---|
| 1 | Python 바인딩 | `python/bindings/nodes/<이름>.cpp` 또는 `python/bindings/models/<모델명>/nodes/<이름>.cpp` |
| 2 | 이름 공개 (공용 `Node`만) | `python/pyvai/nodes.py` |
| 3 | 테스트 | `python/tests/nodes/test_<이름>.py` 또는 `python/tests/models/<모델명>/test_<이름>.py` |
| 4 | 노드 문서 | `docs/ko/06-nodes.md`, `docs/en/06-nodes.md` |

바인딩, 등록, 테스트 중 하나라도 없으면 [`test_node_completeness.py`](../../python/tests/test_node_completeness.py)가 실패합니다.

### 6.1 Python 바인딩

바인딩 파일에는 `Config`의 필드를 나열하고, 마지막 줄에서 등록합니다. 다른 파일의 목록은 고치지 않습니다.
다음은 [`Slice`의 바인딩](../../python/bindings/nodes/slice.cpp)에서 발췌한 코드입니다.

```cpp
static void bindSlice(nb::module_& m)
{
    using C = vai::node::Slice::Config;
    bindNode<vai::node::Slice>(m, "SliceNode", "Take `count` consecutive indices ...",
                               required("start", &C::start),
                               field("count", &C::count).check(atLeast(1u), "must be >= 1"),
                               field("axis", &C::axis),
                               field("o_dtype", &C::oDtype));
}
PYVAI_NODE_BINDING(bindSlice);
```

| 함수 | 의미 |
|---|---|
| `field(이름, &C::멤버)` | 기본값은 `Config{}`의 값입니다. 바인딩에 기본값을 다시 적지 않습니다. dtype 필드는 Python 기본값이 `None`이고, `None`이면 `Config{}`의 값을 그대로 둡니다. |
| `field(이름, &C::멤버, 값)` | 바인딩이 지정한 값이 기본값입니다. |
| `required(이름, &C::멤버)` | 기본값이 없습니다. |
| `.check(조건, 메시지)` | 조건이 거짓이면 `ValueError("<Node>: <이름> <메시지>")`를 발생시킵니다. 잘못된 인자가 엔진의 `VAI_ASSERT`까지 가지 않게 합니다. |
| `PYVAI_NODE_BINDING(함수)` | 공용 `Node`의 바인딩을 등록합니다. |
| `PYVAI_MODEL_NODE_BINDING(모델명, 함수)` | 모델 전용 `Node`의 바인딩을 등록합니다. |

자세한 설명은 [`bind-node.h`](../../python/bindings/nodes/bind-node.h)의 주석에 있습니다.

### 6.2 이름 공개

공용 `Node`는 [`python/pyvai/nodes.py`](../../python/pyvai/nodes.py)의 가져오기 목록과 `__all__`에 이름을 추가합니다.

### 6.3 테스트

테스트는 `check_node()`를 호출합니다. `Node`를 만드는 함수, 입력 형상, `torch`로 작성한 기준 구현, 허용 오차를 전달합니다.
다음은 [`Slice`의 테스트](../../python/tests/nodes/test_slice.py)에서 발췌한 코드입니다.

```python
def test_slice_axis0_block():
    check_node(
        lambda: nodes.SliceNode(start=1, count=2, axis=0),
        {"in0": (4, 5)},
        lambda in0: in0[1:3],
        tol=EXACT,
    )
```

`check_node()`는 출력이 기준 구현과 허용 오차 안에서 같은지, 다시 실행해도 같은지, `NaN`이나 `Inf`가 없는지를 확인합니다.
허용 오차의 종류와 선택 기준은 [`node_check.py`](../../python/tests/node_check.py)에 있습니다.

### 6.4 노드 문서

[4.4](#44-노드-문서)의 방법으로 문서를 다시 생성하고, 생성된 두 파일을 `Node`와 함께 커밋합니다.

## 7. PR 올리고 머지하기

| 항목 | 규칙 |
|---|---|
| 머지 경로 | `master`에는 PR로만 반영합니다. 머지 커밋으로 합칩니다. |
| 승인 | 관리자 팀의 승인 두 개가 필요합니다. [`CODEOWNERS`](../../.github/CODEOWNERS)에 적힌 경로를 바꿨다면 해당 코드 소유자의 승인이 포함되어야 합니다. |
| PR 설명 | 바꾼 내용과 확인한 방법을 적습니다. 자신의 변경과 관계없는 이유로 실패한 검사가 있으면 그 사실도 적습니다. |
| 커밋 제목 | 기존 기록은 `종류(범위): 제목` 형식을 사용합니다. (예: `fix(python): ...`, `ci: ...`, `feat(core): ...`.) |
