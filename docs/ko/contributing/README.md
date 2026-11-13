# 기여 절차와 자동 검사

이 문서는 수정한 코드가 `master`에 들어가기까지의 절차를 설명합니다.
처음 한 번 준비할 것, 커밋과 PR에서 실행되는 검사, 검사가 실패했을 때 고치는 방법, 브랜치·커밋 메시지·버전 규칙을 순서대로 다룹니다.

코드 자체의 작성 방법은 추가하려는 대상에 따라 다음 문서를 참조하면 됩니다.

| 추가하려는 것 | 문서 |
|---|---|
| 모델 | [커스텀 모델 작성](models.md) |
| `Node`와 커널 | [커스텀 노드와 커널 작성](nodes.md) |
| 그룹 | [커스텀 그룹 작성](groups.md) |

## 1. 처음 한 번 준비하기

### 1.1 빌드와 커밋 훅

저장소는 `--recursive`로 받고, [빠른 실행](../02-quickstart.md)의 방법으로 빌드합니다.
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

검사는 PR을 올릴 때와 PR에 커밋을 추가할 때 실행됩니다. 고친 파일의 종류에 따라 필요한 검사만 실행됩니다.

| 검사 | 하는 일 | 실행되지 않는 경우 |
|---|---|---|
| `format` | 코드 형식과 Python 실수를 검사합니다. | 없습니다. 모든 PR에서 실행됩니다. |
| `tidy` | C++ 이름 규칙과 실수를 검사합니다. | `python/` 폴더, `docs/` 폴더, `mkdocs.yml`, `.md` 파일만 고친 PR |
| `vai-linux`, `vai-windows` | C++ 코드를 빌드합니다. (GCC 14, MSVC.) | 위와 같습니다. |
| `pyvai-linux` | Python 패키지를 빌드하고 `python/tests`를 실행합니다. | `docs/` 폴더, `mkdocs.yml`, `.md` 파일만 고친 PR |
| `docs` | [제공 노드들](../06-nodes.md)이 `Node` 헤더와 같은지 확인하고, 문서 사이트를 빌드합니다. | `docs/` 폴더, `mkdocs.yml`, `vai/nodes/` 폴더 중 어느 것도 고치지 않은 PR |

예를 들어 문서만 고친 PR에서는 `format`과 `docs`만 실행됩니다.
PR 화면에는 검사가 `CI / format`처럼 워크플로 이름과 함께 표시됩니다. `docs` 검사는 `docs / build`로 표시됩니다.

`format`과 `tidy`는 도구마다 확인하는 범위가 다릅니다.

| 검사 | 도구 | 확인하는 범위 | 실패하는 경우 |
|---|---|---|---|
| `format` | clang-format | 이번 PR에서 추가하거나 고친 C++ 줄 | 줄바꿈이나 들여쓰기 같은 코드 형식이 [`.clang-format`](../../../.clang-format)의 설정과 다를 때 |
| `format` | ruff format | 이번 PR에서 고친 Python 파일 전체 | 코드 형식이 ruff의 형식과 다를 때 |
| `format` | ruff check | 저장소의 모든 Python 파일 | 사용하지 않는 `import` 같은 실수가 있을 때 |
| `tidy` | clang-tidy | `vai/`, `models/`, `examples/` 폴더에서 이번 PR이 추가하거나 고친 C++ 줄 | 이름 규칙을 어겼거나 실수가 있을 때 |

### 3.2 고친 줄만 검사

C++ 검사는 파일 전체가 아니라 이번 PR에서 추가하거나 고친 줄만 확인합니다.
저장소의 코드가 검사 도구보다 먼저 작성되었기 때문에, 이미 있는 코드의 문제는 PR에서 고칠 대상이 아닙니다.
손대지 않은 줄에서 나온 문제로 PR이 실패하지 않습니다.

### 3.3 포크에서 올린 첫 PR

포크에서 처음 올린 PR은 관리자가 실행을 승인한 뒤에 검사가 시작됩니다.
그때까지는 검사가 대기 중으로 표시됩니다.

## 4. 실패한 검사 고치기

검사 도구의 버전은 [CI 설정](../../../.github/workflows/ci.yml)에 고정되어 있습니다. 세 도구 모두 `pip install`로 설치할 수 있습니다.

| 도구 | 버전 |
|---|---|
| clang-format | 21.1.8 |
| clang-tidy | 22.1.8 |
| ruff | 0.17.4 |

### 4.1 clang-format

브랜치에서 고친 줄만 포맷하고, 결과를 커밋합니다.

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

[제공 노드들](../06-nodes.md)은 `Node` 헤더의 설명 주석에서 생성합니다.
설명 주석, 슬롯, `Config`를 바꿨다면 문서를 다시 생성하고, 생성된 두 파일을 함께 커밋합니다. 이 문서는 직접 고치지 않습니다.

```bash
python docs/tools/gen-node-docs.py
```

## 5. 자동 검사가 확인하는 규칙

규칙의 기준은 저장소 루트의 설정 파일입니다. 이 절은 설정 파일의 내용을 요약한 것입니다.

### 5.1 C++ 형식

[`.clang-format`](../../../.clang-format)이 확인합니다.

| 항목 | 규칙 |
|---|---|
| 들여쓰기와 줄 길이 | 공백 4칸을 쓰고 탭은 쓰지 않습니다. 한 줄은 120자까지입니다. |
| 여는 중괄호 | [공통 코드 컨벤션](../index.md#3-공통-코드-컨벤션)의 3.2를 따릅니다. |
| 한 줄에 쓸 수 있는 것 | 클래스 안의 짧은 함수, 본문이 빈 함수, 짧은 람다, 짧은 `enum`입니다. `if`와 `for`의 본문은 항상 다음 줄에 둡니다. |
| 함수 호출의 인자 | 한 줄에 다 들어가지 않으면 인자마다 한 줄씩 씁니다. |
| 생성자 초기화 목록 | 멤버마다 한 줄씩 쓰고, `:`와 `,`를 줄 앞에 둡니다. |
| 중괄호 초기화 | 중괄호 안쪽에 공백을 둡니다. (예: `{ .inDim = 64, .outDim = 32 }`.) |
| 전처리 지시문 | `#if` 안의 지시문은 한 단계 들여씁니다. 포함 가드 안은 들여쓰지 않습니다. |
| 포인터와 참조 | `*`와 `&`는 타입에 붙입니다. (예: `const char* path`, `Node& node`.) |
| 포함문과 주석 | 포함문의 순서와 주석의 줄바꿈은 작성한 그대로 둡니다. |
| 제외 대상 | `external/`, `vai/shaders/`, `vai/utils/`의 외부 헤더(`stb_image.h`, `stb_image_write.h`, `tiny-json.h`)는 포맷하지 않습니다. |

### 5.2 C++ 이름

[`.clang-tidy`](../../../.clang-tidy)가 확인합니다. 규칙과 예외는 [공통 코드 컨벤션](../index.md#3-공통-코드-컨벤션)의 3.1에 있습니다.

상수 앞에 `k`를 붙이지 않는 규칙은 도구가 확인하지 못합니다. `kBeginOfText`도 형태는 `camelCase`이기 때문입니다. 이 규칙은 리뷰에서 확인합니다.

### 5.3 C++ 실수 검사

[`.clang-tidy`](../../../.clang-tidy)가 `bugprone-*`, `performance-*`, `clang-analyzer-*` 검사를 실행합니다.
취향에 가까운 검사 네 개(`narrowing-conversions`, `exception-escape`, `easily-swappable-parameters`, `enum-size`)와 선택형 분석 검사(`clang-analyzer-optin.*`)는 제외했습니다.

검사 결과가 해당 줄에 맞지 않으면, 그 줄에서 그 검사 하나만 끄고 이유를 적습니다.

```cpp
const float pad = float(M) / float(CEIL_DIV(M, b) * b);   // NOLINT(bugprone-integer-division): 올림 나눗셈이 의도입니다
```

검사 이름이 없는 `// NOLINT`는 그 줄의 모든 검사를 끄므로 쓰지 않습니다.

### 5.4 Python

[`pyproject.toml`](../../../pyproject.toml)의 설정을 따릅니다.

| 항목 | 규칙 |
|---|---|
| 형식 | ruff format을 따릅니다. 한 줄은 100자까지입니다. |
| 검사 | ruff check의 `F`(pyflakes), `E4`·`E7`·`E9`(오류에 가까운 pycodestyle 규칙), `B`(bugbear)를 실행합니다. |
| 예외 | 결과가 해당 줄에 맞지 않으면 `# noqa: <규칙>`을 붙이고 이유를 적습니다. |

## 6. 브랜치, 커밋 메시지, PR

브랜치는 [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow), 커밋 메시지는 [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)를 따릅니다.

### 6.1 브랜치

`master`에서 작업 브랜치를 만들고 PR로 합칩니다. `develop` 브랜치는 두지 않습니다.

| 브랜치 | 용도 |
|---|---|
| `master` | 리뷰와 검사를 마친 코드 |
| `feat/reshape-op` | 기능 추가 |
| `fix/model-loader-crash` | 버그 수정 |
| `docs/build-guide` | 문서 수정 |

- 이름은 `<종류>/<짧은-설명>` 형식이고, 소문자와 하이픈을 사용합니다. 종류는 [6.2](#62-커밋-메시지)의 표와 같습니다.
- 작업 하나에 브랜치 하나, PR 하나를 만듭니다.
- 포크에서 올리는 PR은 브랜치 이름을 검사하지 않습니다. 같은 형식을 권장합니다.

### 6.2 커밋 메시지

제목은 `종류(범위): 설명` 형식입니다.

- `범위`는 변경한 모듈이며 생략할 수 있습니다. (예: `python`, `engine`, `linear`, `yolox`.)
- `설명`은 영어 동사 원형으로 시작하고, 변경한 내용을 적습니다.

| 종류 | 사용하는 경우 | 예시 |
|---|---|---|
| `feat` | 새로운 기능 | `feat(graph): add reshape operation` |
| `fix` | 잘못된 동작 수정 | `fix(loader): reject truncated model files` |
| `perf` | 동작을 유지하는 성능 개선 | `perf(vulkan): reduce allocation overhead` |
| `refactor` | 동작을 유지하는 구조 변경 | `refactor(graph): extract shape helpers` |
| `docs` | 문서 수정 | `docs: clarify build requirements` |
| `test` | 테스트만 변경 | `test(loader): cover empty model files` |
| `build` | 빌드와 의존성 변경 | `build: require CMake 3.16` |
| `ci` | CI 변경 | `ci: add compiler checks` |
| `style` | 동작과 무관한 서식 변경 | `style: format public headers` |
| `chore` | 그 밖의 유지보수 | `chore: update the issue templates` |

종류는 다음 기준으로 고릅니다.

- 새로 할 수 있는 일이 생기면 `feat`입니다.
- 잘못된 동작을 바로잡으면 `fix`입니다.
- 같은 일을 더 빠르게 하면 `perf`입니다.

**한 커밋은 하나의 목적을 담습니다.** 기능과 그 테스트는 함께 넣고, 관계없는 리팩터링은 분리합니다. PR도 같은 기준으로 나눕니다.
나누는 기준은 "이 변경만 되돌려도 나머지 코드가 빌드되고 테스트를 통과하는가"입니다. 함께 되돌려야 하는 구현, 호출부, 테스트는 한 단위로 묶습니다.

기존 사용법과 호환되지 않는 변경은 종류 뒤에 `!`를 붙이고, 본문의 `BREAKING CHANGE:` 뒤에 영향과 전환 방법을 적습니다.

```text
feat(loader)!: require explicit model format

BREAKING CHANGE: load_model(path) now takes a format argument.
Change existing calls to load_model(path, format).
```

### 6.3 PR 올리고 머지하기

| 항목 | 규칙 |
|---|---|
| 머지 경로 | `master`에는 PR로만 반영합니다. |
| PR 제목 | [6.2](#62-커밋-메시지)의 커밋 제목 형식으로 적습니다. 스쿼시 머지에서는 이 제목이 `master`의 커밋 제목이 됩니다. |
| PR 설명 | 바꾼 내용과 확인한 방법을 적습니다. 자신의 변경과 관계없는 이유로 실패한 검사가 있으면 그 사실도 적습니다. 호환되지 않는 변경이면 `BREAKING CHANGE:` 문단을 포함합니다. |
| 승인 | 관리자 팀의 승인 두 개가 필요합니다. [`CODEOWNERS`](../../../.github/CODEOWNERS)에 적힌 경로를 바꿨다면 해당 코드 소유자의 승인이 포함되어야 합니다. |
| 머지 방식: 포크에서 올린 PR | 스쿼시 머지(squash merge)를 사용합니다. PR의 커밋들이 커밋 하나로 합쳐져 `master`에 들어갑니다. |
| 머지 방식: 저장소의 브랜치에서 올린 PR | 머지 커밋(merge commit)을 사용합니다. PR의 커밋들이 그대로 `master`에 들어가므로, 모든 커밋이 [6.2](#62-커밋-메시지)를 따라야 합니다. 그렇지 않은 PR은 스쿼시 머지를 사용합니다. |
| 머지 후 | 머지한 작업 브랜치는 삭제합니다. |

## 7. 버전

태그는 `vMAJOR.MINOR.PATCH` 형식이고, [SemVer](https://semver.org/)를 따릅니다.

| 변경 | 올리는 자리 | 예시 |
|---|---|---|
| 기존 공개 API와 호환되지 않는 변경 | MAJOR | `v1.2.3` → `v2.0.0` |
| 호환되는 기능 추가, 공개 API의 사용 중단 예고 | MINOR | `v1.2.3` → `v1.3.0` |
| 호환되는 버그 수정, 호환되는 성능 개선 | PATCH | `v1.2.3` → `v1.2.4` |

릴리즈 버전은 `master`에 들어간 커밋의 종류로 정합니다. 브랜치 이름은 버전에 영향을 주지 않습니다.

| `master`에 들어간 커밋 | 버전 영향 |
|---|---|
| 종류와 관계없이 `!` 또는 `BREAKING CHANGE` | MAJOR |
| `feat` | MINOR |
| `fix`, `perf` | PATCH |
| 그 밖의 종류 | 단독으로는 릴리즈하지 않음 |

- 여러 변경을 묶어 릴리즈하면 가장 높은 자리를 올립니다.
- `0.x`는 API가 안정되기 전의 버전입니다. 위 두 표는 `1.0` 이후에 적용합니다.
- 정식 릴리즈 전의 검증에는 `v1.0.0-rc.1` 같은 후보 버전을 사용합니다.
