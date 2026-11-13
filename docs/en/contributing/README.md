# Contribution workflow and automated checks

This document describes the process for getting code changes into `master`.
It covers one-time setup, checks run on commits and PRs, how to fix failed checks, and the rules for branches, commit messages, and versions, in that order.

For writing the code itself, see the document for what you are adding.

| What you are adding | Document |
|---|---|
| Model | [Writing custom models](models.md) |
| `Node` and kernel | [Writing custom nodes and kernels](nodes.md) |
| Group | [Writing custom groups](groups.md) |

## 1. One-time setup

### 1.1 Build and commit hooks

Clone the repository with `--recursive` and build it as described in the [Quickstart](../02-quickstart.md).
Then install the commit hooks. The hooks check changes when committing.

```bash
pip install pre-commit
pre-commit install
```

### 1.2 Python package and tests

When modifying Python bindings or a `Node`, install the package and run the tests.
`torch` is used only as a reference implementation for tests. Without installing the CPU build first, the download will include several GB of CUDA binaries.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test]"
python -m pytest python/tests
```

### 1.3 Enable the clang-tidy hook

When modifying C++ code, also enable the clang-tidy hook. This hook does nothing until enabled.
clang-tidy requires compilation information (`compile_commands.json`), which Visual Studio builds do not provide.

```bash
python .github/scripts/clang-tidy-staged.py --setup
```

This command only configures CMake with Ninja in `build-tidy/`; it does not build the project.
On Windows, it finds the latest Visual Studio installation with C++ tools and runs in that environment, so a developer command prompt is not required.
Delete the `build-tidy/` directory to disable the hook.

## 2. Checks run on commits

| Hook | Scope | When a problem is found |
|---|---|---|
| clang-format | Staged C++ lines | Fixes the lines and stops the commit. Stage the fixes and commit again. |
| ruff format | Staged Python files | Same as above. |
| ruff check | Staged Python files | Reports the problem and stops the commit. |
| clang-tidy | Staged C++ lines in `vai/`, `models/`, and `examples/` | Reports the problem and stops the commit. Does nothing until [1.3](#13-enable-the-clang-tidy-hook) has been completed. |

Hooks help catch problems early. PRs run the same checks again, so skipping hook installation only delays when problems are found.

## 3. Checks run on PRs

### 3.1 Check list

Checks run when a PR is opened and when commits are added to it. Only the checks that the changed files need are run.

| Check | What it does | When it does not run |
|---|---|---|
| `format` | Checks code formatting and Python mistakes. | Never. It runs on every PR. |
| `tidy` | Checks C++ naming rules and mistakes. | PRs that change only the `python/` folder, the `docs/` folder, `mkdocs.yml`, or `.md` files |
| `vai-linux`, `vai-windows` | Builds the C++ code. (GCC 14, MSVC.) | Same as above. |
| `pyvai-linux` | Builds the Python package and runs `python/tests`. | PRs that change only the `docs/` folder, `mkdocs.yml`, or `.md` files |
| `docs` | Checks that [Built-in nodes](../06-nodes.md) matches the `Node` headers, and builds the documentation site. | PRs that change none of the `docs/` folder, `mkdocs.yml`, and the `vai/nodes/` folder |

For example, a PR that changes only documentation runs `format` and `docs`.

In `format` and `tidy`, each tool examines a different scope.

| Check | Tool | Scope | Failure condition |
|---|---|---|---|
| `format` | clang-format | C++ lines added or changed in this PR | Formatting such as line breaks or indentation differs from the settings in [`.clang-format`](../../../.clang-format). |
| `format` | ruff format | The whole of each Python file changed in this PR | Formatting differs from ruff's format. |
| `format` | ruff check | Every Python file in the repository | There is a mistake such as an unused `import`. |
| `tidy` | clang-tidy | C++ lines added or changed in this PR in the `vai/`, `models/`, and `examples/` folders | A naming rule is broken or there is a mistake. |

### 3.2 Checking only changed lines

C++ checks examine only the lines added or changed in this PR, rather than entire files.
Since the repository's code predates the checking tools, existing problems are outside the scope of the PR.
Problems on untouched lines do not cause the PR to fail.

### 3.3 First PR from a fork

Checks on the first PR from a fork start after a maintainer approves the run.
Until then, the checks are shown as pending.

## 4. Fixing failed checks

Tool versions are pinned in the [CI configuration](../../../.github/workflows/ci.yml). All three tools can be installed with `pip install`.

| Tool | Version |
|---|---|
| clang-format | 21.1.8 |
| clang-tidy | 22.1.8 |
| ruff | 0.17.4 |

### 4.1 clang-format

Format only the lines changed on the branch, then commit the result.

```bash
git clang-format $(git merge-base origin/master HEAD)
```

### 4.2 ruff

```bash
ruff format path/to/file.py
ruff check .          # Add --fix to apply safe fixes
```

### 4.3 clang-tidy

The check log shows the location and check name. Naming violations also show the suggested replacement name.

```
vai/nodes/slice.h:164:8: error: invalid case style for struct 'probe_struct' [readability-identifier-naming]
  164 | struct probe_struct { int x; };
      |        ^~~~~~~~~~~~
      |        ProbeStruct
```

To check before opening a PR, enable the hook in [1.3](#13-enable-the-clang-tidy-hook) or configure the editor to read `.clang-tidy`.
Visual Studio and clangd read this file and show diagnostics while editing.

On Windows, do not pass `git diff` output directly to `clang-tidy-diff`.
git uses `/` in paths while clang-tidy uses `\`, so checks can pass even when problems exist. The hook script converts the paths before passing them on.

### 4.4 Node documentation

[Built-in nodes](../06-nodes.md) is generated from the descriptive comments in `Node` headers.
After changing descriptive comments, slots, or `Config`, regenerate the documentation and commit both generated files. Do not edit these documents directly.

```bash
python docs/tools/gen-node-docs.py
```

## 5. Rules enforced by automated checks

The configuration files at the repository root define the rules. This section summarizes their contents.

### 5.1 C++ formatting

Checked by [`.clang-format`](../../../.clang-format).

| Item | Rule |
|---|---|
| Indentation and line length | Use four spaces, no tabs. Lines are limited to 120 characters. |
| Opening braces | Follow 3.2 of [Common code conventions](../index.md#3-common-code-conventions). |
| Constructs allowed on one line | Short functions inside classes, empty functions, short lambdas, and short `enum` declarations. Bodies of `if` and `for` always start on the next line. |
| Pointers and references | Attach `*` and `&` to the type. (Examples: `const char* path`, `Node& node`.) |
| Includes and comments | Preserve include order and comment line breaks as written. |
| Exclusions | Do not format `external/`, `vai/shaders/`, or the third-party headers in `vai/utils/` (`stb_image.h`, `stb_image_write.h`, `tiny-json.h`). |

### 5.2 C++ naming

Checked by [`.clang-tidy`](../../../.clang-tidy). Rules and exceptions are in 3.1 of [Common code conventions](../index.md#3-common-code-conventions).

The tool cannot enforce the rule against prefixing constants with `k`, because `kBeginOfText` also matches `camelCase`. This rule is checked in review.

### 5.3 C++ bug checks

[`.clang-tidy`](../../../.clang-tidy) runs `bugprone-*`, `performance-*`, and `clang-analyzer-*` checks.
Four checks that are closer to style preferences (`narrowing-conversions`, `exception-escape`, `easily-swappable-parameters`, `enum-size`) are excluded.

If a diagnostic does not apply to a line, disable only that check on that line and explain why.

```cpp
const float pad = float(M) / float(CEIL_DIV(M, b) * b);   // NOLINT(bugprone-integer-division): Ceiling division is intentional
```

Do not use `// NOLINT` without a check name, as it disables all checks on that line.

### 5.4 Python

Follow the configuration in [`pyproject.toml`](../../../pyproject.toml).

| Item | Rule |
|---|---|
| Formatting | Follow ruff format. Lines are limited to 100 characters. |
| Checks | Run ruff check's `F` (pyflakes), `E4`, `E7`, `E9` (pycodestyle rules closer to errors), and `B` (bugbear). |
| Exceptions | If a diagnostic does not apply to a line, add `# noqa: <rule>` and explain why. |

## 6. Branches, commit messages, and PRs

Branches follow [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow), and commit messages follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

### 6.1 Branches

Create a working branch from `master` and merge it through a PR. There is no `develop` branch.

| Branch | Purpose |
|---|---|
| `master` | Reviewed and checked code |
| `feat/reshape-op` | New feature |
| `fix/model-loader-crash` | Bug fix |
| `docs/build-guide` | Documentation change |

- Name a branch `<type>/<short-description>`, in lower case with hyphens. The types are those in the table in [6.2](#62-commit-messages).
- Create one branch and one PR per piece of work.
- Branch names are not checked for PRs from forks. The same format is recommended.

### 6.2 Commit messages

The subject has the form `type(scope): description`.

- `scope` is the module that changed and can be omitted. (Examples: `python`, `engine`, `linear`, `yolox`.)
- `description` starts with an English verb in the imperative and states what changed.

| Type | When to use | Example |
|---|---|---|
| `feat` | New feature | `feat(graph): add reshape operation` |
| `fix` | Fix for incorrect behavior | `fix(loader): reject truncated model files` |
| `perf` | Performance improvement with the same behavior | `perf(vulkan): reduce allocation overhead` |
| `refactor` | Structural change with the same behavior | `refactor(graph): extract shape helpers` |
| `docs` | Documentation change | `docs: clarify build requirements` |
| `test` | Test-only change | `test(loader): cover empty model files` |
| `build` | Build and dependency change | `build: require CMake 3.16` |
| `ci` | CI change | `ci: add compiler checks` |
| `style` | Formatting change that does not affect behavior | `style: format public headers` |
| `chore` | Other maintenance | `chore: update the issue templates` |

Choose the type as follows.

- Use `feat` when the change makes something new possible.
- Use `fix` when the change corrects incorrect behavior.
- Use `perf` when the change does the same thing faster.

**A commit has one purpose.** Put a feature and its tests together, and keep unrelated refactoring separate. Split PRs the same way.
The test is "if only this change is reverted, does the rest still build and pass its tests?" Implementation, call sites, and tests that must be reverted together belong in one unit.

For a change that is incompatible with existing usage, add `!` after the type and describe the impact and the migration after `BREAKING CHANGE:` in the body.

```text
feat(loader)!: require explicit model format

BREAKING CHANGE: load_model(path) now takes a format argument.
Change existing calls to load_model(path, format).
```

### 6.3 Opening and merging a PR

| Item | Rule |
|---|---|
| Merge path | Changes enter `master` only through PRs. |
| PR title | Use the commit subject format in [6.2](#62-commit-messages). With a squash merge, this title becomes the commit subject on `master`. |
| PR description | Describe the changes and how they were verified. Also mention any checks that failed for reasons unrelated to the changes. For an incompatible change, include a `BREAKING CHANGE:` paragraph. |
| Approval | Two approvals from the maintainer team are required. Changes to paths listed in [`CODEOWNERS`](../../../.github/CODEOWNERS) must include approval from the relevant code owner. |
| Merge method: PRs from forks | Use squash merge. The PR's commits enter `master` as one commit. |
| Merge method: PRs from branches of this repository | Use a merge commit. The PR's commits enter `master` as they are, so every commit must follow [6.2](#62-commit-messages). Otherwise, use squash merge. |
| After merging | Delete the merged working branch. |

## 7. Versions

Tags have the form `vMAJOR.MINOR.PATCH` and follow [SemVer](https://semver.org/).

| Change | Part to increase | Example |
|---|---|---|
| Change incompatible with the existing public API | MAJOR | `v1.2.3` → `v2.0.0` |
| Compatible new feature, deprecation notice for a public API | MINOR | `v1.2.3` → `v1.3.0` |
| Compatible bug fix, compatible performance improvement | PATCH | `v1.2.3` → `v1.2.4` |

The release version is decided by the types of the commits that entered `master`. Branch names do not affect the version.

| Commit on `master` | Effect on the version |
|---|---|
| `!` or `BREAKING CHANGE`, with any type | MAJOR |
| `feat` | MINOR |
| `fix`, `perf` | PATCH |
| Any other type | No release on its own |

- When a release bundles several changes, increase the highest part among them.
- `0.x` versions come before the API is stable. The two tables above apply from `1.0`.
- To verify before a full release, use a candidate version such as `v1.0.0-rc.1`.
