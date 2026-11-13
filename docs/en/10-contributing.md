# Contribution workflow and automated checks

This document describes the process for getting code changes into `master`.
It covers one-time setup, checks run on commits and PRs, and how to fix failed checks, in that order.

For writing the code itself, see [Writing custom models](05-contributing-models.md), [Writing custom nodes and kernels](07-contributing-nodes.md), and [Writing custom groups](08-contributing-groups.md).

## 1. One-time setup

### 1.1 Build and commit hooks

Clone the repository with `--recursive` and build it as described in the [Quickstart](02-quickstart.md).
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

| Check | When it runs | Failure conditions |
|---|---|---|
| `format` | Every PR | Changed C++ lines differ from clang-format output. Changed Python files differ from ruff format output. ruff check finds a problem anywhere in the repository. |
| `tidy` | PRs that change files outside `python/`, `docs/`, and `*.md` | clang-tidy finds a problem on changed C++ lines in `vai/`, `models/`, or `examples/`. |
| `vai-linux`, `vai-windows` | PRs that change files outside `python/`, `docs/`, and `*.md` | The C++ build fails. (GCC 14, MSVC.) |
| `pyvai-linux` | PRs that change files outside `docs/` and `*.md` | The Python package build or `python/tests` fails. |
| `docs` | PRs that change `docs/`, `mkdocs.yml`, or `vai/nodes/` | [Built-in nodes](06-nodes.md) differs from the `Node` headers, or the documentation site build fails. |

### 3.2 Checking only changed lines

C++ checks examine only lines added or changed by the PR, rather than entire files.
Since the repository's code predates the checking tools, existing problems are outside the scope of the PR.
Problems on untouched lines do not cause the PR to fail.

### 3.3 First PR from a fork

Checks on the first PR from a fork start after a maintainer approves the run.
Until then, the checks are shown as pending.

## 4. Fixing failed checks

Tool versions are pinned in the [CI configuration](../../.github/workflows/ci.yml). All three tools can be installed with `pip install`.

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

[Built-in nodes](06-nodes.md) is generated from the descriptive comments in `Node` headers.
After changing descriptive comments, slots, or `Config`, regenerate the documentation and commit both generated files. Do not edit these documents directly.

```bash
python docs/tools/gen-node-docs.py
```

## 5. Rules enforced by automated checks

The configuration files at the repository root define the rules. This section summarizes their contents.

### 5.1 C++ formatting

Checked by [`.clang-format`](../../.clang-format).

| Item | Rule |
|---|---|
| Indentation and line length | Use four spaces, no tabs. Lines are limited to 120 characters. |
| Opening braces | Follow 3.2 of [Common code conventions](index.md#3-common-code-conventions). |
| Constructs allowed on one line | Short functions inside classes, empty functions, short lambdas, and short `enum` declarations. Bodies of `if` and `for` always start on the next line. |
| Pointers and references | Attach `*` and `&` to the type. (Examples: `const char* path`, `Node& node`.) |
| Includes and comments | Preserve include order and comment line breaks as written. |
| Exclusions | Do not format `external/`, `vai/shaders/`, or the third-party headers in `vai/utils/` (`stb_image.h`, `stb_image_write.h`, `tiny-json.h`). |

### 5.2 C++ naming

Checked by [`.clang-tidy`](../../.clang-tidy). Rules and exceptions are in 3.1 of [Common code conventions](index.md#3-common-code-conventions).

The tool cannot enforce the rule against prefixing constants with `k`, because `kBeginOfText` also matches `camelCase`. This rule is checked in review.

### 5.3 C++ bug checks

[`.clang-tidy`](../../.clang-tidy) runs `bugprone-*`, `performance-*`, and `clang-analyzer-*` checks.
Four checks that are closer to style preferences (`narrowing-conversions`, `exception-escape`, `easily-swappable-parameters`, `enum-size`) are excluded.

If a diagnostic does not apply to a line, disable only that check on that line and explain why.

```cpp
const float pad = float(M) / float(CEIL_DIV(M, b) * b);   // NOLINT(bugprone-integer-division): Ceiling division is intentional
```

Do not use `// NOLINT` without a check name, as it disables all checks on that line.

### 5.4 Python

Follow the configuration in [`pyproject.toml`](../../pyproject.toml).

| Item | Rule |
|---|---|
| Formatting | Follow ruff format. Lines are limited to 100 characters. |
| Checks | Run ruff check's `F` (pyflakes), `E4`, `E7`, `E9` (pycodestyle rules closer to errors), and `B` (bugbear). |
| Exceptions | If a diagnostic does not apply to a line, add `# noqa: <rule>` and explain why. |

## 6. Files to include when adding a node

[Writing custom nodes and kernels](07-contributing-nodes.md) describes how to write a `Node` and its kernel.
This section covers the other files required in a PR that adds a `Node`.

| Order | File | Location |
|---|---|---|
| 1 | Python binding | `python/bindings/nodes/<name>.cpp` or `python/bindings/models/<model-name>/nodes/<name>.cpp` |
| 2 | Name export (common `Node` only) | `python/pyvai/nodes.py` |
| 3 | Test | `python/tests/nodes/test_<name>.py` or `python/tests/models/<model-name>/test_<name>.py` |
| 4 | Node documentation | `docs/ko/06-nodes.md`, `docs/en/06-nodes.md` |

[`test_node_completeness.py`](../../python/tests/test_node_completeness.py) fails if any binding, registration, or test is missing.

### 6.1 Python binding

List the `Config` fields in the binding file and register the binding on the last line. Do not modify lists in other files.
The following excerpt is from the [`Slice` binding](../../python/bindings/nodes/slice.cpp).

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

| Function | Meaning |
|---|---|
| `field(name, &C::member)` | The default is the value from `Config{}`. Do not repeat defaults in the binding. dtype fields default to `None` in Python; `None` leaves the `Config{}` value unchanged. |
| `field(name, &C::member, value)` | The binding specifies the default value. |
| `required(name, &C::member)` | No default value. |
| `.check(condition, message)` | Raises `ValueError("<Node>: <name> <message>")` if the condition is false. Prevents invalid arguments from reaching the engine's `VAI_ASSERT`. |
| `PYVAI_NODE_BINDING(function)` | Registers a common `Node` binding. |
| `PYVAI_MODEL_NODE_BINDING(model-name, function)` | Registers a model-specific `Node` binding. |

See the comments in [`bind-node.h`](../../python/bindings/nodes/bind-node.h) for details.

### 6.2 Name export

For a common `Node`, add the name to the import list and `__all__` in [`python/pyvai/nodes.py`](../../python/pyvai/nodes.py).

### 6.3 Tests

Tests call `check_node()` with a function that creates the `Node`, input shapes, a reference implementation written with `torch`, and a tolerance.
The following excerpt is from the [`Slice` test](../../python/tests/nodes/test_slice.py).

```python
def test_slice_axis0_block():
    check_node(
        lambda: nodes.SliceNode(start=1, count=2, axis=0),
        {"in0": (4, 5)},
        lambda in0: in0[1:3],
        tol=EXACT,
    )
```

`check_node()` checks that the output matches the reference implementation within tolerance, remains the same on repeated execution, and contains no `NaN` or `Inf`.
See [`node_check.py`](../../python/tests/node_check.py) for tolerance types and selection criteria.

### 6.4 Node documentation

Regenerate the documentation as described in [4.4](#44-node-documentation), and commit both generated files with the `Node`.

## 7. Opening and merging a PR

| Item | Rule |
|---|---|
| Merge path | Changes enter `master` only through PRs. Use merge commits. |
| Approval | Two approvals from the maintainer team are required. Changes to paths listed in [`CODEOWNERS`](../../.github/CODEOWNERS) must include approval from the relevant code owner. |
| PR description | Describe the changes and how they were verified. Also mention any checks that failed for reasons unrelated to the changes. |
| Commit subject | Existing history uses `type(scope): subject`. (Examples: `fix(python): ...`, `ci: ...`, `feat(core): ...`.) |
