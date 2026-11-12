# Contributing to vAi

How a change gets from your clone to `master`: what to set up once, what checks a
pull request runs, the code rules those checks hold, and how to fix a check that
fails. The guides for the code itself are in the docs:
[writing a model](docs/en/05-contributing-models.md) and
[writing a node and its kernel](docs/en/07-contributing-nodes.md).

## Once per clone

Build as the [README](README.md#build) describes; clone with `--recursive`.

Install the commit hooks:

```bash
pip install pre-commit
pre-commit install
```

For the Python package and its tests:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # the CPU build; the default is several GB
pip install -e ".[test]"
python -m pytest python/tests
```

If you change C++, turn the clang-tidy hook on as well. It is off until you do,
since it needs a compile database and a Visual Studio build has none:

```bash
python .github/scripts/clang-tidy-staged.py --setup
```

This configures `build-tidy/` with Ninja; it does not build. On Windows it finds
the newest Visual Studio with the C++ tools and runs inside its environment, so
it needs no developer prompt. Deleting `build-tidy/` turns the hook off again.

## What happens at commit

| Hook | Reads | On a finding |
|---|---|---|
| clang-format | the staged C++ lines | rewrites them and stops the commit: stage the result and commit again |
| ruff format | the staged Python files | the same |
| ruff check | the staged Python files | reports and stops the commit |
| clang-tidy | the staged C++ lines of `vai/`, `models/`, `examples/` | reports and stops the commit; does nothing until `--setup` |

The hooks are the convenience. CI checks every pull request again, so a clone
without them still gets the same verdict, only later.

## What a pull request runs

| Check | Runs when the PR changes | Fails on |
|---|---|---|
| `format` | anything | C++ lines the PR changed that clang-format would rewrite; Python files it changed that ruff format would rewrite; any ruff check finding in the tree |
| `tidy` | anything outside `python/`, `docs/` and `*.md` | a clang-tidy finding on a C++ line the PR changed, in `vai/`, `models/` or `examples/` |
| `vai-linux`, `vai-windows` | anything outside `python/`, `docs/` and `*.md` | the C++ build failing (GCC 14, MSVC) |
| `pyvai-linux` | anything outside `docs/` and `*.md` | the Python package not building, or `python/tests` failing |
| `docs` | `docs/`, `mkdocs.yml` or `vai/nodes/` | the node reference differing from the node headers, or the site not building |

**Only your lines count.** The C++ checks read the lines the PR adds or changes,
not whole files: the tree is older than the checks, and what it already has is
not yours to fix. A finding on a line you did not touch does not fail your PR.

A first pull request from a fork runs its checks once a maintainer approves the
run; until then they show as waiting.

## Fixing a check

The tools are the versions CI pins, in `.github/workflows/ci.yml`:
clang-format 21.1.8, clang-tidy 22.1.8, ruff 0.17.4. `pip install` takes all
three.

**clang-format.** Format the lines your branch changed and commit the result:

```bash
git clang-format $(git merge-base origin/master HEAD)
```

**ruff format, ruff check.**

```bash
ruff format path/to/file.py
ruff check .          # --fix applies the fixes that are safe
```

**clang-tidy.** The log names the line, the check and, for a name, the name it
wants:

```
vai/nodes/slice.h:164:8: error: invalid case style for struct 'probe_struct' [readability-identifier-naming]
  164 | struct probe_struct { int x; };
      |        ^~~~~~~~~~~~
      |        ProbeStruct
```

To see the findings before pushing, turn the hook on (above), or let your editor
read `.clang-tidy`: Visual Studio and clangd both do. On Windows do not pipe
`git diff` into `clang-tidy-diff` yourself: it passes every change there, git's
paths using `/` and clang-tidy's `\`. The hook's script rewrites them.

**docs.** The node reference (`docs/*/06-nodes.md`) is generated from the node
headers' stamps. After changing a stamp, a slot or a `Config`, regenerate it and
commit both files; do not edit them by hand:

```bash
python docs/tools/gen-node-docs.py
```

## Code rules

The config files are the rules; this is what they say.

### C++ layout: `.clang-format`

- 4 spaces, no tabs, 120 columns.
- An opening brace goes on the next line: functions, `if` / `for` / `while` /
  `switch`, `class`, `struct`, `case` blocks. `namespace`, `enum` and a lambda
  body open on the same line.
- A short function inside a class, an empty body, a short lambda and a short
  `enum` may sit on one line. The body of an `if` or a `for` never does.
- `*` and `&` bind to the type: `const char* path`, `Node& node`.
- Include order and comment wrapping are left as written: a node header's stamp
  is the source of the node reference.
- Not formatted: `external/`, `vai/shaders/`, and the vendored `stb_image.h`,
  `stb_image_write.h`, `tiny-json.h` in `vai/utils/`.

### C++ names: `.clang-tidy`

| | Rule | Example |
|---|---|---|
| class, struct, enum, type alias, concept | `PascalCase` | `NeuralNet`, `PipeParams` |
| function, method | `camelCase` | `loadWeights()`, `setProfileTag()` |
| variable, parameter | `camelCase` | `outShape`, `targetDtype` |
| public member | `camelCase` | `hasBias`, `iDtype` |
| private or protected member | `camelCase_`, ending in `_` | `device_`, `cmdBuffer_` |
| enum value, in an `enum class` or a plain `enum` | `PascalCase` | `Fp16`, `Bilinear` |
| constant | `camelCase`, with no `k` in front | `tokensPerSide` |
| namespace | lower case | `vai`, `node`, `detail` |
| macro | `UPPER_CASE` | `VAI_ASSERT` |

What the check lets through:

- A tensor dimension keeps the letter the maths gives it: `N`, `K`, `Co`, `Dh`,
  `Tmax`.
- A model's size preset starts with `_`, since a name cannot start with a
  digit: `_124M()`, `_3B_224()`.
- A node's public `config_`: the constructor's parameter is `config`. Write new
  nodes the same way for now; the `_` goes when the nodes change together.

The check cannot tell a `k` prefix from any other camelCase name, so that one
is for review.

### C++ mistakes: `.clang-tidy`

`bugprone-*`, `performance-*` and `clang-analyzer-*`, without the checks that
mostly argue taste (`narrowing-conversions`, `exception-escape`,
`easily-swappable-parameters`, `enum-size`).

When a finding is wrong for the line, silence that one check on that one line
and say why:

```cpp
const float pad = float(M) / float(CEIL_DIV(M, b) * b);   // NOLINT(bugprone-integer-division): the ceiling is the point
```

A bare `// NOLINT` silences every check on the line; name the check.

### Python: `pyproject.toml`

- ruff format, 100 columns.
- ruff check: pyflakes (`F`), the pycodestyle errors that are bugs or near it
  (`E4`, `E7`, `E9`) and bugbear (`B`). It reads the whole tree, which passes.
- To silence a finding that is wrong for the line: `# noqa: <rule>` with the
  reason after it.

## Adding a node

1. The node, `vai/nodes/<name>.h`, and its kernel, `vai/shaders/<name>.comp`;
   or `models/<model>/nodes/<name>.h` for a node only one model uses.
   [Writing a node](docs/en/07-contributing-nodes.md) covers both.
2. Its Python binding, `python/bindings/nodes/<name>.cpp`, or
   `python/bindings/models/<model>/nodes/<name>.cpp`. List the `Config` fields
   and end with the registration line; nothing else needs editing:

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

   - `field(name, &C::member)`: the default is `Config{}`'s, so the binding does
     not repeat the core's. A dtype field takes `None` for it.
   - `required(name, &C::member)`: no default.
   - `.check(predicate, message)`: a bad argument raises
     `ValueError("<Node>: <name> <message>")` instead of reaching the core's
     `VAI_ASSERT`.
   - A model's node ends with `PYVAI_MODEL_NODE_BINDING(<model>, bindFn)`.

   `python/bindings/nodes/bind-node.h` has the rest.
3. For a core node, its name in `python/pyvai/nodes.py`: the import list and
   `__all__`.
4. Its test, `python/tests/nodes/test_<name>.py` or
   `python/tests/models/<model>/test_<name>.py`, calling `check_node()` with a
   reference written in torch:

   ```python
   def test_slice_axis0_block():
       check_node(
           lambda: nodes.SliceNode(start=1, count=2, axis=0),
           {"in0": (4, 5)},
           lambda in0: in0[1:3],
           tol=EXACT,
       )
   ```

   `python/tests/test_node_completeness.py` fails for a node with no binding, no
   registration or no test.
5. The node reference: `python docs/tools/gen-node-docs.py`, and commit the two
   `06-nodes.md` files with the node.

## Pull requests

- `master` takes changes through pull requests only, merged with a merge commit.
- A PR needs two approvals from the maintainers' team; a code owner must be
  among them for the paths `.github/CODEOWNERS` lists.
- Say what changed and how you checked it. If a check is red for a reason that
  is not your change, say that too.
- Commit subjects in the history read `type(scope): subject`, e.g.
  `fix(python): ...`, `ci: ...`, `feat(core): ...`.
