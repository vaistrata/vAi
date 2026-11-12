# Contributing to vAi

The contribution guide is part of the documentation:

- [기여 절차와 자동 검사](docs/ko/10-contributing.md): what to set up once, the
  checks a commit and a pull request run, how to fix a failed check, what goes
  with a new node, and how a pull request gets merged. Korean for now.
- [Common code conventions](docs/ko/index.md#3-공통-코드-컨벤션): names and braces.
- Writing [a model](docs/ko/05-contributing-models.md),
  [a node and its kernel](docs/ko/07-contributing-nodes.md) and
  [a group](docs/ko/08-contributing-groups.md).

In short:

```bash
pip install pre-commit && pre-commit install               # format and lint at commit time
python .github/scripts/clang-tidy-staged.py --setup        # if you change C++: turns the clang-tidy hook on
```

A pull request is checked on the lines it changes, needs two approvals from the
maintainers, and a new node comes with its Python binding and its test.
