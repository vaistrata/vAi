# Contributing to vAi

The contribution guide is part of the documentation:

- [Contribution workflow and automated checks](docs/en/contributing/README.md)
  ([한국어](docs/ko/contributing/README.md)): what to set up once, the checks a
  commit and a pull request run, how to fix a failed check, what goes with a
  new node, and the rules for branches, commit messages and versions.
- [Common code conventions](docs/en/index.md) ([한국어](docs/ko/index.md#3-공통-코드-컨벤션)):
  names and braces.
- Writing [a model](docs/en/contributing/models.md),
  [a node and its kernel](docs/en/contributing/nodes.md) and
  [a group](docs/en/contributing/groups.md).

## Setup

```bash
pip install pre-commit && pre-commit install               # format and lint at commit time
python .github/scripts/clang-tidy-staged.py --setup        # if you change C++: turns the clang-tidy hook on
```

## Branches, commits and pull requests

- **Branch**: `<type>/<short-description>` from `master`, for example
  `feat/reshape-op` or `fix/model-loader-crash`. One branch and one pull
  request per piece of work.
- **Commit subject and pull request title**: `type(scope): description`, after
  [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). The
  types are `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`,
  `style` and `chore`; the scope is optional.

  ```text
  feat(graph): add reshape operation
  fix(loader): reject truncated model files
  ```

- **Incompatible change**: `!` after the type, and a `BREAKING CHANGE:`
  paragraph that says what breaks and how to move over.
- **Merge**: two approvals from the maintainers. A pull request from a fork is
  squash-merged, so its title becomes the commit on `master`. One from a branch
  of this repository gets a merge commit, so each of its commits has to follow
  the format.
- **Checks**: a pull request is checked on the lines it changes, and a new
  node comes with its Python binding and its test.

## Versions

Tags are `vMAJOR.MINOR.PATCH`, after [SemVer](https://semver.org/). From `1.0`,
a `feat` raises MINOR, a `fix` or `perf` raises PATCH, and a breaking change
raises MAJOR.
