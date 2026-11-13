"""Every node has a binding, is registered, and has a test.

For each node header, vai/nodes/<name>.h or models/<impl>/<model>/nodes/<name>.h
(<impl> is production or experimental):
- its binding, python/bindings/nodes/<name>.cpp or
  python/bindings/models/<model>/nodes/<name>.cpp, exists;
- every node class the binding names (a "<Name>Node" string) is registered, in
  pyvai._pyvai or in pyvai._pyvai.model_nodes.<model>;
- a test file under python/tests/nodes/ or python/tests/models/<model>/ names each
  of those classes. By convention it is test_<name>.py, hyphens as underscores,
  and calls node_check.check_node().
Nodes written before these tests are listed in UNTESTED until their tests land.
"""

import re
from functools import lru_cache
from pathlib import Path

import pyvai
from pyvai._pyvai import model_nodes

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "python" / "tests"
NOT_NODES = {"vai/nodes/node-common.h"}  # node headers that declare no node
NODE_NAME = re.compile(r'"(\w+Node)"')

# Stateful nodes (KV cache, counters, sampling, branches) whose tests need
# multi-step runs. Only shrinks: drop a name when its test lands. A new node
# never goes here; it comes with its test.
UNTESTED = frozenset(
    {
        "GateNode",
        "RowSliceScatterNode",
        "IndexCopyNode",
        "TokenEmbeddingNode",
        "TimeEmbeddingNode",
        "CyclicTableReaderNode",
        "RopeNode",
        "AttentionNode",
        "TopKSamplingNode",
    }
)


def _rel(path):
    return path.relative_to(ROOT).as_posix()


@lru_cache(maxsize=None)
def _nodes():
    """(header, binding, module, tests dir, node class names) per node header."""
    groups = [
        (
            ROOT / "vai" / "nodes",
            ROOT / "python" / "bindings" / "nodes",
            pyvai._pyvai,
            TESTS / "nodes",
        )
    ]
    for model_dir in sorted(p.parent for p in (ROOT / "models").glob("*/*/nodes")):
        model = model_dir.name
        groups.append(
            (
                model_dir / "nodes",
                ROOT / "python" / "bindings" / "models" / model / "nodes",
                getattr(model_nodes, model.replace("-", "_"), None),
                TESTS / "models" / model,
            )
        )
    found = []
    for headers, bindings, module, tests in groups:
        for header in sorted(headers.glob("*.h")):
            if _rel(header) in NOT_NODES:
                continue
            binding = bindings / f"{header.stem}.cpp"
            text = binding.read_text(encoding="utf-8") if binding.exists() else ""
            found.append((header, binding, module, tests, sorted(set(NODE_NAME.findall(text)))))
    return tuple(found)


@lru_cache(maxsize=None)
def _test_text(tests):
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(tests.glob("test_*.py")))


def _tested(name, tests):
    return re.search(rf"\b{name}\b", _test_text(tests)) is not None


def test_every_node_header_has_a_binding():
    missing = [f"{_rel(h)} -> {_rel(b)}" for h, b, _, _, _ in _nodes() if not b.exists()]
    assert not missing, f"node headers with no binding: {missing}"


def test_every_bound_node_is_registered():
    """A binding file without its PYVAI_NODE_BINDING line binds nothing."""
    unregistered = []
    for _, binding, module, _, names in _nodes():
        if not binding.exists():
            continue
        if not names:
            unregistered.append(f'{_rel(binding)} names no "<Name>Node" class')
        for name in names:
            cls = getattr(module, name, None)
            if not (isinstance(cls, type) and issubclass(cls, pyvai.Node)):
                where = module.__name__ if module else "model_nodes (PYVAI_MODEL_NODE_BINDING)"
                unregistered.append(f"{name} ({_rel(binding)}) is not in {where}")
    assert not unregistered, f"bindings not registered: {unregistered}"


def test_every_node_has_a_test():
    missing = [
        f"{name}: add {_rel(tests / ('test_' + header.stem.replace('-', '_') + '.py'))}"
        for header, _, _, tests, names in _nodes()
        for name in names
        if name not in UNTESTED and not _tested(name, tests)
    ]
    assert not missing, f"nodes with no test: {missing}"


def test_untested_list_is_current():
    named = {name: tests for _, _, _, tests, names in _nodes() for name in names}
    tested = sorted(n for n in UNTESTED if n in named and _tested(n, named[n]))
    unknown = sorted(UNTESTED - set(named))
    assert not tested, f"remove from UNTESTED, a test names them now: {tested}"
    assert not unknown, f"remove from UNTESTED, no binding names them: {unknown}"
