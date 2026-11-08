"""Every node has a binding and a numeric spec.

Covers the core nodes (vai/nodes/) and every model's own nodes
(models/<model>/nodes/). Fails on a node header with no binding, a model whose
nodes have no submodule in pyvai._pyvai.model_nodes, and a bound node with no
spec in node_cases.py. Nodes written before these tests are listed in UNTESTED
until their specs land.
"""

from pathlib import Path

from node_cases import COVERED, MODEL_NODES, node_name

import pyvai

ROOT = Path(__file__).resolve().parents[2]
BINDINGS = ROOT / "python" / "bindings"
NOT_NODES = {"vai/nodes/node-common.h"}  # node headers that declare no node

# Stateful nodes (KV cache, counters, sampling, branches) whose specs need
# multi-step runs. Only shrinks: drop a name when its spec lands. A new node
# never goes here; it comes with its spec.
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


def _node_classes(module):
    return [
        cls
        for cls in vars(module).values()
        if isinstance(cls, type)
        and issubclass(cls, pyvai.Node)
        and cls not in (pyvai.Node, pyvai.InputNode, pyvai.OutputNode)
    ]


def _bound_nodes():
    """node_name() of every bound node, core and model."""
    modules = [pyvai._pyvai, *MODEL_NODES.values()]
    return {node_name(cls) for module in modules for cls in _node_classes(module)}


def _model_of(header):
    """models/<model>/nodes/x.h -> the submodule name for <model>."""
    return header.parent.parent.name.replace("-", "_")


def _node_headers():
    """(header, its binding) for vai/nodes/ and every models/<model>/nodes/."""
    pairs = [
        (h, BINDINGS / "nodes" / f"{h.stem}.cpp") for h in (ROOT / "vai" / "nodes").glob("*.h")
    ]
    pairs += [
        (h, BINDINGS / "models" / h.parent.parent.name / "nodes" / f"{h.stem}.cpp")
        for h in (ROOT / "models").glob("*/nodes/*.h")
    ]
    return sorted((h, b) for h, b in pairs if h.relative_to(ROOT).as_posix() not in NOT_NODES)


def test_every_node_header_has_a_binding():
    missing = [
        f"{h.relative_to(ROOT).as_posix()} -> {b.relative_to(ROOT).as_posix()}"
        for h, b in _node_headers()
        if not b.exists()
    ]
    assert not missing, f"node headers with no binding: {missing}"


def test_every_model_with_nodes_has_a_submodule():
    models = {_model_of(h) for h in (ROOT / "models").glob("*/nodes/*.h")}
    missing = sorted(models - set(MODEL_NODES))
    assert not missing, f"models/<model>/nodes/ not bound under pyvai._pyvai.model_nodes: {missing}"


def test_every_bound_node_has_a_case():
    missing = sorted(_bound_nodes() - COVERED - UNTESTED)
    assert not missing, f"bound nodes with no spec in node_cases.py: {missing}"


def test_untested_list_is_current():
    covered = sorted(UNTESTED & COVERED)
    unknown = sorted(UNTESTED - _bound_nodes())
    assert not covered, f"remove from UNTESTED, node_cases.py has specs for them now: {covered}"
    assert not unknown, f"remove from UNTESTED, no such bound node: {unknown}"
