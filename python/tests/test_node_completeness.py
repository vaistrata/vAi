"""Every core node has a binding and a numeric case.

Fails on a node header in vai/nodes/ with no binding in python/bindings/nodes/,
and on a bound node with no case in node_cases.py. Nodes written before these
tests are listed in UNTESTED until their cases land.
"""

from pathlib import Path

from node_cases import COVERED

import pyvai

ROOT = Path(__file__).resolve().parents[2]
NOT_NODES = {"node-common"}  # vai/nodes/ headers that declare no node

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


def _bound_nodes():
    return {
        name
        for name, cls in vars(pyvai._pyvai).items()
        if isinstance(cls, type)
        and issubclass(cls, pyvai.Node)
        and cls not in (pyvai.Node, pyvai.InputNode, pyvai.OutputNode)
    }


def test_every_node_header_has_a_binding():
    headers = {p.stem for p in (ROOT / "vai" / "nodes").glob("*.h")} - NOT_NODES
    bindings = {p.stem for p in (ROOT / "python" / "bindings" / "nodes").glob("*.cpp")}
    missing = sorted(headers - bindings)
    assert not missing, f"vai/nodes/<name>.h with no python/bindings/nodes/<name>.cpp: {missing}"


def test_every_bound_node_has_a_case():
    missing = sorted(_bound_nodes() - COVERED - UNTESTED)
    assert not missing, f"bound nodes with no spec in node_cases.py: {missing}"


def test_untested_list_is_current():
    covered = sorted(UNTESTED & COVERED)
    unknown = sorted(UNTESTED - _bound_nodes())
    assert not covered, f"remove from UNTESTED, node_cases.py has specs for them now: {covered}"
    assert not unknown, f"remove from UNTESTED, no such bound node: {unknown}"
