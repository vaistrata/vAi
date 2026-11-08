from node_check import EXACT, check_node

from pyvai import nodes


def test_flatten():
    check_node(nodes.FlattenNode, {"in0": (2, 3, 4)}, lambda in0: in0.flatten(), tol=EXACT)
