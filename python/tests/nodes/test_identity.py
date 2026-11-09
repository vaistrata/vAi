from node_check import EXACT, check_node

from pyvai import nodes


def test_identity():
    check_node(nodes.IdentityNode, {"in0": (3, 4)}, lambda in0: in0, tol=EXACT)
