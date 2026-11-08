from node_check import EXACT, check_node

from pyvai import nodes


def test_argmax():
    check_node(
        nodes.ArgmaxNode,
        {"in0": (4, 50)},
        lambda in0: in0.argmax(dim=-1),
        tol=EXACT,
        out="u32",
    )
