from node_check import check_node

from pyvai import nodes


def test_mul():
    check_node(
        nodes.MulNode,
        {"in0": (2, 3, 17), "in1": (2, 3, 17)},
        lambda in0, in1: in0 * in1,
    )
