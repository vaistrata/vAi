from node_check import check_node

from pyvai import nodes


def test_add():
    check_node(
        nodes.AddNode,
        {"in0": (2, 3, 17), "in1": (2, 3, 17)},
        lambda in0, in1: in0 + in1,
        alt_inputs={"in0": (5, 64), "in1": (5, 64)},
    )
