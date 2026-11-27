from node_check import check_node

from pyvai._pyvai import model_nodes

rtdetrv2 = model_nodes.rtdetrv2


def test_add():
    check_node(
        rtdetrv2.AddNode,
        {"in0": (2, 3, 17), "in1": (2, 3, 17)},
        lambda in0, in1: in0 + in1,
        alt_inputs={"in0": (5, 64), "in1": (5, 64)},
    )


def test_add_constant_in1():
    """The encoder's position embedding: in1 is a parameter, not an input."""
    check_node(
        lambda: rtdetrv2.AddNode(constant_in1=True),
        {"in0": (6, 32)},
        lambda in0, in1: in0 + in1,
        params={"in1": (6, 32)},
    )
