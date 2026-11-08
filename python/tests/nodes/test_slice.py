from node_check import EXACT, check_node

from pyvai import nodes


def test_slice_axis0_block():
    check_node(
        lambda: nodes.SliceNode(start=1, count=2, axis=0),
        {"in0": (4, 5)},
        lambda in0: in0[1:3],
        tol=EXACT,
    )


def test_slice_axis1_from_end():
    # A negative start counts from the end.
    check_node(
        lambda: nodes.SliceNode(start=-3, count=2, axis=1),
        {"in0": (3, 6, 2)},
        lambda in0: in0[:, 3:5],
        tol=EXACT,
    )
