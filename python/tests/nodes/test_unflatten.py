from node_check import EXACT, check_node

from pyvai import nodes


def test_unflatten_infer():
    check_node(
        lambda: nodes.UnflattenNode(axis=1, sizes=[-1, 2]),
        {"in0": (2, 6, 4)},
        lambda in0: in0.unflatten(1, (-1, 2)),
        tol=EXACT,
    )


def test_unflatten_negative_axis():
    # A negative axis counts from the end.
    check_node(
        lambda: nodes.UnflattenNode(axis=-1, sizes=[2, 3]),
        {"in0": (4, 6)},
        lambda in0: in0.unflatten(-1, (2, 3)),
        tol=EXACT,
    )
