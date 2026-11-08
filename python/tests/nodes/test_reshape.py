from node_check import EXACT, check_node

from pyvai import nodes


def test_reshape_infer():
    check_node(
        lambda: nodes.ReshapeNode(dims=[4, -1]),
        {"in0": (2, 3, 4)},
        lambda in0: in0.reshape(4, -1),
        tol=EXACT,
    )


def test_reshape_keep_dim0():
    # 0 keeps the input's dim.
    check_node(
        lambda: nodes.ReshapeNode(dims=[0, -1]),
        {"in0": (2, 3, 4)},
        lambda in0: in0.reshape(in0.shape[0], -1),
        tol=EXACT,
    )
