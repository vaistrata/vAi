import pytest
from node_check import check_node

from pyvai._pyvai import model_nodes

rtdetrv2 = model_nodes.rtdetrv2


@pytest.mark.parametrize(
    "in0, in1",
    [
        pytest.param((2, 3, 17), (2, 3, 17), id="same_shape"),
        pytest.param((2, 3, 17), (3, 1), id="in1_broadcast"),
        pytest.param((3, 1), (2, 3, 17), id="in0_broadcast"),
        pytest.param((4, 1, 5), (1, 6, 1), id="both_broadcast"),
    ],
)
def test_mul(in0, in1):
    check_node(
        lambda: rtdetrv2.MulNode(),
        {"in0": in0, "in1": in1},
        lambda in0, in1: in0 * in1,
        # A new shape re-prepares the node and rewrites its stride buffer.
        alt_inputs={"in0": (7, 2, 9), "in1": (2, 1)},
    )


def test_mul_constant_in1():
    """Query selection's valid-anchor mask: in1 is a parameter broadcast over channels."""
    check_node(
        lambda: rtdetrv2.MulNode(constant_in1=True),
        {"in0": (1, 40, 8)},
        lambda in0, in1: in0 * in1,
        params={"in1": (1, 40, 1)},
    )
