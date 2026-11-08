import pytest
import torch.nn.functional as F
from node_check import EXACT, check_node

from pyvai import nodes


@pytest.mark.parametrize(
    "kernel, stride, padding, shape",
    [
        pytest.param(3, 2, 1, (4, 11, 11), id="k3s2p1"),
        pytest.param(2, 2, 0, (3, 8, 8), id="k2s2"),
    ],
)
def test_max_pool2d(kernel, stride, padding, shape):
    # Samples outside the input never win.
    check_node(
        lambda: nodes.MaxPool2dNode(kernel_size=kernel, stride=stride, padding=padding),
        {"in0": shape},
        lambda in0: F.max_pool2d(in0, kernel, stride, padding),
        tol=EXACT,
    )


@pytest.mark.parametrize(
    "kernel, stride, padding, shape",
    [
        pytest.param(2, 2, 0, (3, 8, 8), id="k2s2"),
        pytest.param(3, 1, 1, (4, 7, 7), id="k3s1p1"),
    ],
)
def test_avg_pool2d(kernel, stride, padding, shape):
    # Zero padding counts in the divisor (count_include_pad, the default).
    check_node(
        lambda: nodes.AvgPool2dNode(kernel_size=kernel, stride=stride, padding=padding),
        {"in0": shape},
        lambda in0: F.avg_pool2d(in0, kernel, stride, padding, count_include_pad=True),
    )
