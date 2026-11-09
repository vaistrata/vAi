import pytest
from node_check import EXACT, check_node

from pyvai import nodes


@pytest.mark.parametrize(
    "perms, shape",
    [
        pytest.param([2, 0, 1], (3, 4, 5), id="p201"),
        pytest.param([0, 2, 1, 3], (2, 3, 4, 5), id="p0213"),
        pytest.param([1, 0, 2], (1, 4, 5), id="p102"),
    ],
)
def test_permute(perms, shape):
    check_node(
        lambda: nodes.PermuteNode(perms=perms),
        {"in0": shape},
        lambda in0: in0.permute(perms),
        tol=EXACT,
    )
