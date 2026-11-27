import pytest
from node_check import EXACT, check_node

from pyvai import nodes


@pytest.mark.parametrize(
    "perms, shape",
    [
        pytest.param([2, 0, 1], (3, 4, 5), id="p201"),
        pytest.param([0, 2, 1, 3], (2, 3, 4, 5), id="p0213"),
        pytest.param([1, 0, 2], (1, 4, 5), id="p102"),
        # (0, 1) and (2, 3) move as runs: the kernel sees a 2-axis transpose.
        pytest.param([2, 3, 0, 1], (2, 3, 4, 5), id="p2301"),
        # PatchEmbedding's shape at tileSize 1: the kernel drops the size-1 axes.
        pytest.param([1, 4, 2, 5, 0, 3, 6], (3, 2, 1, 4, 3, 1, 4), id="size1-axes"),
    ],
)
def test_permute(perms, shape):
    check_node(
        lambda: nodes.PermuteNode(perms=perms),
        {"in0": shape},
        lambda in0: in0.permute(perms),
        tol=EXACT,
    )


def test_permute_leading_axes():
    # perms covers the last three axes; the leading two pass through.
    check_node(
        lambda: nodes.PermuteNode(perms=[1, 0, 2]),
        {"in0": (2, 3, 4, 5, 6)},
        lambda in0: in0.permute(0, 1, 3, 2, 4),
        tol=EXACT,
    )
