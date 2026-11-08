import pytest
import torch.nn.functional as F
from node_check import EXACT, check_node

from pyvai import nodes


@pytest.mark.parametrize("scale", [2, 3])
def test_nearest_upsample(scale):
    check_node(
        lambda: nodes.NearestUpsampleNode(scale=scale),
        {"in0": (3, 5, 7)},
        lambda in0: F.interpolate(in0[None], scale_factor=scale, mode="nearest")[0],
        tol=EXACT,
    )
