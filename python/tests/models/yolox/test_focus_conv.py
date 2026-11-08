import numpy as np
import pytest
import torch
import torch.nn.functional as F
from node_check import check_node, normal, uniform

from pyvai._pyvai import model_nodes

yolox = model_nodes.yolox


def _focus(x):
    """Megvii's Focus: the four pixel phases of [C, 2H, 2W], stacked top-left,
    bottom-left, top-right, bottom-right."""
    return torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]])


def _ref(in0, weight, scale, bias):
    out = F.conv2d(_focus(in0)[None], weight, padding=1)[0]
    return F.silu(out * scale[:, None, None] + bias[:, None, None])


@pytest.mark.parametrize(
    "ci, co, hw, alt_hw",
    [
        pytest.param(3, 16, (16, 16), (26, 18), id="rgb"),
        pytest.param(3, 12, (22, 30), None, id="odd_tiles"),  # 11 x 15 output: no aligned tile
    ],
)
def test_focus_conv(ci, co, hw, alt_hw):
    check_node(
        lambda: yolox.FocusConvNode(in_channels=ci, out_channels=co),
        {"in0": (ci, *hw)},
        _ref,
        params={
            "weight": normal((co, 4 * ci, 3, 3), 1.0 / np.sqrt(36 * ci)),
            "scale": uniform((co,), 0.5, 1.5),
            "bias": (co,),
        },
        alt_inputs={"in0": (ci, *alt_hw)} if alt_hw else None,
    )
