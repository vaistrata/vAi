import numpy as np
import pytest
import torch
import torch.nn.functional as F
from node_check import check_node, normal, uniform

from pyvai._pyvai import model_nodes

yolox = model_nodes.yolox


def _focus(x):
    """Megvii's Focus: the four pixel phases of [..., C, 2H, 2W], stacked top-left,
    bottom-left, top-right, bottom-right."""
    phases = [x[..., 0::2, 0::2], x[..., 1::2, 0::2], x[..., 0::2, 1::2], x[..., 1::2, 1::2]]
    return torch.cat(phases, dim=-3)


def _ref(in0, weight, scale, bias):
    x = _focus(in0)
    lead = x.shape[:-3]
    out = F.conv2d(x.reshape(-1, *x.shape[-3:]), weight, padding=1)
    out = F.silu(out * scale[:, None, None] + bias[:, None, None])
    return out.reshape(*lead, *out.shape[-3:])


@pytest.mark.parametrize(
    "ci, co, in_shape, alt_shape",
    [
        pytest.param(3, 16, (3, 16, 16), (3, 26, 18), id="rgb"),
        pytest.param(3, 12, (3, 22, 30), None, id="odd_tiles"),  # 11 x 15 output: no aligned tile
        pytest.param(3, 16, (2, 3, 16, 16), (3, 3, 10, 14), id="batched"),
    ],
)
def test_focus(ci, co, in_shape, alt_shape):
    check_node(
        lambda: yolox.FocusNode(in_channels=ci, out_channels=co),
        {"in0": in_shape},
        _ref,
        params={
            "weight": normal((co, 4 * ci, 3, 3), 1.0 / np.sqrt(36 * ci)),
            "scale": uniform((co,), 0.5, 1.5),
            "bias": (co,),
        },
        alt_inputs={"in0": alt_shape} if alt_shape else None,
    )
