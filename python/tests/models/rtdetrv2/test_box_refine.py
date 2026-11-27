import numpy as np
import torch
from node_check import TRANS, check_node, normal

from pyvai._pyvai import model_nodes

rtdetrv2 = model_nodes.rtdetrv2


def _boxes(rows):
    """Normalized boxes, with exact 0 and 1 entries to reach the clamps."""

    def draw(rng):
        boxes = rng.uniform(0.02, 0.98, (rows, 4))
        boxes[0] = (0.0, 1.0, 0.0, 1.0)
        return boxes

    return draw


def _ref(boxes, delta):
    """RT-DETRv2's inverse_sigmoid(eps=1e-5) then sigmoid."""
    x = boxes.clamp(0, 1)
    logit = torch.log(x.clamp(min=1e-5) / (1 - x).clamp(min=1e-5))
    return torch.sigmoid(delta + logit)


def test_box_refine():
    check_node(
        rtdetrv2.BoxRefineNode,
        {"boxes": _boxes(300), "delta": normal((300, 4))},
        _ref,
        tol=TRANS,
        alt_inputs={"boxes": _boxes(7), "delta": normal((7, 4), scale=3.0)},
    )


def test_box_refine_saturates():
    """Large deltas saturate to 0 or 1 without NaN."""
    check_node(
        rtdetrv2.BoxRefineNode,
        {
            "boxes": _boxes(16),
            "delta": lambda rng: np.repeat([[-60.0], [60.0]], 32, 0).reshape(16, 4),
        },
        _ref,
        tol=TRANS,
    )
