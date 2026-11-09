import pytest
import torch
from node_check import TRANS, check_node

from pyvai._pyvai import model_nodes

yolox = model_nodes.yolox

SCALES = (("p3", 8), ("p4", 16), ("p5", 32))


def _ref(image_size, **slots):
    """Megvii's decode_outputs over the P3, P4, P5 heads: one row per cell, P3
    first and cells row-major, (cx, cy, w, h, obj, class scores)."""
    rows = []
    for scale, stride in SCALES:
        regobj, cls = slots["regobj_" + scale], slots["cls_" + scale]
        g = image_size // stride
        pred = torch.cat([regobj, cls]).reshape(regobj.shape[0] + cls.shape[0], g * g).T
        ys, xs = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
        grid = torch.stack([xs, ys], -1).reshape(-1, 2).to(pred.dtype)
        centers = (pred[:, :2] + grid) * stride
        sizes = pred[:, 2:4].exp() * stride
        rows.append(torch.cat([centers, sizes, pred[:, 4:].sigmoid()], 1))
    return torch.cat(rows)


@pytest.mark.parametrize(
    "image_size, num_classes",
    [
        pytest.param(64, 3, id="img64"),
        pytest.param(96, 80, id="img96_coco"),  # odd 3 x 3 P5 grid, COCO's class count
    ],
)
def test_decode(image_size, num_classes):
    inputs = {}
    for scale, stride in SCALES:
        g = image_size // stride
        inputs["regobj_" + scale] = (5, g, g)
        inputs["cls_" + scale] = (num_classes, g, g)
    check_node(
        lambda: yolox.DecodeNode(image_size=image_size, num_classes=num_classes),
        inputs,
        lambda **slots: _ref(image_size, **slots),
        tol=TRANS,
    )
