import pytest
import torch
import torch.nn.functional as F
from node_check import TRANS, check_node, normal, uniform

from pyvai._pyvai import model_nodes

rtdetrv2 = model_nodes.rtdetrv2

SHAPES = [(4, 5), (2, 3), (1, 1)]  # (height, width) per level
POINTS = [2, 3, 1]
HEADS, DIM = 2, 4


def _ref(offset_scale, values, boxes, offsets, logits):
    """multi_scale_deformable_attn_pytorch with RT-DETRv2's sampling locations,
    softmax over every level's points."""
    batch, _, heads, dim = values.shape
    queries, ref_dims = offsets.shape[1], boxes.shape[3]
    weights = logits.softmax(-1)
    out = torch.zeros(batch, queries, heads, dim, dtype=values.dtype)
    token, point = 0, 0
    for level, ((h, w), count) in enumerate(zip(SHAPES, POINTS)):
        ref = boxes[:, :, 0 if boxes.shape[2] == 1 else level][:, :, None, None]
        off = offsets[:, :, :, point : point + count]
        if ref_dims == 4:
            loc = ref[..., :2] + off / count * ref[..., 2:] * offset_scale
        else:
            loc = ref + off / torch.tensor([w, h], dtype=off.dtype)
        v = values[:, token : token + h * w].reshape(batch, h, w, heads, dim).permute(0, 3, 4, 1, 2)
        grid = (2 * loc - 1).permute(0, 2, 1, 3, 4).reshape(batch * heads, queries, count, 2)
        sampled = F.grid_sample(
            v.reshape(batch * heads, dim, h, w),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(batch, heads, dim, queries, count)
        out += torch.einsum("bmdqn,bqmn->bqmd", sampled, weights[:, :, :, point : point + count])
        token, point = token + h * w, point + count
    return out.reshape(batch, queries, heads * dim)


def _inputs(batch, queries, ref_levels, ref_dims):
    tokens, points = sum(h * w for h, w in SHAPES), sum(POINTS)
    return {
        "values": (batch, tokens, HEADS, DIM),
        "boxes": uniform((batch, queries, ref_levels, ref_dims), 0.05, 0.95),
        # Some samples land outside the map, so zero padding counts.
        "offsets": normal((batch, queries, HEADS, points, 2), scale=1.5),
        "logits": (batch, queries, HEADS, points),
    }


@pytest.mark.parametrize(
    "ref_levels, ref_dims",
    [
        pytest.param(1, 4, id="decoder_boxes"),
        pytest.param(len(SHAPES), 2, id="points_per_level"),
    ],
)
def test_deformable_attention(ref_levels, ref_dims):
    offset_scale = 0.5
    check_node(
        lambda: rtdetrv2.DeformableAttentionNode(
            spatial_shapes=SHAPES, point_counts=POINTS, offset_scale=offset_scale
        ),
        _inputs(2, 7, ref_levels, ref_dims),
        lambda **slots: _ref(offset_scale, **slots),
        tol=TRANS,
        alt_inputs=_inputs(1, 70, ref_levels, ref_dims),
    )
