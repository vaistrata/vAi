import torch.nn.functional as F
from node_check import check_node

from pyvai._pyvai import model_nodes

qwen3vl = model_nodes.qwen3vl


def _ref(grid_h, grid_w, tile):
    """The table resampled to the patch grid as HF does, in tile order, added to in0."""

    def ref(in0, table):
        side = round(table.shape[0] ** 0.5)
        grid = table.reshape(side, side, -1).permute(2, 0, 1)[None]
        pos = F.interpolate(grid, size=(grid_h, grid_w), mode="bilinear", align_corners=True)[0]
        pos = pos.reshape(-1, grid_h // tile, tile, grid_w // tile, tile)
        pos = pos.permute(1, 3, 2, 4, 0).reshape(grid_h * grid_w, -1)
        return in0 + pos

    return ref


def test_interp_pos_embed_raster():
    check_node(
        lambda: qwen3vl.InterpPosEmbedNode(grid_width=5),
        {"in0": (15, 8)},
        _ref(3, 5, 1),
        params={"table": (16, 8)},
    )


def test_interp_pos_embed_tiles():
    # Two images of a 4 x 6 grid in 2 x 2 tiles: each one samples the table alike.
    check_node(
        lambda: qwen3vl.InterpPosEmbedNode(grid_width=6, tile_size=2),
        {"in0": (2, 24, 8)},
        _ref(4, 6, 2),
        params={"table": (9, 8)},
    )
