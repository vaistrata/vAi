import torch.nn.functional as F
from node_check import check_node

from pyvai import nodes


def test_layer_norm_with_shift():
    check_node(
        lambda: nodes.LayerNormNode(eps=1e-5),
        {"in0": (4, 96)},
        lambda in0, scale, shift: F.layer_norm(in0, (96,), scale, shift, 1e-5),
        params={"scale": (96,), "shift": (96,)},
        alt_inputs={"in0": (7, 96)},
    )


def test_layer_norm_no_shift():
    check_node(
        lambda: nodes.LayerNormNode(eps=1e-5),
        {"in0": (3, 50)},
        lambda in0, scale: F.layer_norm(in0, (50,), scale, None, 1e-5),
        params={"scale": (50,)},
    )
