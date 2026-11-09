import torch.nn.functional as F
from node_check import check_node

from pyvai import nodes


def test_layer_norm_with_beta():
    check_node(
        lambda: nodes.LayerNormNode(eps=1e-5),
        {"in0": (4, 96)},
        lambda in0, gamma, beta: F.layer_norm(in0, (96,), gamma, beta, 1e-5),
        params={"gamma": (96,), "beta": (96,)},
        alt_inputs={"in0": (7, 96)},
    )


def test_layer_norm_no_beta():
    check_node(
        lambda: nodes.LayerNormNode(eps=1e-5, has_beta=False),
        {"in0": (3, 50)},
        lambda in0, gamma: F.layer_norm(in0, (50,), gamma, None, 1e-5),
        params={"gamma": (50,)},
    )
