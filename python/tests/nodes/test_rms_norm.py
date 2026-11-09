import torch.nn.functional as F
from node_check import check_node

from pyvai import nodes


def test_rms_norm():
    check_node(
        lambda: nodes.RmsNormNode(eps=1e-5),
        {"in0": (4, 64)},
        lambda in0, scale: F.rms_norm(in0, (64,), scale, 1e-5),
        params={"scale": (64,)},
        alt_inputs={"in0": (9, 64)},
    )
