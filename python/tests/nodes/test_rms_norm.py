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


def test_rms_norm_scale_shift_residual():
    check_node(
        lambda: nodes.RmsNormNode(eps=1e-5, accept_residual=True),
        {"in0": (5, 64), "residual": (5, 64)},
        lambda in0, residual, scale, shift: F.rms_norm(in0, (64,), scale, 1e-5) + shift + residual,
        params={"scale": (64,), "shift": (64,)},
    )


def test_rms_norm_coeffs_as_inputs():
    check_node(
        lambda: nodes.RmsNormNode(eps=1e-5, coeffs_as_inputs=True),
        {"in0": (5, 64), "scale": (64,)},
        lambda in0, scale: F.rms_norm(in0, (64,), scale, 1e-5),
    )
