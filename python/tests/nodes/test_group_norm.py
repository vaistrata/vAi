import torch.nn.functional as F
from node_check import EPILOGUES, check_node

import pyvai
from pyvai import nodes


def test_group_norm():
    check_node(
        lambda: nodes.GroupNormNode(groups=4),
        {"in0": (16, 6, 6)},
        lambda in0, scale, shift: F.group_norm(in0[None], 4, scale, shift, 1e-5)[0],
        params={"scale": (16,), "shift": (16,)},
        alt_inputs={"in0": (16, 9, 5)},
    )


def test_group_norm_no_shift_batched():
    check_node(
        lambda: nodes.GroupNormNode(groups=2, eps=1e-3),
        {"in0": (3, 8, 5, 5)},
        lambda in0, scale: F.group_norm(in0, 2, scale, None, 1e-3),
        params={"scale": (8,)},
    )


def test_group_norm_epilogue_residual():
    check_node(
        lambda: nodes.GroupNormNode(
            groups=4,
            accept_residual=True,
            residual_order=pyvai.ResidualOrder.BEFORE_EPILOGUE,
            epilogue=pyvai.Epilogue.RELU,
        ),
        {"in0": (8, 4, 4), "residual": (8, 4, 4)},
        lambda in0, residual, scale, shift: EPILOGUES["RELU"](
            F.group_norm(in0[None], 4, scale, shift, 1e-5)[0] + residual
        ),
        params={"scale": (8,), "shift": (8,)},
    )
