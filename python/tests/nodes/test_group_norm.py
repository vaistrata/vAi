import torch.nn.functional as F
from node_check import EPILOGUES, check_node

import pyvai
from pyvai import nodes


def test_group_norm_with_shift():
    # [C, H, W] is one sample; torch takes [N, C, H, W].
    check_node(
        lambda: nodes.GroupNormNode(groups=4, eps=1e-5),
        {"in0": (16, 6, 7)},
        lambda in0, scale, shift: F.group_norm(in0[None], 4, scale, shift, 1e-5)[0],
        params={"scale": (16,), "shift": (16,)},
        alt_inputs={"in0": (16, 9, 5)},
    )


def test_group_norm_batched_silu_no_shift():
    check_node(
        lambda: nodes.GroupNormNode(groups=2, eps=1e-5, epilogue=pyvai.Epilogue.SILU),
        {"in0": (3, 8, 5, 5)},
        lambda in0, scale: EPILOGUES["SILU"](F.group_norm(in0, 2, scale, None, 1e-5)),
        params={"scale": (8,)},
    )
