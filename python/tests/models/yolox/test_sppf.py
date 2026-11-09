import torch
import torch.nn.functional as F
from node_check import EXACT, check_node

from pyvai._pyvai import model_nodes

yolox = model_nodes.yolox


def _ref(in0):
    """[C, H, W] -> [4C, H, W]: in0 and three chained 5x5 stride-1 max pools."""
    stages = [in0]
    for _ in range(3):
        stages.append(F.max_pool2d(stages[-1], 5, 1, 2))
    return torch.cat(stages)


def test_sppf():
    check_node(
        yolox.SPPFNode,
        {"in0": (4, 13, 13)},
        _ref,
        tol=EXACT,
        alt_inputs={"in0": (3, 32, 32)},  # rerun at the H * W = 1024 limit
    )
