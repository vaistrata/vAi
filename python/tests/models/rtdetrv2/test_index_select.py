import numpy as np
import torch
from node_check import EXACT, check_node

from pyvai._pyvai import model_nodes

rtdetrv2 = model_nodes.rtdetrv2


def _indices(count, high):
    """uint32 indices as the raw bits of fp32 slots, as TopKNode writes them.
    Some reach past the rows (high > N) and repeat."""
    return lambda rng: rng.integers(0, high, count).astype(np.uint32).view(np.float32)


def _ref(in0, indices):
    rows = indices.to(torch.float32).view(torch.int32).long()
    valid = rows < in0.shape[0]
    out = in0[rows.clamp(max=in0.shape[0] - 1)]
    out[~valid] = 0
    return out


def test_index_select():
    check_node(
        lambda: rtdetrv2.IndexSelectNode(),
        {"in0": (10, 3, 5), "indices": _indices(23, 12)},
        _ref,
        tol=EXACT,
        alt_inputs={"in0": (4, 7), "indices": _indices(9, 6)},
    )


def test_index_select_constant_input():
    """Query selection's anchor table: in0 is a parameter, indices the only input."""
    check_node(
        lambda: rtdetrv2.IndexSelectNode(constant_input=True),
        {"indices": _indices(30, 40)},
        _ref,
        params={"in0": (37, 4)},
        tol=EXACT,
    )
