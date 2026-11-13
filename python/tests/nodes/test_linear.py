import numpy as np
import pytest
import torch.nn.functional as F
from node_check import EPILOGUES, FP16, FP32, check_node, normal

import pyvai
from pyvai import nodes


# M <= 8 takes the slab kernel, larger M the tile kernel (linear.h).
@pytest.mark.parametrize(
    "m, k, n, has_bias, epilogue, split_size, tol, alt_m",
    [
        pytest.param(1, 64, 32, True, "NONE", 0, FP32, 33, id="slab_m1"),  # rerun: slab -> tile
        pytest.param(8, 96, 40, False, "NONE", 0, FP32, None, id="slab_m8"),
        pytest.param(33, 96, 48, True, "RELU", 0, FP16, None, id="tile_relu"),
        pytest.param(64, 128, 100, False, "GELU_TANH", 0, FP16, None, id="tile_odd_n"),
        pytest.param(16, 256, 64, True, "NONE", 64, FP16, None, id="tile_split_k"),
    ],
)
def test_linear(m, k, n, has_bias, epilogue, split_size, tol, alt_m):
    params = {"weight": normal((n, k), 1.0 / np.sqrt(k))}
    if has_bias:
        params["bias"] = (n,)

    def ref(in0, weight, bias=None):
        return EPILOGUES[epilogue](F.linear(in0, weight, bias))

    check_node(
        lambda: nodes.LinearNode(
            in_dim=k,
            out_dim=n,
            epilogue=getattr(pyvai.Epilogue, epilogue),
            split_size=split_size,
        ),
        {"in0": (m, k)},
        ref,
        params=params,
        tol=tol,
        alt_inputs={"in0": (alt_m, k)} if alt_m else None,
        alt_tol=FP16 if alt_m and alt_m > 8 else None,
    )


# The residual joins after the epilogue (a following add) or before it.
@pytest.mark.parametrize(
    "m, order, tol",
    [
        pytest.param(4, "AFTER_EPILOGUE", FP32, id="slab_after"),
        pytest.param(4, "BEFORE_EPILOGUE", FP32, id="slab_before"),
        pytest.param(33, "AFTER_EPILOGUE", FP16, id="tile_after"),
        pytest.param(33, "BEFORE_EPILOGUE", FP16, id="tile_before"),
    ],
)
def test_linear_residual(m, order, tol):
    k, n = 64, 48

    def ref(in0, residual, weight, bias):
        out = F.linear(in0, weight, bias)
        if order == "BEFORE_EPILOGUE":
            return F.relu(out + residual)
        return F.relu(out) + residual

    check_node(
        lambda: nodes.LinearNode(
            in_dim=k,
            out_dim=n,
            accept_residual=True,
            residual_order=getattr(pyvai.ResidualOrder, order),
            epilogue=pyvai.Epilogue.RELU,
        ),
        {"in0": (m, k), "residual": (m, n)},
        ref,
        params={"weight": normal((n, k), 1.0 / np.sqrt(k)), "bias": (n,)},
        tol=tol,
    )
