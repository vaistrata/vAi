import numpy as np
import pytest
import torch.nn.functional as F
from node_check import EPILOGUES, FP16, check_node, normal, uniform

import pyvai
from pyvai import nodes


# kernel_size and stride pick the plan: 1x1, 3x3 s1 (implicit GEMM or Winograd
# by size and device), 3x3 s2, or the general implicit GEMM.
@pytest.mark.parametrize(
    "ci, co, k, stride, hw, epilogue, has_scale",
    [
        pytest.param(16, 32, 1, 1, 13, "NONE", False, id="k1s1"),
        pytest.param(8, 16, 3, 1, 12, "RELU", False, id="k3s1_relu"),
        pytest.param(32, 32, 3, 1, 28, "NONE", False, id="k3s1_large"),
        pytest.param(8, 16, 3, 2, 15, "NONE", False, id="k3s2"),
        pytest.param(4, 8, 5, 1, 9, "NONE", False, id="k5s1"),
        pytest.param(8, 8, 3, 1, 10, "NONE", True, id="k3s1_scaled"),
    ],
)
def test_convolution(ci, co, k, stride, hw, epilogue, has_scale):
    params = {"weight": normal((co, ci, k, k), 1.0 / np.sqrt(ci * k * k)), "bias": (co,)}
    if has_scale:
        params["scale"] = uniform((co,), 0.5, 1.5)

    # Same padding over [C, H, W]: out0 = epilogue(scale * conv + bias). `scale`
    # applies when the test sets it.
    def ref(in0, weight, bias, scale=None):
        out = F.conv2d(in0[None], weight, stride=stride, padding=k // 2)[0]
        if scale is not None:
            out = out * scale[:, None, None]
        return EPILOGUES[epilogue](out + bias[:, None, None])

    check_node(
        lambda: nodes.ConvolutionNode(
            in_channels=ci,
            out_channels=co,
            kernel_size=k,
            stride=stride,
            epilogue=getattr(pyvai.Epilogue, epilogue),
        ),
        {"in0": (ci, hw, hw)},
        ref,
        params=params,
        tol=FP16,
    )


@pytest.mark.parametrize(
    "k, order",
    [
        pytest.param(3, "AFTER_EPILOGUE", id="k3s1_after"),
        pytest.param(3, "BEFORE_EPILOGUE", id="k3s1_before"),
        pytest.param(1, "BEFORE_EPILOGUE", id="k1s1_before"),
    ],
)
def test_convolution_residual(k, order):
    c, hw = 8, 10

    def ref(in0, residual, weight, bias, scale):
        out = F.conv2d(in0[None], weight, padding=k // 2)[0]
        out = out * scale[:, None, None] + bias[:, None, None]
        if order == "BEFORE_EPILOGUE":
            return F.relu(out + residual)
        return F.relu(out) + residual

    check_node(
        lambda: nodes.ConvolutionNode(
            in_channels=c,
            out_channels=c,
            kernel_size=k,
            stride=1,
            accept_residual=True,
            residual_order=getattr(pyvai.ResidualOrder, order),
            epilogue=pyvai.Epilogue.RELU,
        ),
        {"in0": (c, hw, hw), "residual": (c, hw, hw)},
        ref,
        params={
            "weight": normal((c, c, k, k), 1.0 / np.sqrt(c * k * k)),
            "bias": (c,),
            "scale": uniform((c,), 0.5, 1.5),
        },
        tol=FP16,
    )
