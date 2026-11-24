import numpy as np
import pytest
import torch.nn.functional as F
from node_check import ACTIVATIONS, FP16, check_node, normal, uniform

import pyvai
from pyvai import nodes


# kernel_size and stride pick the plan: 1x1, 3x3 s1 (implicit GEMM or Winograd
# by size and device), 3x3 s2, or the general implicit GEMM.
@pytest.mark.parametrize(
    "ci, co, k, stride, hw, activation, has_scale",
    [
        pytest.param(16, 32, 1, 1, 13, "NONE", False, id="k1s1"),
        pytest.param(8, 16, 3, 1, 12, "RELU", False, id="k3s1_relu"),
        pytest.param(32, 32, 3, 1, 28, "NONE", False, id="k3s1_large"),
        pytest.param(8, 16, 3, 2, 15, "NONE", False, id="k3s2"),
        pytest.param(4, 8, 5, 1, 9, "NONE", False, id="k5s1"),
        pytest.param(8, 8, 3, 1, 10, "NONE", True, id="k3s1_scaled"),
    ],
)
def test_convolution(ci, co, k, stride, hw, activation, has_scale):
    params = {"weight": normal((co, ci, k, k), 1.0 / np.sqrt(ci * k * k)), "bias": (co,)}
    if has_scale:
        params["scale"] = uniform((co,), 0.5, 1.5)

    # Same padding over [C, H, W]: out0 = activation(scale * conv + bias). `scale`
    # applies when the test sets it.
    def ref(in0, weight, bias, scale=None):
        out = F.conv2d(in0[None], weight, stride=stride, padding=k // 2)[0]
        if scale is not None:
            out = out * scale[:, None, None]
        return ACTIVATIONS[activation](out + bias[:, None, None])

    check_node(
        lambda: nodes.ConvolutionNode(
            in_channels=ci,
            out_channels=co,
            kernel_size=k,
            stride=stride,
            act_type=getattr(pyvai.ActType, activation),
        ),
        {"in0": (ci, hw, hw)},
        ref,
        params=params,
        tol=FP16,
    )


@pytest.mark.parametrize(
    "k, order",
    [
        pytest.param(3, "AFTER_ACTIVATION", id="k3s1_after"),
        pytest.param(3, "BEFORE_ACTIVATION", id="k3s1_before"),
        pytest.param(1, "BEFORE_ACTIVATION", id="k1s1_before"),
    ],
)
def test_convolution_residual(k, order):
    c, hw = 8, 10

    def ref(in0, residual, weight, bias, scale):
        out = F.conv2d(in0[None], weight, padding=k // 2)[0]
        out = out * scale[:, None, None] + bias[:, None, None]
        if order == "BEFORE_ACTIVATION":
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
            act_type=pyvai.ActType.RELU,
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


# fuse_max_pool: out0 is the 3x3/2 pad-1 max pool of the activation. The fused
# rules take fp16 weights, so here the pool runs as a pass after the
# convolution's own plan, the general one or a 3x3/1 path. An odd side ends on
# a window that the right padding clips.
@pytest.mark.parametrize(
    "ci, co, k, stride, hw",
    [
        pytest.param(3, 16, 7, 2, 32, id="k7s2"),
        pytest.param(3, 16, 7, 2, 29, id="k7s2_odd"),
        pytest.param(8, 16, 3, 1, 13, id="k3s1_odd"),
    ],
)
def test_convolution_fuse_max_pool(ci, co, k, stride, hw):
    def ref(in0, weight, bias, scale):
        out = F.conv2d(in0[None], weight, stride=stride, padding=k // 2)
        out = F.relu(out * scale[None, :, None, None] + bias[None, :, None, None])
        return F.max_pool2d(out, 3, stride=2, padding=1)[0]

    check_node(
        lambda: nodes.ConvolutionNode(
            in_channels=ci,
            out_channels=co,
            kernel_size=k,
            stride=stride,
            act_type=pyvai.ActType.RELU,
            fuse_max_pool=True,
        ),
        {"in0": (ci, hw, hw)},
        ref,
        params={
            "weight": normal((co, ci, k, k), 1.0 / np.sqrt(ci * k * k)),
            "bias": (co,),
            "scale": uniform((co,), 0.5, 1.5),
        },
        tol=FP16,
    )
