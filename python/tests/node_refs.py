"""References for the core nodes, computed by PyTorch in float64 on the CPU.

The operator math comes from torch.nn.functional rather than formulas written
here; these functions only adapt vAi's conventions (no batch axis on images,
same padding for odd kernels, out0 = scale * conv + bias) to PyTorch's. They
take and return numpy arrays. PyTorch is a test dependency only (the [test]
extra); pyvai itself never imports it.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _t(array):
    return torch.from_numpy(np.asarray(array, dtype=np.float64))


def _np(tensor):
    return tensor.numpy()


# The kernels' epilogues (vai/shaders/common/activation.h).
_EPILOGUES = {
    "NONE": lambda x: x,
    "RELU": F.relu,
    "GELU": F.gelu,
    "GELU_TANH": lambda x: F.gelu(x, approximate="tanh"),
    "SILU": F.silu,
    "SIGMOID": torch.sigmoid,
}


def activation(in0, epilogue):
    return _np(_EPILOGUES[epilogue](_t(in0)))


def linear(in0, weight, bias=None, epilogue="NONE"):
    """out0 = epilogue(in0 W^T + bias); weight is [N, K]."""
    out = F.linear(_t(in0), _t(weight), None if bias is None else _t(bias))
    return _np(_EPILOGUES[epilogue](out))


def conv2d(in0, weight, bias=None, scale=None, stride=1, epilogue="NONE"):
    """[Ci, H, W] by [Co, Ci, K, K] with same padding (K odd):
    out0 = epilogue(scale * conv + bias), [Co, ceil(H / stride), ceil(W / stride)]."""
    k = weight.shape[-1]
    out = F.conv2d(_t(in0)[None], _t(weight), stride=stride, padding=k // 2)[0]
    if scale is not None:
        out = out * _t(scale)[:, None, None]
    if bias is not None:
        out = out + _t(bias)[:, None, None]
    return _np(_EPILOGUES[epilogue](out))


def pool2d(in0, kernel, stride, padding, mode):
    """Square-window pooling over the last two axes. Max ignores samples
    outside the input; Avg counts the zero padding in the divisor."""
    x = _t(in0)
    if mode == "max":
        return _np(F.max_pool2d(x, kernel, stride, padding))
    return _np(F.avg_pool2d(x, kernel, stride, padding, count_include_pad=True))


def batch_norm(in0, gamma, beta, mean, variance, eps=1e-5):
    """Inference batch norm over [C, H, W]."""
    out = F.batch_norm(
        _t(in0)[None], _t(mean), _t(variance), _t(gamma), _t(beta), training=False, eps=eps
    )
    return _np(out[0])


def layer_norm(in0, gamma, beta=None, eps=1e-5):
    shape = (in0.shape[-1],)
    return _np(F.layer_norm(_t(in0), shape, _t(gamma), None if beta is None else _t(beta), eps))


def rms_norm(in0, scale, eps=1e-5):
    return _np(F.rms_norm(_t(in0), (in0.shape[-1],), _t(scale), eps))


def gated_mul(in0, activation="SILU"):
    """act(first half) * second half along the last axis."""
    gate, up = _t(in0).chunk(2, dim=-1)
    return _np(_EPILOGUES[activation](gate) * up)


def nearest_upsample(in0, scale):
    """Replicate every [C, H, W] pixel into a scale x scale block."""
    return _np(F.interpolate(_t(in0)[None], scale_factor=scale, mode="nearest")[0])
