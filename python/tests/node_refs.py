"""numpy references for the core nodes, worked in float64.

Each function states the node's math as plainly as numpy allows; node_cases.py
names them in its specs. They take and return numpy arrays.
"""

import math

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


_erf = np.vectorize(math.erf, otypes=[np.float64])

# The kernels' epilogues (vai/shaders/common/activation.h).
EPILOGUES = {
    "NONE": lambda x: x,
    "RELU": lambda x: np.maximum(x, 0.0),
    "GELU": lambda x: 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0))),
    "GELU_TANH": lambda x: (
        0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
    ),
    "SILU": lambda x: x * sigmoid(x),
    "SIGMOID": sigmoid,
}


def linear(in0, weight, bias=None, epilogue="NONE"):
    """out0 = epilogue(in0 W^T + bias); weight is [N, K]."""
    out = in0 @ weight.T
    if bias is not None:
        out = out + bias
    return EPILOGUES[epilogue](out)


def conv2d(in0, weight, bias=None, scale=None, stride=1, epilogue="NONE"):
    """Same-padded 2D convolution of [Ci, H, W] by [Co, Ci, K, K] (K odd):
    out0 [Co, ceil(H / stride), ceil(W / stride)] = epilogue(scale * conv + bias)."""
    co, _, k, _ = weight.shape
    _, h, w = in0.shape
    pad = (k - 1) // 2
    ho, wo = -(-h // stride), -(-w // stride)
    xp = np.pad(in0, ((0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((co, ho, wo))
    for ky in range(k):
        for kx in range(k):
            patch = xp[
                :,
                ky : ky + stride * (ho - 1) + 1 : stride,
                kx : kx + stride * (wo - 1) + 1 : stride,
            ]
            out += np.einsum("oc,chw->ohw", weight[:, :, ky, kx], patch)
    if scale is not None:
        out = out * scale[:, None, None]
    if bias is not None:
        out = out + bias[:, None, None]
    return EPILOGUES[epilogue](out)


def pool2d(in0, kernel, stride, padding, mode, count_include_pad=True):
    """Square-window pooling over the last two axes of [..., C, H, W].

    Max ignores samples outside the input. Avg pads with zeros and, with
    count_include_pad, divides by the full window."""
    h, w = in0.shape[-2:]
    ho = (h + 2 * padding - kernel) // stride + 1
    wo = (w + 2 * padding - kernel) // stride + 1
    pads = [(0, 0)] * (in0.ndim - 2) + [(padding, padding), (padding, padding)]
    fill = -np.inf if mode == "max" else 0.0
    xp = np.pad(in0, pads, constant_values=fill)
    ones = np.pad(np.ones(in0.shape), pads)
    windows, counts = [], []
    for ky in range(kernel):
        for kx in range(kernel):
            rows = slice(ky, ky + stride * (ho - 1) + 1, stride)
            cols = slice(kx, kx + stride * (wo - 1) + 1, stride)
            windows.append(xp[..., rows, cols])
            counts.append(ones[..., rows, cols])
    windows = np.stack(windows)
    if mode == "max":
        return windows.max(axis=0)
    divisor = kernel * kernel if count_include_pad else np.stack(counts).sum(axis=0)
    return windows.sum(axis=0) / divisor


def batch_norm(in0, gamma, beta, mean, variance, eps=1e-5):
    """Inference batch norm over [C, H, W] with per-channel statistics."""
    c = (slice(None), None, None)
    return gamma[c] * (in0 - mean[c]) / np.sqrt(variance[c] + eps) + beta[c]


def layer_norm(in0, gamma, beta=None, eps=1e-5):
    mean = in0.mean(axis=-1, keepdims=True)
    var = in0.var(axis=-1, keepdims=True)
    out = (in0 - mean) / np.sqrt(var + eps) * gamma
    return out if beta is None else out + beta


def rms_norm(in0, scale, eps=1e-5):
    return in0 / np.sqrt(np.mean(in0**2, axis=-1, keepdims=True) + eps) * scale


def gated_mul(in0, activation="SILU"):
    """act(first half) * second half along the last axis."""
    half = in0.shape[-1] // 2
    return EPILOGUES[activation](in0[..., :half]) * in0[..., half:]


def nearest_upsample(in0, scale):
    """Replicate every [C, H, W] pixel into a scale x scale block."""
    return in0.repeat(scale, axis=-2).repeat(scale, axis=-1)
