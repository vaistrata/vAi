"""Numeric specs for the core nodes (vai/nodes/).

A spec names a node, how to build it, the shapes of its inputs and parameters,
and its math as a reference (node_refs.py, mostly PyTorch). test_node_numerics.py turns
every spec into the same checks: the output against the reference, a second
forward giving the identical output, no NaN or Inf, and, for specs that list
alt_inputs, the same net rerun on new shapes. test_node_completeness.py fails
when a bound node has no spec.

Adding a node: bind it in python/bindings/nodes/<name>.cpp, then add at least
one spec(...) for it below.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import node_refs as ref
import numpy as np

import pyvai
from pyvai import nodes

# (rtol, atol). Pick by what the kernel does, not by what passes.
EXACT = (0.0, 0.0)  # copies, layout changes, max, argmax, casts
FP32 = (1e-5, 1e-5)  # fp32 arithmetic and reductions
TRANS = (1e-4, 1e-5)  # exp / tanh / erf in fp32
# Kernels that take a cooperative-matrix or Winograd path on devices that have
# one (Linear's tile path, Convolution) read fp32 operands as fp16. A wrong
# index or tail still misses by far more than this.
FP16 = (1e-3, 5e-3)


@dataclass
class Spec:
    node: type  # the bound class; its name is what completeness counts
    label: str
    make: Callable[[], object]  # builds the node inside the net
    inputs: Dict[str, object]  # slot -> shape (standard normal) or fn(rng) -> array
    ref: Callable[..., np.ndarray]  # ref(**inputs, **params) -> expected out0
    params: Dict[str, object] = field(default_factory=dict)
    tol: Tuple[float, float] = FP32
    out: str = "f32"  # out0 read back as "f32", "f16" or "u32"
    alt_inputs: Optional[Dict[str, object]] = None  # rerun shapes, same params
    alt_tol: Optional[Tuple[float, float]] = None  # None: tol; set when the rerun
    # takes another kernel, e.g. Linear crossing from slab to tile

    @property
    def id(self):
        return f"{self.node.__name__}-{self.label}"


SPECS = []
COVERED = set()


def spec(node, label, make, inputs, ref, **kwargs):
    s = Spec(node, label, make, inputs, ref, **kwargs)
    SPECS.append(s)
    COVERED.add(node.__name__)
    return s


def normal(shape, scale=1.0):
    return lambda rng: scale * rng.standard_normal(shape)


def uniform(shape, low, high):
    return lambda rng: rng.uniform(low, high, shape)


def _draw(gen, rng):
    array = gen(rng) if callable(gen) else rng.standard_normal(gen)
    return np.ascontiguousarray(array, dtype=np.float32)


def _tensor(array):
    return pyvai.Tensor.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


class Run:
    """One spec's net with drawn parameters, fed and read back."""

    def __init__(self, s, rng):
        self.spec = s
        self.params = {k: _draw(g, rng) for k, g in s.params.items()}
        slots = list(s.inputs)

        class Net(pyvai.NeuralNet):
            def __init__(self):
                super().__init__(w_default_dtype=pyvai.Dtype.F32)
                self.node = s.make()
                self.input(slots[0]) - slots[0] / self.node - self.output("out0")
                for slot in slots[1:]:
                    self.input(slot) - slot / self.node

        self.net = Net()
        for slot, array in self.params.items():
            self.net.node[slot] = _tensor(array)
        self.draw(s.inputs, rng)

    def draw(self, inputs, rng):
        self.inputs = {k: _draw(g, rng) for k, g in inputs.items()}

    def forward(self):
        out = self.net(**{k: _tensor(a) for k, a in self.inputs.items()})
        return out.to_u32() if self.spec.out == "u32" else out.to_numpy()

    def expected(self):
        args = {k: a.astype(np.float64) for k, a in {**self.inputs, **self.params}.items()}
        return self.spec.ref(**args)


# ---------------------------------------------------------------------------
# Element-wise
# ---------------------------------------------------------------------------

spec(
    nodes.AddNode,
    "same_shape",
    nodes.AddNode,
    {"in0": (2, 3, 17), "in1": (2, 3, 17)},
    lambda in0, in1: in0 + in1,
    alt_inputs={"in0": (5, 64), "in1": (5, 64)},
)

spec(
    nodes.MulNode,
    "same_shape",
    nodes.MulNode,
    {"in0": (2, 3, 17), "in1": (2, 3, 17)},
    lambda in0, in1: in0 * in1,
)

for _ep in ("RELU", "GELU", "GELU_TANH", "SILU", "SIGMOID"):
    spec(
        nodes.ActivationNode,
        _ep.lower(),
        lambda ep=_ep: nodes.ActivationNode(epilogue=getattr(pyvai.Epilogue, ep)),
        {"in0": normal((5, 33), 3.0)},
        lambda in0, ep=_ep: ref.activation(in0, ep),
        tol=TRANS,
    )

for _ep in ("SILU", "RELU", "GELU_TANH"):
    spec(
        nodes.GatedMulNode,
        _ep.lower(),
        lambda ep=_ep: nodes.GatedMulNode(activation=getattr(pyvai.Epilogue, ep)),
        {"in0": (4, 48)},
        lambda in0, ep=_ep: ref.gated_mul(in0, ep),
        tol=TRANS,
    )

spec(
    nodes.EleAffineNode,
    "inputs_broadcast_row",
    nodes.EleAffineNode,
    {"in0": (6, 32), "scale": (32,), "shift": (6, 32)},
    lambda in0, scale, shift: scale * in0 + shift,
)

spec(
    nodes.EleAffineNode,
    "params_per_channel",
    lambda: nodes.EleAffineNode(scale_as_input=False, shift_as_input=False),
    {"in0": (4, 5, 6)},
    lambda in0, scale, shift: scale * in0 + shift,
    params={"scale": (4, 1, 1), "shift": (4, 1, 1)},
)

spec(
    nodes.TypeCastNode,
    "f32_to_f16",
    lambda: nodes.TypeCastNode(o_dtype=pyvai.Dtype.F16),
    {"in0": normal((3, 40), 10.0)},
    lambda in0: in0.astype(np.float16),
    tol=EXACT,
    out="f16",
)


# ---------------------------------------------------------------------------
# Linear / normalization
# ---------------------------------------------------------------------------


def _linear(m, k, n, label, tol, bias=False, epilogue="NONE", split_size=0, alt_m=None):
    """M <= 8 takes the slab kernel, larger M the tile kernel (linear.h)."""
    params = {"weight": normal((n, k), 1.0 / np.sqrt(k))}
    if bias:
        params["bias"] = (n,)
    spec(
        nodes.LinearNode,
        label,
        lambda: nodes.LinearNode(
            in_dim=k,
            out_dim=n,
            has_bias=bias,
            epilogue=getattr(pyvai.Epilogue, epilogue),
            split_size=split_size,
        ),
        {"in0": (m, k)},
        lambda in0, weight, bias=None: ref.linear(in0, weight, bias, epilogue),
        params=params,
        tol=tol,
        alt_inputs={"in0": (alt_m, k)} if alt_m else None,
        alt_tol=FP16 if alt_m and alt_m > 8 else None,
    )


_linear(1, 64, 32, "slab_m1", FP32, bias=True, alt_m=33)  # rerun crosses slab -> tile
_linear(8, 96, 40, "slab_m8", FP32)
_linear(33, 96, 48, "tile_relu", FP16, bias=True, epilogue="RELU")
_linear(64, 128, 100, "tile_odd_n", FP16, epilogue="GELU_TANH")
_linear(16, 256, 64, "tile_split_k", FP16, bias=True, split_size=64)

spec(
    nodes.RmsNormNode,
    "rows",
    lambda: nodes.RmsNormNode(eps=1e-5),
    {"in0": (4, 64)},
    lambda in0, scale: ref.rms_norm(in0, scale),
    params={"scale": (64,)},
    alt_inputs={"in0": (9, 64)},
)

spec(
    nodes.LayerNormNode,
    "with_beta",
    lambda: nodes.LayerNormNode(eps=1e-5),
    {"in0": (4, 96)},
    lambda in0, gamma, beta: ref.layer_norm(in0, gamma, beta),
    params={"gamma": (96,), "beta": (96,)},
    alt_inputs={"in0": (7, 96)},
)

spec(
    nodes.LayerNormNode,
    "no_beta",
    lambda: nodes.LayerNormNode(eps=1e-5, has_beta=False),
    {"in0": (3, 50)},
    lambda in0, gamma: ref.layer_norm(in0, gamma),
    params={"gamma": (50,)},
)

spec(
    nodes.BatchNormNode,
    "chw",
    lambda: nodes.BatchNormNode(channels=8),
    {"in0": (8, 6, 6)},
    ref.batch_norm,
    params={"gamma": (8,), "beta": (8,), "mean": (8,), "variance": uniform((8,), 0.5, 2.0)},
)


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------


def _conv(ci, co, k, stride, hw, label, epilogue="NONE", has_scale=False):
    """kernel_size / stride pick the plan: 1x1, 3x3 s1 (implicit GEMM or
    Winograd by size and device), 3x3 s2, or the general implicit GEMM."""
    params = {"weight": normal((co, ci, k, k), 1.0 / np.sqrt(ci * k * k)), "bias": (co,)}
    if has_scale:
        params["scale"] = uniform((co,), 0.5, 1.5)
    spec(
        nodes.ConvolutionNode,
        label,
        lambda: nodes.ConvolutionNode(
            in_channels=ci,
            out_channels=co,
            kernel_size=k,
            stride=stride,
            has_scale=has_scale,
            epilogue=getattr(pyvai.Epilogue, epilogue),
        ),
        {"in0": (ci, hw, hw)},
        lambda in0, weight, bias, scale=None: ref.conv2d(
            in0, weight, bias, scale, stride, epilogue
        ),
        params=params,
        tol=FP16,
    )


_conv(16, 32, 1, 1, 13, "k1s1")
_conv(8, 16, 3, 1, 12, "k3s1_relu", epilogue="RELU")
_conv(32, 32, 3, 1, 28, "k3s1_large")
_conv(8, 16, 3, 2, 15, "k3s2")
_conv(4, 8, 5, 1, 9, "k5s1")
_conv(8, 8, 3, 1, 10, "k3s1_scaled", has_scale=True)

spec(
    nodes.MaxPool2dNode,
    "k3s2p1",
    lambda: nodes.MaxPool2dNode(kernel_size=3, stride=2, padding=1),
    {"in0": (4, 11, 11)},
    lambda in0: ref.pool2d(in0, 3, 2, 1, "max"),
    tol=EXACT,
)

spec(
    nodes.MaxPool2dNode,
    "k2s2",
    lambda: nodes.MaxPool2dNode(kernel_size=2, stride=2),
    {"in0": (3, 8, 8)},
    lambda in0: ref.pool2d(in0, 2, 2, 0, "max"),
    tol=EXACT,
)

spec(
    nodes.AvgPool2dNode,
    "k2s2",
    lambda: nodes.AvgPool2dNode(kernel_size=2, stride=2),
    {"in0": (3, 8, 8)},
    lambda in0: ref.pool2d(in0, 2, 2, 0, "avg"),
)

spec(
    nodes.AvgPool2dNode,
    "k3s1p1",
    lambda: nodes.AvgPool2dNode(kernel_size=3, stride=1, padding=1),
    {"in0": (4, 7, 7)},
    lambda in0: ref.pool2d(in0, 3, 1, 1, "avg"),
)

for _s in (2, 3):
    spec(
        nodes.NearestUpsampleNode,
        f"x{_s}",
        lambda s=_s: nodes.NearestUpsampleNode(scale=s),
        {"in0": (3, 5, 7)},
        lambda in0, s=_s: ref.nearest_upsample(in0, s),
        tol=EXACT,
    )


# ---------------------------------------------------------------------------
# Shape / plumbing / output heads
# ---------------------------------------------------------------------------

spec(nodes.IdentityNode, "pass", nodes.IdentityNode, {"in0": (3, 4)}, lambda in0: in0, tol=EXACT)

spec(
    nodes.FlattenNode,
    "rank3",
    nodes.FlattenNode,
    {"in0": (2, 3, 4)},
    lambda in0: in0.ravel(),
    tol=EXACT,
)

spec(
    nodes.ReshapeNode,
    "infer",
    lambda: nodes.ReshapeNode(dims=[4, -1]),
    {"in0": (2, 3, 4)},
    lambda in0: in0.reshape(4, -1),
    tol=EXACT,
)

spec(
    nodes.ReshapeNode,
    "keep_dim0",
    lambda: nodes.ReshapeNode(dims=[0, -1]),
    {"in0": (2, 3, 4)},
    lambda in0: in0.reshape(in0.shape[0], -1),
    tol=EXACT,
)

for _perms, _shape in (
    ([2, 0, 1], (3, 4, 5)),
    ([0, 2, 1, 3], (2, 3, 4, 5)),
    ([1, 0, 2], (1, 4, 5)),
):
    spec(
        nodes.PermuteNode,
        "p" + "".join(map(str, _perms)),
        lambda p=_perms: nodes.PermuteNode(perms=p),
        {"in0": _shape},
        lambda in0, p=_perms: in0.transpose(p),
        tol=EXACT,
    )

spec(
    nodes.SliceNode,
    "axis0_block",
    lambda: nodes.SliceNode(start=1, count=2, axis=0),
    {"in0": (4, 5)},
    lambda in0: in0[1:3],
    tol=EXACT,
)

spec(
    nodes.SliceNode,
    "axis1_from_end",
    lambda: nodes.SliceNode(start=-3, count=2, axis=1),
    {"in0": (3, 6, 2)},
    lambda in0: in0[:, 3:5],
    tol=EXACT,
)

spec(
    nodes.ConcatNode,
    "axis0_copy",
    lambda: nodes.ConcatNode(axis=0),
    {"in0": (2, 5), "in1": (3, 5)},
    lambda in0, in1: np.concatenate([in0, in1], 0),
    tol=EXACT,
)

spec(
    nodes.ConcatNode,
    "axis1_gather",
    lambda: nodes.ConcatNode(axis=1),
    {"in0": (2, 3, 4), "in1": (2, 5, 4)},
    lambda in0, in1: np.concatenate([in0, in1], 1),
    tol=EXACT,
)

spec(
    nodes.ConcatNode,
    "three_last_axis",
    lambda: nodes.ConcatNode(axis=-1, input_count=3),
    {"in0": (2, 3), "in1": (2, 4), "in2": (2, 1)},
    lambda in0, in1, in2: np.concatenate([in0, in1, in2], -1),
    tol=EXACT,
)

spec(
    nodes.ArgmaxNode,
    "rows",
    nodes.ArgmaxNode,
    {"in0": (4, 50)},
    lambda in0: in0.argmax(axis=-1),
    tol=EXACT,
    out="u32",
)
