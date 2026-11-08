"""check_node(), the numeric check every node test calls.

check_node() builds a one-node net, draws the node's inputs and parameters once
(seeded), and gives the same arrays to the node and, as float64 torch tensors on
the CPU, to the reference. Then it checks, in order:

1. out0 matches the reference within the tolerance tier;
2. a second forward, which replays the recorded commands, gives the identical out0;
3. out0 has no NaN or Inf;
4. with alt_inputs, a rerun on those shapes, which re-prepares the node, matches too.

torch is a test dependency only (the [test] extra); pyvai never imports it.
"""

import numpy as np
import torch
import torch.nn.functional as F

import pyvai

# (rtol, atol). Pick by what the kernel does, not by what passes.
EXACT = (0.0, 0.0)  # copies, layout changes, max, argmax, casts
FP32 = (1e-5, 1e-5)  # fp32 arithmetic and reductions
TRANS = (1e-4, 1e-5)  # exp / tanh / erf in fp32
# Kernels that take a cooperative-matrix or Winograd path on devices that have
# one (Linear's tile path, Convolution) read fp32 operands as fp16. A wrong
# index or tail still misses by far more than this.
FP16 = (1e-3, 5e-3)

# The kernels' epilogues (vai/shaders/common/activation.h), by pyvai.Epilogue name.
EPILOGUES = {
    "NONE": lambda x: x,
    "RELU": F.relu,
    "GELU": F.gelu,
    "GELU_TANH": lambda x: F.gelu(x, approximate="tanh"),
    "SILU": F.silu,
    "SIGMOID": torch.sigmoid,
}


def normal(shape, scale=1.0):
    """An input or parameter drawn as scale * standard normal."""
    return lambda rng: scale * rng.standard_normal(shape)


def uniform(shape, low, high):
    """An input or parameter drawn uniformly from [low, high)."""
    return lambda rng: rng.uniform(low, high, shape)


def check_node(
    make,
    inputs,
    ref,
    *,
    params=None,
    tol=FP32,
    out="f32",
    alt_inputs=None,
    alt_tol=None,
    seed=0,
):
    """Run the four checks on one node.

    make        builds the node; it is called inside the net.
    inputs      {slot: shape or generator}. A bare shape draws standard normal.
                The first slot's chain ends at out0.
    ref         ref(**inputs, **params) -> the expected out0, a torch tensor.
    params      {slot: shape or generator}, uploaded into the node's parameter slots.
    tol         (rtol, atol), one of the tiers above.
    out         how out0 reads back: "f32", or "f16" / "u32", which compare exactly.
    alt_inputs  {slot: shape or generator} for a rerun on the same net and params.
    alt_tol     the rerun's tol when it takes another kernel; None keeps tol.
    """
    rng = np.random.default_rng(seed)
    params = {slot: _draw(gen, rng) for slot, gen in (params or {}).items()}
    net = _net(make, list(inputs), params)
    fed = {slot: _draw(gen, rng) for slot, gen in inputs.items()}

    got = _forward(net, fed, out)
    _compare(got, _expected(ref, fed, params), tol, out, "out0 vs the reference")
    np.testing.assert_array_equal(
        _forward(net, fed, out), got, err_msg="the second forward differs from the first"
    )
    if out != "u32":
        assert np.isfinite(got.astype(np.float64)).all(), "out0 has NaN or Inf"

    if alt_inputs:
        fed = {slot: _draw(gen, rng) for slot, gen in alt_inputs.items()}
        _compare(
            _forward(net, fed, out),
            _expected(ref, fed, params),
            alt_tol or tol,
            out,
            "out0 vs the reference, rerun on alt_inputs",
        )


def _draw(gen, rng):
    array = gen(rng) if callable(gen) else rng.standard_normal(gen)
    return np.ascontiguousarray(array, dtype=np.float32)


def _tensor(array):
    return pyvai.Tensor.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def _net(make, slots, params):
    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__(w_default_dtype=pyvai.Dtype.F32)
            self.node = make()
            self.input(slots[0]) - slots[0] / self.node - self.output("out0")
            for slot in slots[1:]:
                self.input(slot) - slot / self.node

    net = Net()
    for slot, array in params.items():
        net.node[slot] = _tensor(array)
    return net


def _forward(net, fed, out):
    result = net(**{slot: _tensor(array) for slot, array in fed.items()})
    return result.to_u32() if out == "u32" else result.to_numpy()


def _expected(ref, fed, params):
    args = {k: torch.from_numpy(a.astype(np.float64)) for k, a in {**fed, **params}.items()}
    want = ref(**args)
    return want.numpy() if isinstance(want, torch.Tensor) else np.asarray(want)


def _compare(got, want, tol, out, what):
    assert got.shape == want.shape, f"{what}: shape {got.shape} != {want.shape}"
    if out == "u32":
        np.testing.assert_array_equal(got, want.astype(np.uint32), err_msg=what)
    elif out == "f16":
        np.testing.assert_array_equal(got, want.astype(np.float16), err_msg=what)
    else:
        np.testing.assert_allclose(got, want, rtol=tol[0], atol=tol[1], err_msg=what)
