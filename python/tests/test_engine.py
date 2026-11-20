"""Tensor <-> numpy, forward I/O and Weights."""
import gc
import json
import struct
import subprocess
import sys
import weakref

import numpy as np

import pyvai
from pyvai import nodes


# ---------------------------------------------------------------------------
# Tensor <-> numpy
# ---------------------------------------------------------------------------

def test_tensor_numpy_roundtrip(rng):
    """from_numpy() / to_numpy() keep shape and values, through reshape() and permute() too."""
    arr = rng.standard_normal((28, 28, 1)).astype(np.float32)
    t = pyvai.Tensor.from_numpy(arr)
    assert t.shape == (28, 28, 1) and t.ndim == 3 and len(t) == 28
    np.testing.assert_array_equal(t.to_numpy(), arr)
    np.testing.assert_array_equal(t.reshape([28, 28]).to_numpy(), arr.reshape(28, 28))
    np.testing.assert_array_equal(t.permute(2, 0, 1).to_numpy(), np.transpose(arr, (2, 0, 1)))
    assert pyvai.Tensor().shape == () and pyvai.Tensor().ndim == 0


def test_u32_roundtrip():
    """from_u32 / to_u32: uint32 index bits in an F32-sized tensor, from an
    int32 or uint32 array or a plain sequence of ints."""
    ids = np.array([0, 1, 50256, 2**32 - 1], dtype=np.uint32)
    for src in (ids, ids.astype(np.int32), [0, 1, 50256, 2**32 - 1], (0, 1, 50256, 2**32 - 1)):
        t = pyvai.Tensor.from_u32(src)
        assert t.dtype is pyvai.Dtype.F32 and t.shape == (4,)
        got = t.to_u32()
        assert got.dtype == np.uint32
        np.testing.assert_array_equal(got, ids)


# ---------------------------------------------------------------------------
# Forward I/O
# ---------------------------------------------------------------------------

class _ReluNet(pyvai.NeuralNet):
    """A real kernel with its own output buffer (no aliasing)."""
    def __init__(self):
        super().__init__()
        self.act = nodes.ActivationNode(act_type=pyvai.ActType.RELU)
        self.input("in0") - self.act - self.output("out0")


class _AddNet(pyvai.NeuralNet):
    """Two inputs (a, b), one output."""
    def __init__(self):
        super().__init__()
        self.add = nodes.AddNode()
        self.input("a") - "in0"/self.add - self.output("out0")
        self.input("b") - "in1"/self.add


class _TwoOutNet(pyvai.NeuralNet):
    """One input fanning out to two outputs: `same` (identity) and `relu`."""
    def __init__(self):
        super().__init__()
        self.id = nodes.IdentityNode()
        self.act = nodes.ActivationNode(act_type=pyvai.ActType.RELU)
        self.input("in0") - self.id - self.output("same")
        self.id - self.act - self.output("relu")


def _x():
    return pyvai.Tensor.from_numpy(np.array([-1, 2, -3, 4], dtype=np.float32))


_RELU_X = np.array([0, 2, 0, 4], dtype=np.float32)


def test_call_forms():
    """net(x), net({...}) and net(name=x) are the same call; a two-input net
    takes keywords."""
    net = _ReluNet()
    x = _x()
    for out in (net(x), net({"in0": x}), net(in0=x)):
        np.testing.assert_array_equal(out.to_numpy(), _RELU_X)
    ones = pyvai.Tensor.from_numpy(np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(_AddNet()(a=ones, b=ones).to_numpy(), [2, 2, 2, 2])


def test_multi_output_returns_dict():
    """One output comes back as that Tensor, several as a name dict;
    input_names / output_names list the registered boundaries."""
    net = _TwoOutNet()
    out = net(_x())
    assert isinstance(out, dict) and sorted(out) == ["relu", "same"]
    np.testing.assert_array_equal(out["same"].to_numpy(), _x().to_numpy())
    np.testing.assert_array_equal(out["relu"].to_numpy(), _RELU_X)
    assert net.input_names == ["in0"] and net.output_names == ["relu", "same"]
    assert _AddNet().input_names == ["a", "b"]


def test_output_keeps_net_alive():
    """A forward output, or a reshape() view of it, views the net's buffers and
    so keeps the net alive, without making the net uncollectable."""
    net = _ReluNet()
    out = net(_x())
    view = out.reshape([2, 2])
    assert not out.has_host_data
    dead = weakref.ref(net)
    del net
    gc.collect()
    assert dead() is not None
    np.testing.assert_array_equal(out.to_numpy(), _RELU_X)
    np.testing.assert_array_equal(view.to_numpy(), _RELU_X.reshape(2, 2))
    del out
    gc.collect()
    assert dead() is not None                       # the view alone still pins it
    del view
    gc.collect()
    assert dead() is None                           # the net <-> cached-output cycle is collectable


def test_no_leaks_at_interpreter_exit(tmp_path):
    """A forward in a fresh interpreter exits silently: nanobind reports what a
    reference cycle the GC cannot see leaves alive. The neutral cwd keeps a
    source checkout's pyvai/ from shadowing the installed package."""
    script = (
        "import numpy as np, pyvai\n"
        "from pyvai import nodes\n"
        "net = pyvai.NeuralNet()\n"
        "net.input('in0') - nodes.ActivationNode(act_type=pyvai.ActType.RELU)"
        " - net.output('out0')\n"
        "out = net(pyvai.Tensor.from_numpy(np.arange(4, dtype=np.float32)))\n"
        "view = out.reshape([2, 2]); view.to_numpy()\n"
        "nodes.IdentityNode() - nodes.IdentityNode()\n"      # wired, never given a net
        "del net, out, view\n"
    )
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                       stdin=subprocess.DEVNULL, timeout=120, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "leaked" not in r.stderr, r.stderr


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def _write_safetensors(path, tensors: dict) -> None:
    """Hand-rolled minimal safetensors writer — no `safetensors` dependency."""
    metadata, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        metadata[name] = {"dtype": "F32", "shape": list(arr.shape),
                          "data_offsets": [offset, offset + arr.nbytes]}
        blobs.append(arr.tobytes())
        offset += arr.nbytes
    header = json.dumps(metadata).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        for b in blobs:
            f.write(b)


def test_weights_roundtrip_and_conversion(tmp_path, rng):
    """Weights lists a file's tensors, loads one in the file's dtype or a requested one, and
    reads its header without loading it."""
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal((2,)).astype(np.float32)
    path = tmp_path / "w.safetensors"
    _write_safetensors(str(path), {"a": a, "b": b})

    w = pyvai.Weights(path)
    assert sorted(w.keys()) == ["a", "b"] and len(w) == 2 and "a" in w and "c" not in w
    assert w.format == "safetensors" and w.path == path
    np.testing.assert_array_equal(w["a"].to_numpy(), a)
    assert w.get("b").dtype == pyvai.Dtype.F32
    t16 = w.get("b", dtype=pyvai.Dtype.F16)
    assert t16.dtype == pyvai.Dtype.F16
    np.testing.assert_allclose(t16.to_numpy(), b.astype(np.float16), atol=1e-3)
    assert w.info("a") == pyvai.TensorInfo(key="a", shape=(3, 4), dtype="F32")
