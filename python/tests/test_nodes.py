"""Builtin nodes: the re-export list, a node's weights set from Python, and
IndexCopy's binding-owned index buffer.

The kernels themselves are the core's to test.
"""

import numpy as np

import pyvai
from pyvai import nodes


def test_nodes_reexports_every_bound_node():
    """pyvai.nodes lists its names by hand, so a node bound in C++ but never
    re-exported is caught here."""
    bound = {
        name
        for name, cls in vars(pyvai._pyvai).items()
        if isinstance(cls, type)
        and issubclass(cls, pyvai.Node)
        and cls not in (pyvai.Node, pyvai.InputNode, pyvai.OutputNode)
    }
    assert set(nodes.__all__) == bound
    for name in nodes.__all__:
        assert getattr(nodes, name) is getattr(pyvai._pyvai, name)


def test_linear_with_bias():
    """Identity weight + bias: out = x + b."""

    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__(w_default_dtype=pyvai.Dtype.F32)
            self.l = nodes.LinearNode(in_dim=4, out_dim=4)
            self.input("in0") - self.l - self.output("out0")

    net = Net()
    net.l["weight"] = pyvai.Tensor.from_numpy(np.eye(4, dtype=np.float32))
    net.l["bias"] = pyvai.Tensor.from_numpy(np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))
    out = net(pyvai.Tensor.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)))
    np.testing.assert_allclose(out.to_numpy(), [11.0, 22.0, 33.0, 44.0])


def test_index_copy_rewrites_and_grows_its_index():
    """set_index rewrites the node's own index in place, or grows it, which
    re-records the net; a forward without in1 passes in0 through."""
    T = pyvai.Tensor.from_numpy

    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__()
            self.c = nodes.IndexCopyNode()
            self.input("dst") - "in0" / self.c - self.output("out0")
            self.input("src") - "in1" / self.c

    dst = np.zeros((5, 2), dtype=np.float32)
    net = Net()
    for index in ([3, 0], [1, 4], [4, 2, 0]):
        src = np.arange(2 * len(index), dtype=np.float32).reshape(-1, 2) + 1
        net.c.set_index(index)
        want = dst.copy()
        want[index] = src
        np.testing.assert_array_equal(net(dst=T(dst), src=T(src)).to_numpy(), want)
    np.testing.assert_array_equal(net(dst=T(dst + 7)).to_numpy(), dst + 7)


def test_attention_accumulator_dtypes_keep_the_result():
    """qk_acc_dtype / o_acc_dtype only change the accumulation precision."""
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((2, 6, 8), dtype=np.float32) for _ in range(3))

    class Net(pyvai.NeuralNet):
        def __init__(self, **kw):
            super().__init__(o_default_dtype=pyvai.Dtype.F32)
            self.a = nodes.AttentionNode(num_q_heads=2, head_dim=8, **kw)
            self.input("q") - "Q" / self.a
            self.input("k") - "K" / self.a
            self.input("v") - "V" / self.a
            self.a - self.output("out0")

    T = pyvai.Tensor.from_numpy
    base = Net()(q=T(q), k=T(k), v=T(v)).to_numpy()
    wide = Net(qk_acc_dtype=pyvai.Dtype.F32)(q=T(q), k=T(k), v=T(v)).to_numpy()
    np.testing.assert_allclose(base, wide, atol=1e-2)


def test_linear_weight_slot_after_a_forward():
    """The Intel path packs W into its own layout at the first prepare."""
    K, N = 64, 48
    rng = np.random.default_rng(0)
    w = rng.standard_normal((N, K), dtype=np.float32)
    x = rng.standard_normal((8, K), dtype=np.float32)

    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__(w_default_dtype=pyvai.Dtype.F32, o_default_dtype=pyvai.Dtype.F32)
            self.l = nodes.LinearNode(in_dim=K, out_dim=N)
            self.input("in0") - self.l - self.output("out0")

    net = Net()
    net.l["weight"] = pyvai.Tensor.from_numpy(w)
    assert list(net.l["weight"].shape) == [N, K]
    out = net(pyvai.Tensor.from_numpy(x)).to_numpy()
    np.testing.assert_allclose(out, x @ w.T, atol=1e-4)
    assert list(net.l["weight"].shape) in ([N, K], [(N + 31) // 32, (K + 1) // 2, 64])
