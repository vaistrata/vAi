import math

import numpy as np
import pytest

import pyvai
from pyvai._pyvai import model_nodes

rtdetrv2 = model_nodes.rtdetrv2


def _ref(rows, k):
    """Descending value, ties by ascending index, NaN after every number."""
    values, indices = [], []
    for row in rows:
        order = sorted(
            range(len(row)),
            key=lambda i: (math.isnan(row[i]), 0.0 if math.isnan(row[i]) else -row[i], i),
        )
        indices.append(order[:k])
        values.append(row[order[:k]])
    return np.array(values, np.float32), np.array(indices, np.uint32)


def _run(k, inputs):
    """TopKNode has no out0, so check_node cannot read it: this net outputs
    both slots. Each array runs twice, the second replaying the recording."""

    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__(w_default_dtype=pyvai.Dtype.F32)
            self.node = rtdetrv2.TopKNode(k=k)
            self.input("in0") - self.node
            self.node / "values" - self.output("values")
            self.node / "indices" - self.output("indices")

    net = Net()
    for x in inputs:
        got = [net(in0=pyvai.Tensor.from_numpy(x)) for _ in range(2)]
        for result in got:
            want_values, want_indices = _ref(x.reshape(-1, x.shape[-1]), k)
            np.testing.assert_array_equal(
                result["values"].to_numpy(), want_values.reshape(*x.shape[:-1], k)
            )
            np.testing.assert_array_equal(
                result["indices"].to_u32(), want_indices.reshape(*x.shape[:-1], k)
            )


@pytest.mark.parametrize(
    "k, shape",
    [
        pytest.param(1, (37,), id="k1_vector"),
        pytest.param(300, (4, 512), id="one_block"),
        pytest.param(300, (2, 2000), id="merge_passes"),  # 4 blocks of 512, two merge passes
        pytest.param(512, (1, 513), id="k_max_tail"),
    ],
)
def test_topk(k, shape):
    rng = np.random.default_rng(0)
    # A second shape re-prepares the node; its block count differs.
    alt = (3, 700) if k <= 300 else (2, 1100)
    _run(
        k,
        [
            rng.standard_normal(shape).astype(np.float32),
            rng.standard_normal(alt).astype(np.float32),
        ],
    )


def test_topk_ties_and_nan():
    rng = np.random.default_rng(1)
    x = rng.integers(-3, 4, (3, 1200)).astype(np.float32)
    x[0, 5] = x[1, 700] = np.nan
    x[2, 1000] = -np.inf
    _run(40, [x])
