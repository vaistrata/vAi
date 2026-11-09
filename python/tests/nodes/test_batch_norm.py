import torch.nn.functional as F
from node_check import check_node, uniform

from pyvai import nodes


def test_batch_norm():
    def ref(in0, gamma, beta, mean, variance):
        return F.batch_norm(in0[None], mean, variance, gamma, beta, training=False, eps=1e-5)[0]

    check_node(
        lambda: nodes.BatchNormNode(channels=8),
        {"in0": (8, 6, 6)},
        ref,
        params={"gamma": (8,), "beta": (8,), "mean": (8,), "variance": uniform((8,), 0.5, 2.0)},
    )
