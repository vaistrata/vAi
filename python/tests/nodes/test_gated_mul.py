import pytest
from node_check import ACTIVATIONS, TRANS, check_node

import pyvai
from pyvai import nodes


@pytest.mark.parametrize("activation", ["SILU", "RELU", "GELU_TANH"])
def test_gated_mul(activation):
    def ref(in0):
        gate, up = in0.chunk(2, dim=-1)
        return ACTIVATIONS[activation](gate) * up

    check_node(
        lambda: nodes.GatedMulNode(act_type=getattr(pyvai.ActType, activation)),
        {"in0": (4, 48)},
        ref,
        tol=TRANS,
    )
