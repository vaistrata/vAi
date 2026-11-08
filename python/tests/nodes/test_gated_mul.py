import pytest
from node_check import EPILOGUES, TRANS, check_node

import pyvai
from pyvai import nodes


@pytest.mark.parametrize("activation", ["SILU", "RELU", "GELU_TANH"])
def test_gated_mul(activation):
    def ref(in0):
        gate, up = in0.chunk(2, dim=-1)
        return EPILOGUES[activation](gate) * up

    check_node(
        lambda: nodes.GatedMulNode(activation=getattr(pyvai.Epilogue, activation)),
        {"in0": (4, 48)},
        ref,
        tol=TRANS,
    )
