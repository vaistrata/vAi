import pytest
from node_check import ACTIVATIONS, TRANS, check_node, normal

import pyvai
from pyvai import nodes


@pytest.mark.parametrize("activation", ["RELU", "GELU", "GELU_TANH", "SILU", "SIGMOID"])
def test_activation(activation):
    check_node(
        lambda: nodes.ActivationNode(act_type=getattr(pyvai.ActType, activation)),
        {"in0": normal((5, 33), 3.0)},
        lambda in0: ACTIVATIONS[activation](in0),
        tol=TRANS,
    )
