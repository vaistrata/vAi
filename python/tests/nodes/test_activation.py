import pytest
from node_check import EPILOGUES, TRANS, check_node, normal

import pyvai
from pyvai import nodes


@pytest.mark.parametrize("epilogue", ["RELU", "GELU", "GELU_TANH", "SILU", "SIGMOID"])
def test_activation(epilogue):
    check_node(
        lambda: nodes.ActivationNode(epilogue=getattr(pyvai.Epilogue, epilogue)),
        {"in0": normal((5, 33), 3.0)},
        lambda in0: EPILOGUES[epilogue](in0),
        tol=TRANS,
    )
