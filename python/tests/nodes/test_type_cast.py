from node_check import EXACT, check_node, normal

import pyvai
from pyvai import nodes


def test_type_cast_f32_to_f16():
    check_node(
        lambda: nodes.TypeCastNode(o_dtype=pyvai.Dtype.F16),
        {"in0": normal((3, 40), 10.0)},
        lambda in0: in0,
        tol=EXACT,
        out="f16",
    )
