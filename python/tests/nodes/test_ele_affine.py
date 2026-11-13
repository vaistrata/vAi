from node_check import check_node

from pyvai import nodes


def test_ele_affine_inputs_broadcast_row():
    check_node(
        lambda: nodes.EleAffineNode(scale_as_input=True, shift_as_input=True),
        {"in0": (6, 32), "scale": (32,), "shift": (6, 32)},
        lambda in0, scale, shift: scale * in0 + shift,
    )


def test_ele_affine_params_per_channel():
    check_node(
        nodes.EleAffineNode,
        {"in0": (4, 5, 6)},
        lambda in0, scale, shift: scale * in0 + shift,
        params={"scale": (4, 1, 1), "shift": (4, 1, 1)},
    )
