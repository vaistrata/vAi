import torch
from node_check import EXACT, check_node

from pyvai import nodes


def test_concat_axis0_copy():
    # Axis 0 takes buffer copies.
    check_node(
        lambda: nodes.ConcatNode(axis=0),
        {"in0": (2, 5), "in1": (3, 5)},
        lambda in0, in1: torch.cat([in0, in1], 0),
        tol=EXACT,
    )


def test_concat_axis1_gather():
    # Other axes take the gather kernel.
    check_node(
        lambda: nodes.ConcatNode(axis=1),
        {"in0": (2, 3, 4), "in1": (2, 5, 4)},
        lambda in0, in1: torch.cat([in0, in1], 1),
        tol=EXACT,
    )


def test_concat_three_last_axis():
    check_node(
        lambda: nodes.ConcatNode(axis=-1, input_count=3),
        {"in0": (2, 3), "in1": (2, 4), "in2": (2, 1)},
        lambda in0, in1, in2: torch.cat([in0, in1, in2], -1),
        tol=EXACT,
    )
