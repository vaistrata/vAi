from node_check import EXACT, check_node

from pyvai._pyvai import model_nodes

rtdetrv2 = model_nodes.rtdetrv2


def test_query_score():
    check_node(
        rtdetrv2.QueryScoreNode,
        {"in0": (300, 80)},
        lambda in0: in0.max(1).values,
        tol=EXACT,
        alt_inputs={"in0": (65, 1)},
    )
