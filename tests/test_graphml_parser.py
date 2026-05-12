from src.grn_engine.graphml_parser import load_graphml


def test_load_graphml_minimal_network():
    model = load_graphml("data/networks/test_minimal.graphml")

    assert model.node_names == [
        "InCollisionImpulse",
        "SignalA",
        "OutStickiness",
    ]

    assert model.node_index["InCollisionImpulse"] == 0
    assert model.node_index["SignalA"] == 1
    assert model.node_index["OutStickiness"] == 2

    assert model.edges_src == [0, 1]
    assert model.edges_dst == [1, 2]
    assert model.edges_weight == [1.0, 0.8]

    assert model.input_indices == [0]
    assert model.output_indices == [2]