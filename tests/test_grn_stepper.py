from src.grn_engine.graphml_parser import load_graphml
from src.grn_engine.grn_stepper import grn_step


def test_grn_step_propagates_signal():

    model = load_graphml("data/networks/test_minimal.graphml")

    # initial node values
    state = [1.0, 0.0, 0.0]

    new_state = grn_step(model, state, dt=0.5)

    # SignalA should increase
    assert new_state[1] > 0.0

    # OutStickiness should remain small but >= 0
    assert new_state[2] >= 0.0