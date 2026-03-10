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

from src.grn_engine.grn_stepper import run_grn


def test_run_grn_multiple_steps():
    model = load_graphml("data/networks/test_minimal.graphml")

    initial_state = [1.0, 0.0, 0.0]

    history = run_grn(model, initial_state, steps=5)

    assert len(history) == 6

def test_run_grn_stays_bounded():
    model = load_graphml("data/networks/test_minimal.graphml")

    initial_state = [1.0, 0.0, 0.0]

    history = run_grn(model, initial_state, steps=20)

    for state in history:
        for value in state:
            assert 0.0 <= value <= 1.0

def test_output_increases_over_time():
    model = load_graphml("data/networks/test_minimal.graphml")

    initial_state = [1.0, 0.0, 0.0]

    history = run_grn(model, initial_state, steps=10)

    first_output = history[0][2]
    last_output = history[-1][2]

    assert last_output > first_output                