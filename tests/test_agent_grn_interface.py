from src.grn_engine.graphml_parser import load_graphml
from src.grn_engine.io_mapping import apply_sensor_inputs


def test_apply_sensor_inputs():

    model = load_graphml("data/networks/test_minimal.graphml")

    # initial state
    state = [0.0, 0.0, 0.0]

    sensors = {
        "InCollisionImpulse": 1.0
    }

    new_state = apply_sensor_inputs(model, state, sensors)

    assert new_state[0] == 1.0