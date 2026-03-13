def apply_sensor_inputs(model, state, sensors):
    """
    Write sensor values into GRN input nodes.

    model   : GRNModel
    state   : list of node values
    sensors : dictionary of sensor values
    """

    new_state = state.copy()

    for name, value in sensors.items():

        if name not in model.node_index:
            continue

        idx = model.node_index[name]

        if idx not in model.input_indices:
            continue

        # clamp value to [0,1]
        value = max(0.0, min(1.0, value))

        new_state[idx] = value

    return new_state

def read_actuator_outputs(model, state):
    """
    Read GRN output nodes into a dictionary.

    model : GRNModel
    state : list of node values
    """
    outputs = {}

    for idx in model.output_indices:
        name = model.node_names[idx]
        outputs[name] = state[idx]

    return outputs

from src.grn_engine.grn_stepper import run_grn


def run_grn_pipeline(model, initial_state, sensors, steps=10):

    state_with_inputs = apply_sensor_inputs(model, initial_state, sensors)

    history = run_grn(model, state_with_inputs, steps)

    final_state = history[-1]

    outputs = read_actuator_outputs(model, final_state)

    return final_state, outputs, history