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