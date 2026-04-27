from src.grn_engine.grn_stepper import run_grn


INPUT_CANONICAL = {
    "InImpulse": "InCollisionImpulse",
    "InCollisionImpulse": "InCollisionImpulse",
    "InMolecule": "InChemicalConcentration",
    "InChemicalConcentration": "InChemicalConcentration",
    "InShearStress": "InShearStress",
}


OUTPUT_CANONICAL = {
    "OutStickiness": "OutStickiness",
    "OutCellShapeChange": "OutMorphologyChange",
    "OutMorphologyChange": "OutMorphologyChange",
    "OutSecretionRate": "OutSecretionRate",
}


def canonical_sensor_name(name):
    return INPUT_CANONICAL.get(name, name)


def canonical_output_name(name):
    return OUTPUT_CANONICAL.get(name, name)


def apply_sensor_inputs(model, state, sensors):
    new_state = state.copy()

    for raw_name, value in sensors.items():
        name = canonical_sensor_name(raw_name)

        if name not in model.node_index:
            continue

        idx = model.node_index[name]

        if idx not in model.input_indices:
            continue

        value = max(0.0, min(1.0, float(value)))
        new_state[idx] = value

    return new_state


def read_actuator_outputs(model, state):
    outputs = {}

    for idx in model.output_indices:
        raw_name = model.node_names[idx]
        name = canonical_output_name(raw_name)
        outputs[name] = float(state[idx])

    return outputs


def run_grn_pipeline(model, initial_state, sensors, steps=10):
    state_with_inputs = apply_sensor_inputs(model, initial_state, sensors)

    history = run_grn(
        model,
        state_with_inputs,
        steps=steps,
        dt=0.15,
    )

    final_state = history[-1]
    outputs = read_actuator_outputs(model, final_state)

    return final_state, outputs, history