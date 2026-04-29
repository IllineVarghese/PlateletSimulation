from src.grn_engine.io_mapping import run_grn_pipeline


class GRNAgent:
    def __init__(self, model):
        self.model = model
        self.state = [0.0] * len(model.node_names)

        self.sensors = {
            "InCollisionImpulse": 0.0,
            "InChemicalConcentration": 0.0,
            "InMolecule": 0.0,
            "InShearStress": 0.0,
        }

        self.outputs = {
            "OutStickiness": 0.0,
            "OutMorphologyChange": 0.0,
            "OutCellShapeChange": 0.0,
            "OutSecretionRate": 0.0,
        }

    def set_sensor(self, name, value):
        if name not in self.sensors:
            raise KeyError(f"Unknown sensor name: {name}")

        self.sensors[name] = float(value)

    def step(self, steps=1, dt=0.08):
        final_state, outputs, history = run_grn_pipeline(
            model=self.model,
            initial_state=self.state,
            sensors=self.sensors,
            steps=steps,
            dt=dt,
        )

        self.state = final_state
        self.outputs.update(outputs)

        if "OutMorphologyChange" in outputs:
            self.outputs["OutCellShapeChange"] = outputs["OutMorphologyChange"]

        return self.outputs

    def get_output(self, name):
        if name == "OutCellShapeChange":
            name = "OutMorphologyChange"

        if name not in self.outputs:
            raise KeyError(f"Unknown output name: {name}")

        return float(self.outputs[name])