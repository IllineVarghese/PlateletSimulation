from src.grn_engine.io_mapping import run_grn_pipeline


class GRNAgent:
    def __init__(self, model):
        self.model = model

        # initial GRN node values
        self.state = [0.0] * len(model.node_names)

        # sensors
        # Phase 2 active sensor:
        #   InCollisionImpulse
        # Phase 3 placeholders:
        #   InMolecule
        #   InShearStress
        self.sensors = {
            "InCollisionImpulse": 0.0,
            "InMolecule": 0.0,
            "InShearStress": 0.0,
        }

        # actuator outputs
        # Phase 2 active output:
        #   OutStickiness
        # Phase 3 placeholder:
        #   OutCellShapeChange
        self.outputs = {
            "OutStickiness": 0.0,
            "OutCellShapeChange": 0.0,
        }

    def set_sensor(self, name, value):
        if name not in self.sensors:
            raise KeyError(f"Unknown sensor name: {name}")
        self.sensors[name] = float(value)

    def step(self):
        final_state, outputs, history = run_grn_pipeline(
            self.model,
            self.state,
            self.sensors,
            steps=1
        )

        self.state = final_state

        # update real GRN outputs that exist in the current model
        self.outputs.update(outputs)

        # keep Phase 3 placeholder output stable for now
        if "OutCellShapeChange" not in outputs:
            self.outputs["OutCellShapeChange"] = 0.0

        return self.outputs

    def get_output(self, name):
        if name not in self.outputs:
            raise KeyError(f"Unknown output name: {name}")
        return float(self.outputs[name])