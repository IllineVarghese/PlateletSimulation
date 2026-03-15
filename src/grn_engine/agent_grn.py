from src.grn_engine.io_mapping import run_grn_pipeline


class GRNAgent:

    def __init__(self, model):
        self.model = model

        # initial GRN node values
        self.state = [0.0] * len(model.node_names)

        # sensors
        self.sensors = {
            "InCollisionImpulse": 0.0
        }

        # actuator outputs
        self.outputs = {
            "OutStickiness": 0.0
        }

    def set_sensor(self, name, value):
        self.sensors[name] = value

    def step(self):

        final_state, outputs, history = run_grn_pipeline(
            self.model,
            self.state,
            self.sensors,
            steps=1
        )

        self.state = final_state
        self.outputs.update(outputs)

    def get_output(self, name):
        return self.outputs.get(name, 0.0)