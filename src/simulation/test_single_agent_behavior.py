# src/simulation/test_single_agent_behavior.py

import numpy as np

# TODO: import your actual classes
# from src.simulation.grn_agent import GRNAgent
# from src.grn_engine.grn_model import GRNModel

# Temporary placeholder (remove later)
class DummyGRN:
    def __init__(self):
        self.nodes = {}

    def set_node(self, name, value):
        self.nodes[name] = value

    def get_node(self, name):
        return self.nodes.get(name, 0.0)

    def step(self, dt):
        # simple fake behavior for now
        self.nodes["OutStickiness"] = self.nodes.get("InCollisionImpulse", 0.0)


class AgentSensors:
    def __init__(self):
        self.collision_impulse = 0.0
        self.chemical_concentration = 0.0
        self.shear_stress = 0.0


class AgentOutputs:
    def __init__(self):
        self.stickiness = 0.0
        self.morphology = 0.0
        self.secretion_rate = 0.0


class GRNAgent:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([1.0, 0.0, 0.0])

        self.grn = DummyGRN()
        self.sensors = AgentSensors()
        self.outputs = AgentOutputs()

        self.debug_history = {
            "collision_impulse": [],
            "stickiness": [],
            "speed": [],
        }


# -----------------------------
# Simulation loop
# -----------------------------
def run_simulation():
    agent = GRNAgent()

    dt = 0.1
    steps = 100

    for step in range(steps):

        # --- Step 1: reset sensors ---
        agent.sensors.collision_impulse = 0.0

        # --- Step 2: fake collision input (for testing) ---
        if step == 20:
            agent.sensors.collision_impulse = 1.0

        # --- Step 3: send to GRN ---
        agent.grn.set_node("InCollisionImpulse", agent.sensors.collision_impulse)

        # --- Step 4: step GRN ---
        agent.grn.step(dt)

        # --- Step 5: read output ---
        agent.outputs.stickiness = agent.grn.get_node("OutStickiness")

        # --- Step 6: apply behavior ---
        agent.velocity *= (1.0 - 0.5 * agent.outputs.stickiness)

        # --- Step 7: update position ---
        agent.position += agent.velocity * dt

        # --- Step 8: record debug ---
        agent.debug_history["collision_impulse"].append(agent.sensors.collision_impulse)
        agent.debug_history["stickiness"].append(agent.outputs.stickiness)
        agent.debug_history["speed"].append(np.linalg.norm(agent.velocity))

    return agent


if __name__ == "__main__":
    agent = run_simulation()

    print("Final speed:", agent.debug_history["speed"][-1])