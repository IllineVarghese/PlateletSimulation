# src/simulation/test_single_agent_behavior.py

import os
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------
GRN_INPUT_NODES = {
    "collision_impulse": "InCollisionImpulse",
    "chemical_concentration": "InChemicalConcentration",
    "shear_stress": "InShearStress",
}

GRN_OUTPUT_NODES = {
    "stickiness": "OutStickiness",
    "morphology": "OutCellShapeChange",
    "secretion_rate": "OutSecretionRate",
}

CHEMICAL_SOURCE_POSITION = np.array([1.5, 0.0, 0.0], dtype=float)
CHEMICAL_SOURCE_RADIUS = 0.75
VESSEL_RADIUS = 1.0

RESULTS_DIR = "results"
PLOT_PATH = os.path.join(RESULTS_DIR, "week1_day5_single_agent_behavior.png")
CSV_PATH = os.path.join(RESULTS_DIR, "week1_day5_single_agent_behavior.csv")


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


# -----------------------------
# Temporary GRN placeholder
# Replace later with your real GRN model
# -----------------------------
class DummyGRN:
    def __init__(self):
        self.nodes = {}

    def set_node(self, name, value):
        self.nodes[name] = float(value)

    def get_node(self, name):
        return float(self.nodes.get(name, 0.0))

    def step(self, dt):
        collision = self.nodes.get(GRN_INPUT_NODES["collision_impulse"], 0.0)
        chemical = self.nodes.get(GRN_INPUT_NODES["chemical_concentration"], 0.0)
        shear = self.nodes.get(GRN_INPUT_NODES["shear_stress"], 0.0)

        stickiness = clamp(0.6 * collision + 0.15 * chemical + 0.25 * shear)
        morphology = clamp(0.15 * collision + 0.35 * chemical + 0.50 * shear)
        secretion_rate = clamp(0.15 * collision + 0.65 * chemical + 0.20 * shear)

        self.nodes[GRN_OUTPUT_NODES["stickiness"]] = stickiness
        self.nodes[GRN_OUTPUT_NODES["morphology"]] = morphology
        self.nodes[GRN_OUTPUT_NODES["secretion_rate"]] = secretion_rate


# -----------------------------
# Agent data containers
# -----------------------------
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
        self.position = np.array([0.0, 0.4, 0.0], dtype=float)
        self.velocity = np.array([1.0, 0.0, 0.0], dtype=float)

        self.grn = DummyGRN()
        self.sensors = AgentSensors()
        self.outputs = AgentOutputs()

        self.morphology_level = 0.0
        self.secreted_amount = 0.0

        self.debug_history = {
            "step": [],
            "collision_impulse": [],
            "chemical_concentration": [],
            "shear_stress": [],
            "stickiness": [],
            "morphology": [],
            "secretion_rate": [],
            "speed": [],
            "x_position": [],
            "y_position": [],
            "radial_distance": [],
        }


# -----------------------------
# Helper functions
# -----------------------------
def reset_sensors(agent: GRNAgent) -> None:
    agent.sensors.collision_impulse = 0.0
    agent.sensors.chemical_concentration = 0.0
    agent.sensors.shear_stress = 0.0


def compute_collision_impulse(agent: GRNAgent, step: int) -> float:
    if 20 <= step <= 30:
        return 1.0
    if 31 <= step <= 40:
        return 0.5
    return 0.0


def compute_chemical_concentration(agent: GRNAgent, step: int) -> float:
    distance = np.linalg.norm(agent.position - CHEMICAL_SOURCE_POSITION)
    concentration = 1.0 - (distance / CHEMICAL_SOURCE_RADIUS)
    return clamp(concentration)


def compute_shear_stress(agent: GRNAgent, step: int) -> float:
    radial_distance = np.linalg.norm(agent.position[1:3])
    normalized_shear = radial_distance / VESSEL_RADIUS
    return clamp(normalized_shear)


def write_sensors_to_grn(agent: GRNAgent) -> None:
    agent.grn.set_node(GRN_INPUT_NODES["collision_impulse"], clamp(agent.sensors.collision_impulse))
    agent.grn.set_node(GRN_INPUT_NODES["chemical_concentration"], clamp(agent.sensors.chemical_concentration))
    agent.grn.set_node(GRN_INPUT_NODES["shear_stress"], clamp(agent.sensors.shear_stress))


def step_agent_grn(agent: GRNAgent, dt: float) -> None:
    agent.grn.step(dt)


def read_grn_outputs(agent: GRNAgent) -> None:
    agent.outputs.stickiness = clamp(agent.grn.get_node(GRN_OUTPUT_NODES["stickiness"]))
    agent.outputs.morphology = clamp(agent.grn.get_node(GRN_OUTPUT_NODES["morphology"]))
    agent.outputs.secretion_rate = clamp(agent.grn.get_node(GRN_OUTPUT_NODES["secretion_rate"]))


def apply_stickiness(agent: GRNAgent, dt: float) -> None:
    damping_factor = 1.0 - 0.1 * agent.outputs.stickiness
    damping_factor = max(0.0, damping_factor)
    agent.velocity *= damping_factor


def apply_morphology(agent: GRNAgent) -> None:
    agent.morphology_level = agent.outputs.morphology


def apply_secretion(agent: GRNAgent, dt: float) -> None:
    agent.secreted_amount += agent.outputs.secretion_rate * dt


def apply_agent_outputs(agent: GRNAgent, dt: float) -> None:
    apply_stickiness(agent, dt)
    apply_morphology(agent)
    apply_secretion(agent, dt)


def update_position(agent: GRNAgent, dt: float) -> None:
    agent.position += agent.velocity * dt


def record_agent_debug(agent: GRNAgent, step: int) -> None:
    radial_distance = float(np.linalg.norm(agent.position[1:3]))

    agent.debug_history["step"].append(step)
    agent.debug_history["collision_impulse"].append(agent.sensors.collision_impulse)
    agent.debug_history["chemical_concentration"].append(agent.sensors.chemical_concentration)
    agent.debug_history["shear_stress"].append(agent.sensors.shear_stress)
    agent.debug_history["stickiness"].append(agent.outputs.stickiness)
    agent.debug_history["morphology"].append(agent.outputs.morphology)
    agent.debug_history["secretion_rate"].append(agent.outputs.secretion_rate)
    agent.debug_history["speed"].append(float(np.linalg.norm(agent.velocity)))
    agent.debug_history["x_position"].append(float(agent.position[0]))
    agent.debug_history["y_position"].append(float(agent.position[1]))
    agent.debug_history["radial_distance"].append(radial_distance)


def save_debug_csv(agent: GRNAgent, csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    headers = list(agent.debug_history.keys())
    rows = zip(*(agent.debug_history[key] for key in headers))

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(value) for value in row) + "\n")


def save_debug_plot(agent: GRNAgent, plot_path: str) -> None:
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    steps = agent.debug_history["step"]

    plt.figure(figsize=(12, 7))
    plt.plot(steps, agent.debug_history["collision_impulse"], label="collision_impulse")
    plt.plot(steps, agent.debug_history["chemical_concentration"], label="chemical_concentration")
    plt.plot(steps, agent.debug_history["shear_stress"], label="shear_stress")
    plt.plot(steps, agent.debug_history["stickiness"], label="stickiness")
    plt.plot(steps, agent.debug_history["morphology"], label="morphology")
    plt.plot(steps, agent.debug_history["secretion_rate"], label="secretion_rate")
    plt.plot(steps, agent.debug_history["speed"], label="speed")

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Week 1 Day 5: Single-Agent Sensor-GRN-Behavior Validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def print_key_results(agent: GRNAgent) -> None:
    print("Final position:", agent.position)
    print("Final velocity:", agent.velocity)
    print("Final speed:", agent.debug_history["speed"][-1])
    print("Max collision impulse:", max(agent.debug_history["collision_impulse"]))
    print("Max chemical concentration:", max(agent.debug_history["chemical_concentration"]))
    print("Max shear stress:", max(agent.debug_history["shear_stress"]))
    print("Max stickiness:", max(agent.debug_history["stickiness"]))
    print("Max morphology:", max(agent.debug_history["morphology"]))
    print("Max secretion rate:", max(agent.debug_history["secretion_rate"]))
    print("Final morphology level:", agent.morphology_level)
    print("Total secreted amount:", agent.secreted_amount)
    print("Saved CSV:", CSV_PATH)
    print("Saved plot:", PLOT_PATH)


# -----------------------------
# Main simulation loop
# -----------------------------
def run_simulation():
    agent = GRNAgent()

    dt = 0.1
    steps = 100

    for step in range(steps):
        reset_sensors(agent)

        agent.sensors.collision_impulse = compute_collision_impulse(agent, step)
        agent.sensors.chemical_concentration = compute_chemical_concentration(agent, step)
        agent.sensors.shear_stress = compute_shear_stress(agent, step)

        write_sensors_to_grn(agent)
        step_agent_grn(agent, dt)
        read_grn_outputs(agent)

        apply_agent_outputs(agent, dt)
        update_position(agent, dt)

        record_agent_debug(agent, step)

    return agent


if __name__ == "__main__":
    agent = run_simulation()
    save_debug_csv(agent, CSV_PATH)
    save_debug_plot(agent, PLOT_PATH)
    print_key_results(agent)