import os
import csv
import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR = "results"
CSV_AGENT_A = os.path.join(RESULTS_DIR, "week3_day3_agent_A.csv")
CSV_AGENT_B = os.path.join(RESULTS_DIR, "week3_day3_agent_B.csv")
PLOT_PATH = os.path.join(RESULTS_DIR, "week3_day3_two_agents_comparison.png")

DT = 0.1
STEPS = 100

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
U_MAX = 1.0


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def compute_radial_distance(position: np.ndarray) -> float:
    return float(np.linalg.norm(position[1:3]))


def compute_poiseuille_velocity_at_position(position: np.ndarray) -> float:
    r = compute_radial_distance(position)
    if r >= VESSEL_RADIUS:
        return 0.0
    u = U_MAX * (1.0 - (r / VESSEL_RADIUS) ** 2)
    return max(0.0, u)


def compute_poiseuille_shear_from_gradient(position: np.ndarray) -> float:
    r = compute_radial_distance(position)
    raw_gradient = (2.0 * U_MAX * r) / (VESSEL_RADIUS ** 2)
    max_gradient = (2.0 * U_MAX) / VESSEL_RADIUS
    normalized_shear = raw_gradient / max_gradient
    return clamp(normalized_shear)


class DummyGRN:
    def __init__(self):
        self.nodes = {}

    def set_node(self, name: str, value: float) -> None:
        self.nodes[name] = float(value)

    def get_node(self, name: str) -> float:
        return float(self.nodes.get(name, 0.0))

    def step(self, dt: float) -> None:
        collision = self.nodes.get(GRN_INPUT_NODES["collision_impulse"], 0.0)
        chemical = self.nodes.get(GRN_INPUT_NODES["chemical_concentration"], 0.0)
        shear = self.nodes.get(GRN_INPUT_NODES["shear_stress"], 0.0)

        stickiness = clamp(0.55 * collision + 0.15 * chemical + 0.30 * shear)
        morphology = clamp(0.10 * collision + 0.30 * chemical + 0.60 * shear)
        secretion_rate = clamp(0.10 * collision + 0.60 * chemical + 0.30 * shear)

        self.nodes[GRN_OUTPUT_NODES["stickiness"]] = stickiness
        self.nodes[GRN_OUTPUT_NODES["morphology"]] = morphology
        self.nodes[GRN_OUTPUT_NODES["secretion_rate"]] = secretion_rate


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
    def __init__(self, name: str):
        self.name = name

        self.position = np.array([0.0, 0.0, 0.0], dtype=float)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)

        self.grn = DummyGRN()
        self.sensors = AgentSensors()
        self.outputs = AgentOutputs()

        self.debug_history = {
            "step": [],
            "position_x": [],
            "position_y": [],
            "position_z": [],
            "flow_velocity": [],
            "collision_impulse": [],
            "chemical_concentration": [],
            "shear_stress": [],
            "stickiness": [],
            "morphology": [],
            "secretion_rate": [],
            "speed": [],
        }


def reset_sensors(agent: GRNAgent) -> None:
    agent.sensors.collision_impulse = 0.0
    agent.sensors.chemical_concentration = 0.0
    agent.sensors.shear_stress = 0.0


def compute_collision_impulse(agent: GRNAgent, step: int) -> float:
    if agent.name == "A":
        if 20 <= step <= 30:
            return 1.0
        if 31 <= step <= 40:
            return 0.5
        return 0.0

    if agent.name == "B":
        if 50 <= step <= 60:
            return 0.8
        if 61 <= step <= 70:
            return 0.3
        return 0.0

    return 0.0


def compute_chemical_concentration(agent: GRNAgent, step: int) -> float:
    distance = np.linalg.norm(agent.position - CHEMICAL_SOURCE_POSITION)
    concentration = 1.0 - (distance / CHEMICAL_SOURCE_RADIUS)
    concentration = clamp(concentration)

    if agent.name == "B":
        concentration *= 0.5

    return clamp(concentration)


def compute_shear_stress(agent: GRNAgent, step: int) -> float:
    return compute_poiseuille_shear_from_gradient(agent.position)


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


def apply_agent_outputs(agent: GRNAgent, dt: float) -> None:
    local_flow_velocity = compute_poiseuille_velocity_at_position(agent.position)
    mobility_factor = max(0.15, 1.0 - 0.55 * agent.outputs.stickiness)

    agent.velocity[0] = local_flow_velocity * mobility_factor
    agent.velocity[1] = 0.0
    agent.velocity[2] = 0.0


def update_position(agent: GRNAgent, dt: float) -> None:
    agent.position += agent.velocity * dt


def record_agent_debug(agent: GRNAgent, step: int) -> None:
    flow_velocity = compute_poiseuille_velocity_at_position(agent.position)

    agent.debug_history["step"].append(step)
    agent.debug_history["position_x"].append(float(agent.position[0]))
    agent.debug_history["position_y"].append(float(agent.position[1]))
    agent.debug_history["position_z"].append(float(agent.position[2]))
    agent.debug_history["flow_velocity"].append(flow_velocity)
    agent.debug_history["collision_impulse"].append(agent.sensors.collision_impulse)
    agent.debug_history["chemical_concentration"].append(agent.sensors.chemical_concentration)
    agent.debug_history["shear_stress"].append(agent.sensors.shear_stress)
    agent.debug_history["stickiness"].append(agent.outputs.stickiness)
    agent.debug_history["morphology"].append(agent.outputs.morphology)
    agent.debug_history["secretion_rate"].append(agent.outputs.secretion_rate)
    agent.debug_history["speed"].append(float(np.linalg.norm(agent.velocity)))


def update_one_agent(agent: GRNAgent, step: int, dt: float) -> None:
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


def run_simulation():
    agent_A = GRNAgent("A")
    agent_A.position = np.array([0.0, 0.0, 0.0], dtype=float)

    agent_B = GRNAgent("B")
    agent_B.position = np.array([0.0, 0.6, 0.0], dtype=float)

    for step in range(STEPS):
        update_one_agent(agent_A, step, DT)
        update_one_agent(agent_B, step, DT)

    return agent_A, agent_B


def save_agent_csv(agent: GRNAgent, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    headers = list(agent.debug_history.keys())
    rows = zip(*(agent.debug_history[key] for key in headers))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def save_comparison_plot(agent_A: GRNAgent, agent_B: GRNAgent, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    steps_A = agent_A.debug_history["step"]
    steps_B = agent_B.debug_history["step"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Speed
    axes[0, 0].plot(steps_A, agent_A.debug_history["speed"], label="Agent A")
    axes[0, 0].plot(steps_B, agent_B.debug_history["speed"], label="Agent B")
    axes[0, 0].set_title("Speed")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Speed")
    axes[0, 0].legend()

    # Stickiness
    axes[0, 1].plot(steps_A, agent_A.debug_history["stickiness"], label="Agent A")
    axes[0, 1].plot(steps_B, agent_B.debug_history["stickiness"], label="Agent B")
    axes[0, 1].set_title("Stickiness")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Stickiness")
    axes[0, 1].legend()

    # Shear stress
    axes[1, 0].plot(steps_A, agent_A.debug_history["shear_stress"], label="Agent A")
    axes[1, 0].plot(steps_B, agent_B.debug_history["shear_stress"], label="Agent B")
    axes[1, 0].set_title("Shear Stress")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Shear")
    axes[1, 0].legend()

    # Chemical concentration
    axes[1, 1].plot(steps_A, agent_A.debug_history["chemical_concentration"], label="Agent A")
    axes[1, 1].plot(steps_B, agent_B.debug_history["chemical_concentration"], label="Agent B")
    axes[1, 1].set_title("Chemical Concentration")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Concentration")
    axes[1, 1].legend()

    fig.suptitle("Week 3 Day 3: Two-Agent Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_summary(agent_A: GRNAgent, agent_B: GRNAgent) -> None:
    print("\n--- Agent A ---")
    print("Final x position:", agent_A.debug_history["position_x"][-1])
    print("Final speed:", agent_A.debug_history["speed"][-1])
    print("Max shear:", max(agent_A.debug_history["shear_stress"]))
    print("Max stickiness:", max(agent_A.debug_history["stickiness"]))

    print("\n--- Agent B ---")
    print("Final x position:", agent_B.debug_history["position_x"][-1])
    print("Final speed:", agent_B.debug_history["speed"][-1])
    print("Max shear:", max(agent_B.debug_history["shear_stress"]))
    print("Max stickiness:", max(agent_B.debug_history["stickiness"]))

    print("\nSaved CSV A:", CSV_AGENT_A)
    print("Saved CSV B:", CSV_AGENT_B)
    print("Saved comparison plot:", PLOT_PATH)


if __name__ == "__main__":
    agent_A, agent_B = run_simulation()

    save_agent_csv(agent_A, CSV_AGENT_A)
    save_agent_csv(agent_B, CSV_AGENT_B)
    save_comparison_plot(agent_A, agent_B, PLOT_PATH)

    print_summary(agent_A, agent_B)