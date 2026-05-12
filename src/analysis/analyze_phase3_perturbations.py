import os
import numpy as np
import matplotlib.pyplot as plt

from src.grn_engine.graphml_parser import load_graphml
from src.grn_engine.agent_grn import GRNAgent
from src.simulation.chemical_field import ChemicalField


N_AGENTS = 40
N_STEPS = 50
DT = 0.01
OUTPUT_DIR = "results/analysis/perturbations"


def compute_shear_from_position(pos, mode="baseline"):
    y = pos[1]
    z = pos[2]
    radius = 0.5

    r = np.sqrt((y - 0.5) ** 2 + (z - 0.5) ** 2)
    shear = min(1.0, max(0.0, r / radius))

    if mode == "low_shear":
        shear *= 0.4
    elif mode == "high_shear":
        shear = min(1.0, shear * 1.8)

    return shear


def run_condition(condition_name):
    rng = np.random.default_rng(42)

    positions = rng.random((N_AGENTS, 3), dtype=np.float32)

    model = load_graphml("data/networks/platelet_squad_like_complex.graphml")
    agents = [GRNAgent(model) for _ in range(N_AGENTS)]

    chemical_field = ChemicalField(nx=20, ny=20, nz=20, spacing=0.05, decay_rate=0.1)

    mean_adhesion = []
    mean_chemical = []
    max_chemical = []

    for step in range(N_STEPS):
        chemical_field.decay(DT)

        adhesion_values = []

        for i, agent in enumerate(agents):
            pos = positions[i]

            collision_input = 0.0
            shear_input = compute_shear_from_position(pos, mode=condition_name)

            if condition_name == "no_chemical_feedback":
                chemical_input = 0.0
            else:
                chemical_input = chemical_field.sample(pos)

            agent.set_sensor("InCollisionImpulse", collision_input)
            agent.set_sensor("InShearStress", shear_input)
            agent.set_sensor("InMolecule", chemical_input)

            for _ in range(3):
                agent.step()

            adhesion = float(agent.get_output("OutStickiness"))
            secretion = float(agent.get_output("OutSecretionRate"))

            adhesion_values.append(adhesion)

            if condition_name != "no_secretion":
                secretion_amount = max(0.0, secretion) * DT
                chemical_field.deposit(pos, secretion_amount)

        mean_adhesion.append(float(np.mean(adhesion_values)))
        mean_chemical.append(float(np.mean(chemical_field.values)))
        max_chemical.append(float(np.max(chemical_field.values)))

    return {
        "mean_adhesion": np.array(mean_adhesion),
        "mean_chemical": np.array(mean_chemical),
        "max_chemical": np.array(max_chemical),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conditions = [
        "baseline",
        "low_shear",
        "high_shear",
        "no_chemical_feedback",
        "no_secretion",
    ]

    all_results = {}

    for condition in conditions:
        print(f"Running condition: {condition}")
        all_results[condition] = run_condition(condition)

    steps = np.arange(1, N_STEPS + 1)

    # Plot 1: adhesion comparison
    plt.figure(figsize=(9, 5))
    for condition in conditions:
        plt.plot(
            steps,
            all_results[condition]["mean_adhesion"],
            label=condition,
        )
    plt.xlabel("Simulation step")
    plt.ylabel("Mean adhesion")
    plt.title("Perturbation analysis: adhesion response")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/perturbation_adhesion_response.png", dpi=200)
    plt.close()

    # Plot 2: mean chemical comparison
    plt.figure(figsize=(9, 5))
    for condition in conditions:
        plt.plot(
            steps,
            all_results[condition]["mean_chemical"],
            label=condition,
        )
    plt.xlabel("Simulation step")
    plt.ylabel("Mean chemical field")
    plt.title("Perturbation analysis: mean chemical accumulation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/perturbation_mean_chemical.png", dpi=200)
    plt.close()

    # Plot 3: max chemical comparison
    plt.figure(figsize=(9, 5))
    for condition in conditions:
        plt.plot(
            steps,
            all_results[condition]["max_chemical"],
            label=condition,
        )
    plt.xlabel("Simulation step")
    plt.ylabel("Max chemical field")
    plt.title("Perturbation analysis: local chemical hotspots")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/perturbation_max_chemical.png", dpi=200)
    plt.close()

    # Save numerical data
    np.savez(
        f"{OUTPUT_DIR}/perturbation_results.npz",
        **{
            f"{condition}_{metric}": values
            for condition, result in all_results.items()
            for metric, values in result.items()
        },
    )

    print(f"Saved perturbation analysis to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()