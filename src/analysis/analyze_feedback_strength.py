import os
import numpy as np
import matplotlib.pyplot as plt

from src.grn_engine.graphml_parser import load_graphml
from src.grn_engine.agent_grn import GRNAgent
from src.simulation.chemical_field import ChemicalField


OUTPUT_DIR = "results/analysis/feedback_sweep"
GRAPHML_PATH = "data/networks/platelet_squad_like_complex.graphml"

N_AGENTS = 40
N_STEPS = 80
DT = 0.02


def compute_shear(pos):
    y, z = pos[1], pos[2]
    r = np.sqrt((y - 0.5) ** 2 + (z - 0.5) ** 2)
    return min(1.0, max(0.0, r / 0.5))


def run_feedback_condition(feedback_strength):
    rng = np.random.default_rng(42)
    positions = rng.random((N_AGENTS, 3), dtype=np.float32)

    model = load_graphml(GRAPHML_PATH)
    agents = [GRNAgent(model) for _ in range(N_AGENTS)]
    chemical_field = ChemicalField(nx=20, ny=20, nz=20, spacing=0.05, decay_rate=0.2)

    mean_adhesion = []
    mean_chemical = []
    max_chemical = []

    for step in range(N_STEPS):
        chemical_field.decay(DT)

        adhesion_values = []

        for i, agent in enumerate(agents):
            pos = positions[i]

            collision_input = 0.2 if i % 7 == 0 else 0.0
            shear_input = compute_shear(pos)

            raw_chemical = chemical_field.sample(pos)
            chemical_input = min(1.0, max(0.0, raw_chemical * feedback_strength * 20.0))

            agent.set_sensor("InCollisionImpulse", collision_input)
            agent.set_sensor("InShearStress", shear_input)
            agent.set_sensor("InMolecule", chemical_input)

            for _ in range(2):
                agent.step()

            adhesion = float(agent.get_output("OutStickiness"))
            secretion = float(agent.get_output("OutSecretionRate"))

            adhesion_values.append(adhesion)

            secretion_amount = max(0.0, secretion) * DT * 0.8
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

    feedback_values = [0.0, 0.25, 0.5, 1.0, 2.0]
    results = {}

    for strength in feedback_values:
        print(f"Running feedback strength = {strength}")
        results[strength] = run_feedback_condition(strength)

    steps = np.arange(1, N_STEPS + 1)

    plt.figure(figsize=(9, 5))
    for strength in feedback_values:
        plt.plot(steps, results[strength]["mean_adhesion"], label=f"feedback={strength}")
    plt.xlabel("Simulation step")
    plt.ylabel("Mean adhesion")
    plt.title("Feedback strength sweep: adhesion dynamics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feedback_sweep_adhesion.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 5))
    for strength in feedback_values:
        plt.plot(steps, results[strength]["mean_chemical"], label=f"feedback={strength}")
    plt.xlabel("Simulation step")
    plt.ylabel("Mean chemical field")
    plt.title("Feedback strength sweep: chemical accumulation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feedback_sweep_mean_chemical.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 5))
    for strength in feedback_values:
        plt.plot(steps, results[strength]["max_chemical"], label=f"feedback={strength}")
    plt.xlabel("Simulation step")
    plt.ylabel("Max chemical field")
    plt.title("Feedback strength sweep: local chemical hotspots")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feedback_sweep_max_chemical.png", dpi=220)
    plt.close()

    np.savez(
        f"{OUTPUT_DIR}/feedback_sweep_results.npz",
        **{
            f"feedback_{strength}_{metric}": values
            for strength, result in results.items()
            for metric, values in result.items()
        },
    )

    print(f"Saved feedback sweep results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()