import os
import numpy as np
import matplotlib.pyplot as plt

from src.grn_engine.graphml_parser import load_graphml
from src.grn_engine.agent_grn import GRNAgent


OUTPUT_DIR = "results/analysis/shear_sweep"
GRAPHML_PATH = "data/networks/platelet_squad_like_complex.graphml"

N_STEPS = 60


def run_single_shear_value(shear_value):
    model = load_graphml(GRAPHML_PATH)
    agent = GRNAgent(model)

    adhesion_history = []
    secretion_history = []
    morphology_history = []

    for step in range(N_STEPS):
        agent.set_sensor("InCollisionImpulse", 0.15)
        agent.set_sensor("InShearStress", shear_value)
        agent.set_sensor("InMolecule", 0.15)

        for _ in range(2):
            agent.step()

        adhesion_history.append(float(agent.get_output("OutStickiness")))
        secretion_history.append(float(agent.get_output("OutSecretionRate")))
        morphology_history.append(float(agent.get_output("OutCellShapeChange")))

    return {
        "adhesion": np.array(adhesion_history),
        "secretion": np.array(secretion_history),
        "morphology": np.array(morphology_history),
    }


def time_to_threshold(values, threshold=0.45):
    indices = np.where(values >= threshold)[0]
    if len(indices) == 0:
        return np.nan
    return int(indices[0] + 1)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    shear_values = np.linspace(0.0, 1.0, 11)

    final_adhesion = []
    final_secretion = []
    final_morphology = []
    activation_time = []
    peak_adhesion = []

    for shear in shear_values:
        print(f"Running shear={shear:.2f}")
        result = run_single_shear_value(float(shear))

        final_adhesion.append(float(result["adhesion"][-1]))
        final_secretion.append(float(result["secretion"][-1]))
        final_morphology.append(float(result["morphology"][-1]))
        peak_adhesion.append(float(np.max(result["adhesion"])))
        activation_time.append(time_to_threshold(result["adhesion"], threshold=0.45))

    final_adhesion = np.array(final_adhesion)
    final_secretion = np.array(final_secretion)
    final_morphology = np.array(final_morphology)
    peak_adhesion = np.array(peak_adhesion)
    activation_time = np.array(activation_time)

    plt.figure(figsize=(8, 5))
    plt.plot(shear_values, final_adhesion, marker="o", label="Final adhesion")
    plt.plot(shear_values, final_secretion, marker="o", label="Final secretion")
    plt.plot(shear_values, final_morphology, marker="o", label="Final morphology")
    plt.xlabel("Shear input")
    plt.ylabel("Final GRN output")
    plt.title("Dose-response curve: shear input vs GRN outputs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shear_dose_response_outputs.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(shear_values, peak_adhesion, marker="o")
    plt.xlabel("Shear input")
    plt.ylabel("Peak adhesion")
    plt.title("Peak adhesion response to shear")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shear_peak_adhesion.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(shear_values, activation_time, marker="o")
    plt.xlabel("Shear input")
    plt.ylabel("Time to adhesion threshold")
    plt.title("Activation timing response to shear")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shear_activation_time.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 5))
    steps = np.arange(1, N_STEPS + 1)
    for shear in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = run_single_shear_value(shear)
        plt.plot(steps, result["adhesion"], label=f"shear={shear}")
    plt.xlabel("Simulation step")
    plt.ylabel("Adhesion")
    plt.title("Adhesion dynamics under increasing shear")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shear_adhesion_timecourses.png", dpi=220)
    plt.close()

    np.savez(
        f"{OUTPUT_DIR}/shear_sweep_results.npz",
        shear_values=shear_values,
        final_adhesion=final_adhesion,
        final_secretion=final_secretion,
        final_morphology=final_morphology,
        peak_adhesion=peak_adhesion,
        activation_time=activation_time,
    )

    print(f"Saved shear sweep results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()