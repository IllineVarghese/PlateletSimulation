import os
import numpy as np
import matplotlib.pyplot as plt

import src.simulation.platelet_step as platelet_step


def run_analysis(num_steps=50):
    platelet_step._SIM_STATE = None

    mean_adhesion = []
    std_adhesion = []
    min_adhesion = []
    max_adhesion = []

    mean_chemical = []
    max_chemical = []

    for step in range(num_steps):
        print(f"Running analysis step {step + 1}/{num_steps}")

        positions, adhesion_strengths = platelet_step.run_step("cpu")
        chemical_field = platelet_step._SIM_STATE["chemical_field"]

        adhesion_np = adhesion_strengths.numpy()
        field_values = chemical_field.values

        mean_adhesion.append(float(np.mean(adhesion_np)))
        std_adhesion.append(float(np.std(adhesion_np)))
        min_adhesion.append(float(np.min(adhesion_np)))
        max_adhesion.append(float(np.max(adhesion_np)))

        mean_chemical.append(float(np.mean(field_values)))
        max_chemical.append(float(np.max(field_values)))

    return {
        "mean_adhesion": np.array(mean_adhesion),
        "std_adhesion": np.array(std_adhesion),
        "min_adhesion": np.array(min_adhesion),
        "max_adhesion": np.array(max_adhesion),
        "mean_chemical": np.array(mean_chemical),
        "max_chemical": np.array(max_chemical),
    }


def save_plots(results, output_dir="results/analysis"):
    os.makedirs(output_dir, exist_ok=True)

    steps = np.arange(1, len(results["mean_adhesion"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(steps, results["mean_adhesion"], label="Mean adhesion")
    plt.fill_between(
        steps,
        results["mean_adhesion"] - results["std_adhesion"],
        results["mean_adhesion"] + results["std_adhesion"],
        alpha=0.25,
        label="±1 std"
    )
    plt.xlabel("Simulation step")
    plt.ylabel("Adhesion")
    plt.title("Adhesion dynamics with population variability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/adhesion_mean_std.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(steps, results["min_adhesion"], label="Min adhesion")
    plt.plot(steps, results["mean_adhesion"], label="Mean adhesion")
    plt.plot(steps, results["max_adhesion"], label="Max adhesion")
    plt.xlabel("Simulation step")
    plt.ylabel("Adhesion")
    plt.title("Adhesion range over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/adhesion_min_mean_max.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(steps, results["mean_chemical"], label="Mean chemical field")
    plt.plot(steps, results["max_chemical"], label="Max chemical field")
    plt.xlabel("Simulation step")
    plt.ylabel("Chemical concentration")
    plt.title("Chemical field accumulation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/chemical_accumulation.png", dpi=200)
    plt.close()

    np.savez(
        f"{output_dir}/phase3_dynamics_results.npz",
        **results
    )

    print(f"Saved analysis plots and data to: {output_dir}")


def main():
    results = run_analysis(num_steps=50)
    save_plots(results)


if __name__ == "__main__":
    main()