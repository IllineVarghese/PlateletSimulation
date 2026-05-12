import os
import matplotlib.pyplot as plt
import numpy as np
import src.simulation.platelet_step as platelet_step


def main():
    num_steps = 20

    mean_adhesion = []
    mean_chemical = []
    max_chemical = []

    # reset simulation state so plot run starts clean
    platelet_step._SIM_STATE = None

    for step in range(num_steps):
        print(f"Running step {step + 1}/{num_steps}")
        positions, adhesion_strengths = platelet_step.run_step("cpu")

        chemical_field = platelet_step._SIM_STATE["chemical_field"]

        adhesion_np = adhesion_strengths.numpy()
        field_values = chemical_field.values

        mean_adhesion.append(float(np.mean(adhesion_np)))
        mean_chemical.append(float(np.mean(field_values)))
        max_chemical.append(float(np.max(field_values)))

    os.makedirs("results", exist_ok=True)
    x = np.arange(1, num_steps + 1)

    # Plot 1: mean adhesion
    plt.figure(figsize=(8, 5))
    plt.plot(x, mean_adhesion)
    plt.xlabel("Simulation step")
    plt.ylabel("Mean adhesion")
    plt.title("Mean adhesion over time")
    plt.tight_layout()
    plt.savefig("results/plot_mean_adhesion_over_time.png", dpi=200)
    plt.close()

    # Plot 2: mean chemical field
    plt.figure(figsize=(8, 5))
    plt.plot(x, mean_chemical)
    plt.xlabel("Simulation step")
    plt.ylabel("Mean chemical field")
    plt.title("Mean chemical field over time")
    plt.tight_layout()
    plt.savefig("results/plot_mean_chemical_over_time.png", dpi=200)
    plt.close()

    # Plot 3: max chemical field
    plt.figure(figsize=(8, 5))
    plt.plot(x, max_chemical)
    plt.xlabel("Simulation step")
    plt.ylabel("Max chemical field")
    plt.title("Max chemical field over time")
    plt.tight_layout()
    plt.savefig("results/plot_max_chemical_over_time.png", dpi=200)
    plt.close()

    # Plot 4: combined
    plt.figure(figsize=(8, 5))
    plt.plot(x, mean_adhesion, label="Mean adhesion")
    plt.plot(x, mean_chemical, label="Mean chemical field")
    plt.xlabel("Simulation step")
    plt.ylabel("Value")
    plt.title("Adhesion and chemical field over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plot_combined_adhesion_chemical.png", dpi=200)
    plt.close()

    print("Saved plots:")
    print("results/plot_mean_adhesion_over_time.png")
    print("results/plot_mean_chemical_over_time.png")
    print("results/plot_max_chemical_over_time.png")
    print("results/plot_combined_adhesion_chemical.png")


if __name__ == "__main__":
    main()