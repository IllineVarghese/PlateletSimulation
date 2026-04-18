import numpy as np
import src.simulation.platelet_step as platelet_step


def main():
    num_steps = 20

    mean_chemical = []
    mean_adhesion = []

    for step in range(num_steps):
        print(f"\n=== LOOP STEP {step + 1} / {num_steps} ===")
        positions, adhesion_strengths = platelet_step.run_step("cpu")

        chemical_field = platelet_step._SIM_STATE["chemical_field"]

        adhesion_np = adhesion_strengths.numpy()
        field_values = chemical_field.values

        mean_adhesion_value = float(np.mean(adhesion_np))
        mean_chemical_value = float(np.mean(field_values))
        max_chemical_value = float(np.max(field_values))

        mean_adhesion.append(mean_adhesion_value)
        mean_chemical.append(mean_chemical_value)

        print(f"Mean adhesion: {mean_adhesion_value:.6f}")
        print(f"Mean chemical field: {mean_chemical_value:.6f}")
        print(f"Max chemical field: {max_chemical_value:.6f}")

    print("\n=== FINAL SUMMARY ===")
    for i in range(num_steps):
        print(
            f"step={i+1:02d} "
            f"mean_adhesion={mean_adhesion[i]:.6f} "
            f"mean_chemical={mean_chemical[i]:.6f}"
        )


if __name__ == "__main__":
    main()