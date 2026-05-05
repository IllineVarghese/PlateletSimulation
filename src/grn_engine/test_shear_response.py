import csv
from pathlib import Path

from src.grn_engine.agent_grn import GRNAgent
from src.grn_engine.graphml_parser import load_graphml


OUTPUT_DIR = Path("results/analysis/shear_response_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment(model, condition_name, collision, chemical):
    print(f"\n=== Running {condition_name} ===")

    test_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []

    for shear in test_values:
        agent = GRNAgent(model)

        agent.set_sensor("InCollisionImpulse", collision)
        agent.set_sensor("InChemicalConcentration", chemical)
        agent.set_sensor("InShearStress", shear * 3.0)

        for _ in range(80):
            agent.step()

        stickiness = agent.get_output("OutStickiness")
        secretion = agent.get_output("OutSecretionRate")
        morphology = agent.get_output("OutMorphologyChange")

        print(
            f"shear={shear:.2f} -> "
            f"stickiness={stickiness:.4f}, "
            f"secretion={secretion:.4f}, "
            f"morphology={morphology:.4f}"
        )

        results.append([shear, stickiness, secretion, morphology])

    filename = OUTPUT_DIR / f"{condition_name}_shear_response.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["shear", "stickiness", "secretion", "morphology"])
        writer.writerows(results)

    print(f"Saved: {filename}")


def main():
    model = load_graphml("data/networks/platelet_squad_like_complex.graphml")

    run_experiment(
        model,
        condition_name="high_background",
        collision=0.8,
        chemical=0.6,
    )

    run_experiment(
        model,
        condition_name="low_background",
        collision=0.05,
        chemical=0.02,
    )


if __name__ == "__main__":
    main()