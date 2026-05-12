import csv
from pathlib import Path

import matplotlib.pyplot as plt

from src.grn_engine.graphml_parser import load_graphml
from src.grn_engine.agent_grn import GRNAgent


OUTPUT_DIR = Path("results/analysis/grn_paper_style")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRAPHML_FILE = "data/networks/platelet_squad_like_complex.graphml"


def run_grn_timecourse(condition_name, collision, chemical, shear, steps=120):
    model = load_graphml(GRAPHML_FILE)
    agent = GRNAgent(model)

    rows = []

    for t in range(steps):
        agent.set_sensor("InCollisionImpulse", collision)
        agent.set_sensor("InChemicalConcentration", chemical)
        agent.set_sensor("InShearStress", shear)

        agent.step()

        row = {"time": t}

        for name, value in zip(model.node_names, agent.state):
            row[name] = float(value)

        rows.append(row)

    return rows, model.node_names


def save_csv(rows, filename):
    if not rows:
        return

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_selected_nodes(rows, nodes, title, filename):
    time = [r["time"] for r in rows]

    plt.figure(figsize=(12, 7))

    for node in nodes:
        if node in rows[0]:
            values = [r[node] for r in rows]
            plt.plot(time, values, marker="o", markersize=3, linewidth=2, label=node)

    plt.xlabel("Time step")
    plt.ylabel("Node state")
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.35)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_all_nodes(rows, node_names, title, filename):
    time = [r["time"] for r in rows]

    plt.figure(figsize=(14, 8))

    for node in node_names:
        values = [r[node] for r in rows]
        plt.plot(time, values, linewidth=1.4, alpha=0.85, label=node)

    plt.xlabel("Time step")
    plt.ylabel("Node state")
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def main():
    key_nodes = [
        "InCollisionImpulse",
        "InShearStress",
        "InChemicalConcentration",
        "vWF_GPIb_ShearSensing",
        "GPVI_CollagenSignal",
        "Mechanosensitive_Ca2_Entry",
        "ADP_P2Y12_Receptor",
        "Thromboxane_TXA2_Signal",
        "PI3K_Akt_Pathway",
        "PLCgamma2",
        "IP3_Ca2_Signaling",
        "PKC",
        "Rap1",
        "PlateletActivation",
        "TalinKindlin_Activation",
        "Integrin_alphaIIb_beta3",
        "AdhesionProgram",
        "DenseGranuleSecretion",
        "ADP_TXA2_Feedback",
        "SecretionProgram",
        "RhoA_ROCK_Cytoskeleton",
        "ActinRemodeling",
        "MorphologyProgram",
        "OutStickiness",
        "OutSecretionRate",
        "OutMorphologyChange",
    ]

    conditions = [
        {
            "name": "low_shear_low_chemical",
            "collision": 0.2,
            "chemical": 0.05,
            "shear": 0.15,
        },
        {
            "name": "high_shear_low_chemical",
            "collision": 0.2,
            "chemical": 0.05,
            "shear": 0.85,
        },
        {
            "name": "high_shear_high_chemical",
            "collision": 0.4,
            "chemical": 0.35,
            "shear": 0.85,
        },
    ]

    for condition in conditions:
        rows, node_names = run_grn_timecourse(
            condition_name=condition["name"],
            collision=condition["collision"],
            chemical=condition["chemical"],
            shear=condition["shear"],
            steps=120,
        )

        csv_path = OUTPUT_DIR / f"{condition['name']}_grn_states.csv"
        save_csv(rows, csv_path)

        plot_selected_nodes(
            rows,
            key_nodes,
            title=f"GRN node dynamics: {condition['name']}",
            filename=OUTPUT_DIR / f"{condition['name']}_selected_nodes.png",
        )

        plot_all_nodes(
            rows,
            node_names,
            title=f"All GRN node states: {condition['name']}",
            filename=OUTPUT_DIR / f"{condition['name']}_all_nodes.png",
        )

        print(f"Saved results for {condition['name']}")

    print(f"\nAll paper-style GRN plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()