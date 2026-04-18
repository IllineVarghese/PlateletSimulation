from src.grn_engine.graphml_parser import load_graphml
from src.grn_engine.agent_grn import GRNAgent


def main():
    model = load_graphml("data/networks/test_minimal.graphml")

    print("Loaded nodes:")
    print(model.node_names)
    print()

    test_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    for shear in test_values:
        agent = GRNAgent(model)

        # isolate shear effect
        agent.set_sensor("InCollisionImpulse", 0.0)
        agent.set_sensor("InShearStress", shear)

        # run multiple GRN steps so response has time to develop
        for _ in range(10):
            agent.step()

        stickiness = agent.get_output("OutStickiness")

        print(f"shear={shear:.2f} -> stickiness={stickiness:.6f}")


if __name__ == "__main__":
    main()