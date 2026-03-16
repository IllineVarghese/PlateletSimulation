from src.grn_engine.graphml_parser import load_graphml
from src.grn_engine.agent_grn import GRNAgent


def test_single_agent_simulation_loop():

    model = load_graphml("data/networks/test_minimal.graphml")

    agent = GRNAgent(model)

    stickiness_history = []

    for step in range(5):

        agent.set_sensor("InCollisionImpulse", 1.0)

        agent.step()

        stickiness_history.append(
            agent.get_output("OutStickiness")
        )

    assert len(stickiness_history) == 5

    assert stickiness_history[-1] >= stickiness_history[0]