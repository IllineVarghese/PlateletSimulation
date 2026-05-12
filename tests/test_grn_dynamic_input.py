from src.grn_engine.graphml_parser import load_graphml
from src.grn_engine.agent_grn import GRNAgent


def test_agent_reacts_to_collision_spike():

    model = load_graphml("data/networks/test_minimal.graphml")

    agent = GRNAgent(model)

    stickiness_history = []

    collision_pattern = [0, 0, 1, 1, 0]

    for collision in collision_pattern:

        agent.set_sensor("InCollisionImpulse", collision)

        agent.step()

        stickiness_history.append(
            agent.get_output("OutStickiness")
        )

    assert len(stickiness_history) == 5

    assert stickiness_history[2] >= stickiness_history[1]