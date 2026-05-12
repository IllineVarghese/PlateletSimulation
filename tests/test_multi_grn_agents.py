from src.grn_engine.graphml_parser import load_graphml
from src.grn_engine.agent_grn import GRNAgent


def test_multiple_agents_have_independent_outputs():

    model = load_graphml("data/networks/test_minimal.graphml")

    agent1 = GRNAgent(model)
    agent2 = GRNAgent(model)
    agent3 = GRNAgent(model)

    agent1.set_sensor("InCollisionImpulse", 1.0)
    agent2.set_sensor("InCollisionImpulse", 0.0)
    agent3.set_sensor("InCollisionImpulse", 1.0)

    for _ in range(5):
        agent1.step()
        agent2.step()
        agent3.step()

    stickiness1 = agent1.get_output("OutStickiness")
    stickiness2 = agent2.get_output("OutStickiness")
    stickiness3 = agent3.get_output("OutStickiness")

    assert stickiness1 >= stickiness2
    assert stickiness3 >= stickiness2

def test_agents_do_not_share_state():

    model = load_graphml("data/networks/test_minimal.graphml")

    agent1 = GRNAgent(model)
    agent2 = GRNAgent(model)

    agent1.set_sensor("InCollisionImpulse", 1.0)
    agent2.set_sensor("InCollisionImpulse", 0.0)

    agent1.step()
    agent2.step()

    assert agent1.state != agent2.state    