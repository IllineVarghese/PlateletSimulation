from src.grn_engine.graphml_parser import load_graphml
from src.grn_engine.agent_grn import GRNAgent


def test_agent_collision_increases_stickiness():

    model = load_graphml("data/networks/test_minimal.graphml")

    agent = GRNAgent(model)

    agent.set_sensor("InCollisionImpulse", 1.0)

    agent.step()

    stickiness = agent.get_output("OutStickiness")

    assert stickiness > 0.0

def test_agent_state_persists_across_steps():

    model = load_graphml("data/networks/test_minimal.graphml")

    agent = GRNAgent(model)

    agent.set_sensor("InCollisionImpulse", 1.0)

    agent.step()
    stickiness1 = agent.get_output("OutStickiness")
    state1 = agent.state.copy()

    agent.step()
    stickiness2 = agent.get_output("OutStickiness")
    state2 = agent.state.copy()

    assert stickiness2 >= stickiness1
    assert state2 != state1    