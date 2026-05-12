from src.grn_engine.graphml_parser import load_graphml
from src.grn_engine.agent_grn import GRNAgent


def test_stickiness_reduces_movement():
    model = load_graphml("data/networks/test_minimal.graphml")

    agent = GRNAgent(model)
    agent.set_sensor("InCollisionImpulse", 1.0)

    position = 0.0
    velocity = 1.0
    dt = 1.0

    agent.step()
    stickiness = agent.get_output("OutStickiness")

    adhesion_strength = max(0.0, min(1.0, stickiness))
    slowdown = max(0.0, min(1.0, 1.0 - adhesion_strength))

    new_position = position + velocity * slowdown * dt

    assert new_position < 1.0