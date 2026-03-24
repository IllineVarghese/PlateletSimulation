from src.grn_engine.graphml_parser import load_graphml
from src.grn_engine.agent_grn import GRNAgent


def test_phase3_placeholder_interface_names_exist():
    model = load_graphml("data/networks/test_minimal.graphml")
    agent = GRNAgent(model)

    agent.set_sensor("InMolecule", 0.7)
    agent.set_sensor("InShearStress", 0.3)

    shape_change = agent.get_output("OutCellShapeChange")

    assert shape_change == 0.0