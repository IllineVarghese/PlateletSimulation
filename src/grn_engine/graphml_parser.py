from pathlib import Path
import xml.etree.ElementTree as ET

from src.grn_engine.grn_model import GRNModel


INPUT_ALIASES = {
    "InCollisionImpulse": "InCollisionImpulse",
    "InImpulse": "InCollisionImpulse",
    "InChemicalConcentration": "InChemicalConcentration",
    "InMolecule": "InChemicalConcentration",
    "InShearStress": "InShearStress",
}

OUTPUT_ALIASES = {
    "OutStickiness": "OutStickiness",
    "OutMorphologyChange": "OutMorphologyChange",
    "OutCellShapeChange": "OutMorphologyChange",
    "OutSecretionRate": "OutSecretionRate",
}


def canonical_node_name(name: str) -> str:
    if name in INPUT_ALIASES:
        return INPUT_ALIASES[name]
    if name in OUTPUT_ALIASES:
        return OUTPUT_ALIASES[name]
    return name


def load_graphml(path: str) -> GRNModel:
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()

    namespace = {"g": "http://graphml.graphdrawing.org/xmlns"}

    graph = root.find("g:graph", namespace)
    if graph is None:
        raise ValueError("No <graph> element found in GraphML file.")

    raw_node_ids = []
    node_names = []

    for node in graph.findall("g:node", namespace):
        raw_id = node.attrib["id"]
        canonical_name = canonical_node_name(raw_id)

        raw_node_ids.append(raw_id)
        node_names.append(canonical_name)

    node_index = {name: i for i, name in enumerate(node_names)}
    raw_id_to_index = {raw_id: i for i, raw_id in enumerate(raw_node_ids)}

    edges_src = []
    edges_dst = []
    edges_weight = []

    for edge in graph.findall("g:edge", namespace):
        source_raw = edge.attrib["source"]
        target_raw = edge.attrib["target"]

        if source_raw not in raw_id_to_index:
            raise ValueError(f"Unknown edge source node: {source_raw}")
        if target_raw not in raw_id_to_index:
            raise ValueError(f"Unknown edge target node: {target_raw}")

        source_idx = raw_id_to_index[source_raw]
        target_idx = raw_id_to_index[target_raw]

        weight = 1.0

        for data in edge.findall("g:data", namespace):
            text = (data.text or "").strip()
            if text:
                try:
                    weight = float(text)
                    break
                except ValueError:
                    pass

        edges_src.append(source_idx)
        edges_dst.append(target_idx)
        edges_weight.append(weight)

    input_indices = []
    output_indices = []

    for i, name in enumerate(node_names):
        if name in set(INPUT_ALIASES.values()):
            input_indices.append(i)
        if name in set(OUTPUT_ALIASES.values()):
            output_indices.append(i)

    return GRNModel(
        node_names=node_names,
        node_index=node_index,
        edges_src=edges_src,
        edges_dst=edges_dst,
        edges_weight=edges_weight,
        input_indices=input_indices,
        output_indices=output_indices,
    )