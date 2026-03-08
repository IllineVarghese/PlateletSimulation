from dataclasses import dataclass
from typing import Dict, List


@dataclass
class GRNModel:
    node_names: List[str]
    node_index: Dict[str, int]
    edges_src: List[int]
    edges_dst: List[int]
    edges_weight: List[float]
    input_indices: List[int]
    output_indices: List[int]