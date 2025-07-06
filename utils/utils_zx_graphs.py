from typing import Dict, List, Optional, Tuple

from utils.classes import SimpleDictGraph


def strip_boundaries_from_zx_graph(
    circuit_graph_dict: SimpleDictGraph,
) -> SimpleDictGraph:

    boundary_ids: List[int] = []
    new_circuit_graph: SimpleDictGraph = {"nodes": [], "edges": []}

    for node in circuit_graph_dict["nodes"]:
        if node[1] != "O":
            new_circuit_graph["nodes"].append(node)
        else:
            boundary_ids.append(node[0])

    for edge in circuit_graph_dict["edges"]:
        if not any([e in boundary_ids for e in edge[0]]):
            new_circuit_graph["edges"].append(edge)

    return new_circuit_graph


def zx_types_validity_checks(graph: SimpleDictGraph) -> bool:

    valid_types: List[str] = ["X", "Y", "Z", "O", "SIMPLE", "HADAMARD"]
    valid_types_lower = [key.lower() for key in [t.lower() for t in valid_types]]
    nodes_data: List[Tuple[int, str]] = graph.get("nodes", [])
    for _, node_type in nodes_data:
        if node_type.lower() not in valid_types_lower:
            print(f"Error: Node type '{node_type}' is not valid.")
            return False
    return True


def get_type_family(node_type: str) -> Optional[List[str]]:

    families: Dict[str, List[str]] = {
        "X": ["xxz", "xzx", "zxx"],
        "Y": ["yyy"],
        "Z": ["xzz", "zzx", "zxz"],
        "O": ["ooo"],
        "SIMPLE": ["zxo", "xzo", "oxz", "ozx", "xoz", "zox"],
        "HADAMARD": ["zxoh", "xzoh", "oxzh", "ozxh", "xozh", "zoxh"],
    }

    if node_type not in families:
        print(f"Warning: type '{node_type}' not found.")
        return None

    return families[node_type]


def get_zx_type_from_kind(kind: str) -> str:

    if kind == "ooo":
        zx_type = "BOUNDARY"
    elif "o" in kind:
        zx_type = "HADAMARD" if "h" in kind else "SIMPLE"
    else:
        zx_type = max(set(kind), key=lambda c: kind.count(c)).capitalize()

    return zx_type
