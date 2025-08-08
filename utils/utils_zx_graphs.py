from typing import Dict, List, Optional, Tuple
from utils.classes import SimpleDictGraph


def strip_zx_g_boundaries(
    circuit_graph_dict: SimpleDictGraph,
) -> SimpleDictGraph:
    """Strips an incoming ZX graph from "O" (boundaries) nodes and their corresponding edges.

    Args:
        - circuit_graph_dict: a ZX circuit as a simple dictionary of nodes and edges.

    Returns
        - new_circuit_graph: a new version of the incoming ZX circuit, with "O" nodes and corresponding edges removed

    """

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


def validate_zx_types(graph: SimpleDictGraph) -> bool:
    """Checks that all nodes in an incoming ZX graph have valid types.

    Args:
        - graph: a simple graph dictionary with nodes and edges representing a ZX graph.


    Returns:
        - bool:
            - True: all types are valid
            - False: there is at least one invalid type.

    """

    valid_types: List[str] = ["X", "Y", "Z", "O", "SIMPLE", "HADAMARD"]
    valid_types_lower = [key.lower() for key in [t.lower() for t in valid_types]]
    nodes_data: List[Tuple[int, str]] = graph.get("nodes", [])
    for _, node_type in nodes_data:
        if node_type.lower() not in valid_types_lower:
            print(f"Error: Node type '{node_type}' is not valid.")
            return False
    return True


def get_zx_type_fam(node_type: str) -> Optional[List[str]]:
    """Gets the family of block or pipe types/kinds that correspond to a given ZX type.

    Args:
        - node_type: the ZX type of a given node.

    Returns:
        - array: a list of possible block or pipe kinds that correspond to the node_type given to the function.

    """

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


def kind_to_zx_type(kind: str) -> str:
    """Gets the ZX type corresponding to a given block or pipe type/kind.

    Args:
        - kind: the type/kind of a given block/type.

    Returns:
        - zx_type: the ZX type that corresponds to the block or pipe type/kind given to the function.

    """

    if kind == "ooo":
        zx_type = "BOUNDARY"
    elif "o" in kind:
        zx_type = "HADAMARD" if "h" in kind else "SIMPLE"
    else:
        zx_type = max(set(kind), key=lambda c: kind.count(c)).capitalize()

    return zx_type
