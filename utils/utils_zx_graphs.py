from typing import Dict, List, Optional, Tuple
from utils.classes import SimpleDictGraph


def strip_boundaries(
    c_g: SimpleDictGraph,
) -> SimpleDictGraph:
    """ Strips an incoming ZX graph from "O" (boundaries) nodes and their corresponding edges.
    Args:
        - c_g: ZX circuit as a simple dictionary of nodes and edges.
    Returns
        - new_c_g: a new ZX circuit with "O" nodes and corresponding edges removed
    """

    ids: List[int] = []
    new_c_g: SimpleDictGraph = {"nodes": [], "edges": []}

    for n in c_g["nodes"]:
        if n[1] != "O":
            new_c_g["nodes"].append(n)
        else:
            ids.append(n[0])

    for e in c_g["edges"]:
        if not any([e in ids for e in e[0]]):
            new_c_g["edges"].append(e)

    return new_c_g


def check_zx_types(g: SimpleDictGraph) -> bool:
    """Checks that all nodes in an incoming ZX graph have valid types.
    Args:
        - g: ZX graph as a simple dictionary with nodes and edges.
    Returns:
        - bool:
            - True: types are valid
            - False: at least one invalid type.
    """

    ok: List[str] = ["X", "Y", "Z", "O", "SIMPLE", "HADAMARD"]
    nodes: List[Tuple[int, str]] = g.get("nodes", [])
    for _, t in nodes:
        if t.upper() not in ok:
            print(f"Error: Node type '{t}' is not valid.")
            return False
    return True


def get_zx_type_fam(t: str) -> Optional[List[str]]:
    """Gets the family of block or pipe types/kinds that correspond to a given ZX type.
    Args:
        - t: the ZX type of a given node.
    Returns:
        - (array): a list of possible block or pipe kinds that correspond to the t given to the function.

    """

    fams: Dict[str, List[str]] = {
        "X": ["xxz", "xzx", "zxx"],
        "Y": ["yyy"],
        "Z": ["xzz", "zzx", "zxz"],
        "O": ["ooo"],
        "SIMPLE": ["zxo", "xzo", "oxz", "ozx", "xoz", "zox"],
        "HADAMARD": ["zxoh", "xzoh", "oxzh", "ozxh", "xozh", "zoxh"],
    }

    if t not in fams:
        print(f"Warning: type '{t}' not found.")
        return None

    return fams[t]


def kind_to_zx_type(k: str) -> str:
    """Gets the ZX type corresponding to a given block or pipe kind.
    Args:
        - k: the /kind of a given block.
    Returns:
        - zx_t: the ZX type corresponding to the kind.
    """

    if k == "ooo":
        zx_t = "BOUNDARY"
    elif "o" in k:
        zx_t = "HADAMARD" if "h" in k else "SIMPLE"
    else:
        zx_t = max(set(k), key=lambda c: k.count(c)).capitalize()

    return zx_t
