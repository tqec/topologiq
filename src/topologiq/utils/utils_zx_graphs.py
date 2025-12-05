"""Util facilities to aid usage and manipulation of ZX graphs.

Usage:
    Call any function/class from a separate script.

"""

import math

from topologiq.utils.classes import SimpleDictGraph


def strip_boundaries(c_g: SimpleDictGraph) -> SimpleDictGraph:
    """Strip an incoming ZX graph from "O" (boundaries) nodes and their corresponding edges.

    Args:
        c_g: ZX circuit as a simple dictionary of nodes and edges.

    Returns:
        new_c_g: a new ZX circuit with "O" nodes and corresponding edges removed

    """

    ids: list[int] = []
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
    """Check that all nodes in an incoming ZX graph have valid types.

    Args:
        - g: ZX graph as a simple dictionary with nodes and edges.

    Returns:
        (bool): True if types are valid else False.

    """

    ok: list[str] = ["X", "Y", "Z", "O", "SIMPLE", "HADAMARD"]
    nodes: list[tuple[int, str]] = g.get("nodes", [])
    for _, t in nodes:
        if t.upper() not in ok:
            print(f"Error: Node type '{t}' is not valid.")
            return False
    return True


def get_zx_type_fam(t: str) -> list[str | None]:
    """Get the family of block or pipe types/kinds that correspond to a given ZX type.

    Args:
        t: the ZX type of a given node.

    Returns:
        (array): a list of possible block or pipe kinds that correspond to the t given to the function.

    """

    fams: dict[str, list[str]] = {
        "X": ["zzx", "zxz", "xzz"],
        "Y": ["yyy"],
        "Z": ["zxx", "xxz", "xzx"],
        "O": ["ooo"],
        "SIMPLE": ["zxo", "xzo", "oxz", "ozx", "xoz", "zox"],
        "HADAMARD": ["zxoh", "xzoh", "oxzh", "ozxh", "xozh", "zoxh"],
    }

    if t not in fams:
        print(f"Warning: type '{t}' not found.")
        return None

    return fams[t]


def kind_to_zx_type(k: str) -> str:
    """Get the ZX type corresponding to a given block or pipe kind.

    Args:
        k: the /kind of a given block.

    Returns:
        zx_t: the ZX type corresponding to the kind.

    """

    if k == "ooo":
        zx_t = "BOUNDARY"
    elif "o" in k:
        zx_t = "HADAMARD" if "h" in k else "SIMPLE"
    else:
        zx_t = min(set(k), key=lambda c: k.count(c)).capitalize()
    return zx_t


def break_single_spider_graph(simple_graph: SimpleDictGraph) -> SimpleDictGraph:
    """Break single spider graph into graph with min. num. spiders needed for lattice surgery.

    Args:
        simple_graph: a ZX circuit as a simple dictionary of nodes and edges.

    Returns:
        new_simple_graph: a new circuit where the central node is broken in min. num. spiders needed for lattice surgery.

    """

    # SPLIT SPIDERS INTO PRIMARY AND BOUNDARY SPIDERS
    spiders: list[tuple[int, str]] = []
    boundaries: list[tuple[int, str]] = []
    new_spiders: list[tuple[int, str]] = []
    new_edges: list[tuple[tuple[int, int], str]] = []
    new_simple_graph: SimpleDictGraph = {"nodes": [], "edges": []}

    for spider_id, zx_type in simple_graph["nodes"]:
        boundaries.append((spider_id, zx_type)) if zx_type == "O" else spiders.append((spider_id, zx_type))

    # DETERMINE IF GRAPH HAS ONLY ONE COLOUR SPIDER
    single_spider_graph = True if len(spiders) == 1 else False

    # PROCESS SINGLE SPIDER GRAPH
    if single_spider_graph is True:

        # Get spider ZX_type
        zx_type = spiders[0][1]

        # Calculate number of required spiders
        n_spiders_required = max(1, math.ceil(2 + (len(boundaries)-6)/2))

        # Add required number of spiders
        for i in range(1,n_spiders_required+1):
            new_spiders.append((i, zx_type))
            if i != n_spiders_required:
                new_edges.append(((i,i+1), "SIMPLE"))

        # Connect boundaries
        current_spider_id = 1
        current_boundary_id = n_spiders_required + 1
        zx_type = "O"

        for i, (_, edge_type) in enumerate(simple_graph["edges"]):
            num_edges_current_spider = sum([current_spider_id in (src, tgt) for ((src, tgt), _) in new_edges])

            if num_edges_current_spider >= 4:
                current_spider_id += 1

            new_spiders.append((current_boundary_id, zx_type))
            new_edges.append(((current_spider_id, current_boundary_id), edge_type))
            current_boundary_id += 1

    # ASSEMBLE NEW SIMPLE GRAPH
    new_simple_graph["nodes"] = new_spiders if new_spiders else simple_graph["nodes"]
    new_simple_graph["edges"] = new_edges if new_edges else simple_graph["edges"]

    # RETURN
    return new_simple_graph
