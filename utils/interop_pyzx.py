from typing import cast, Union
from pyzx.graph.base import BaseGraph
from pyzx.graph.graph_s import GraphS
from pyzx.utils import EdgeType

from utils.classes import SimpleDictGraph


def get_dict_from_pyzx(g: Union[BaseGraph, GraphS]):
    """Extracts circuit information from a PyZX graph and dumps it into a dictionary.
    Output is slightly more complete than a `to_json` export using PyZX native capacities.

    Args:
        - g: a PyZX graph.

    Returns:
        - g_dict: a dictionary with graph info.

    """

    # EMPTY DICT FOR RESULTS
    g_dict: dict[str, dict] = {"meta": {}, "nodes": {}, "edges": {}}

    # GET AND TRANSFER DATA FROM PyZX
    try:

        # Dump graph into dict
        dict_graph = g.to_dict(include_scalar=True)

        # Add meta-information
        g_dict["meta"]["scalar"] = dict_graph["scalar"]

        # Add nodes
        for v in g.vertices():
            g_dict["nodes"][v] = {
                "pos": (0, 0, 0),
                "rot": (0, 0, 0),
                "scale": (0, 0, 0),
                "type": g.type(v).name,
                "phase": str(g.phase(v)),
                "degree": g.vertex_degree(v),
                "connections": list(g.neighbors(v)),
            }

        # Add edges
        c = 0
        for e in g.edges():
            typed_type: EdgeType = cast(EdgeType, g.edge_type(e))
            g_dict["edges"][f"e{c}"] = {
                "type": typed_type.name,
                "src": e[0],
                "tgt": e[1],
            }
            c += 1

    except Exception as e:
        print(f"Error extracting info from graph: {e}")

    return g_dict


def pyzx_g_to_simple_g(g: Union[BaseGraph, GraphS]) -> SimpleDictGraph:
    """Extracts circuit information from a PyZX graph and dumps it into a simple graph.

    Args:
        - g: a PyZX graph.

    Returns:
        - g_simple: a dictionary with graph info.

    """

    # GET FULL GRAPH INTO DICTIONARY
    g_full = get_dict_from_pyzx(g)

    # TRANSFER INTO A SIMPLE GRAPH
    g_simple: SimpleDictGraph = {"nodes": [], "edges": []}
    for n in g_full["nodes"]:
        n_type = (
            "O"
            if g_full["nodes"][n]["type"] == "BOUNDARY"
            else g_full["nodes"][n]["type"]
        )
        g_simple["nodes"].append((n, n_type))

    for e in g_full["edges"]:
        src = g_full["edges"][e]["src"]
        tgt = g_full["edges"][e]["tgt"]
        e_type = g_full["edges"][e]["type"]
        g_simple["edges"].append(((src, tgt), e_type))

    return g_simple
