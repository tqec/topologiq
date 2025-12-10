"""Util facilities to assist PyZX interoperability.

Usage:
    Call any function/class from a separate script.

"""

from pathlib import Path
from typing import cast

import pyzx as zx
from pyzx.graph.base import BaseGraph
from pyzx.graph.graph_s import GraphS
from pyzx.utils import EdgeType

from topologiq.utils.classes import SimpleDictGraph


#########################
# PyZX METHODS WRAPPERS #
#########################
def pyzx_to_qasm(pyzx_circuit: zx.Circuit, circuit_name:str, save_dir_path: Path) -> str:
    """Export a PyZX graph to QASM.

    Args:
        pyzx_circuit: A PyZX circuit.
        circuit_name: The name of the circuit being saved.
        save_dir_path: A path specifying the destination folder.

    """
    # QASM --> PyZX circuit --> PyZX graph
    qasm_str = zx.Circuit.to_qasm(pyzx_circuit)
    file_path = save_dir_path / f"{circuit_name}.qasm"
    with open(file_path, "w") as f:
        f.write(qasm_str)
        f.close

    return qasm_str

def qasm_to_pyzx(qasm_str:str) -> BaseGraph:
    """Import a circuit from QASM and convert it to a PyZX graph.

    Args:
        qasm_str: A quantum circuit encoded as a QASM string.

    """
    # QASM --> PyZX circuit --> PyZX graph
    zx_circuit = zx.Circuit.from_qasm(qasm_str)
    pyzx_graph = zx_circuit.to_graph()

    return zx_circuit, pyzx_graph


########################
# EXTRACT & MANIPULATE #
########################
def get_dict_from_pyzx(g: BaseGraph | GraphS):
    """Extract circuit information from a PyZX graph and dumps it into a dictionary.

    Args:
        g: a PyZX graph.

    Returns:
        g_dict: a dictionary with graph info.

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
                "coords": (0, 0, 0),
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


def pyzx_g_to_simple_g(g: BaseGraph | GraphS) -> SimpleDictGraph:
    """Extract circuit information from a PyZX graph and dumps it into a simple graph.

    Args:
        g: a PyZX graph.

    Returns:
        g_simple: a dictionary with graph info.

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
