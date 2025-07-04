from typing import List

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
