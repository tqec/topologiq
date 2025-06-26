import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from two_stage_greedy_bfs import main
from utils.classes import SimpleDictGraph


# USAGE EXAMPLE: STEANE
# The code below tests algorithm with a 7 qubit steane code inputed manually as a non-descript graph.
if __name__ == "__main__":
    import time

    t1 = time.time()
    steane: SimpleDictGraph = {
        "nodes": [
            (1, "X"),
            (2, "Z"),
            (3, "Z"),
            (4, "Z"),
            (5, "X"),
            (6, "X"),
            (7, "X"),
            (8, "X"),
            (9, "X"),
            (10, "X"),
            (11, "X"),
            (12, "Z"),
            (13, "Z"),
            (14, "Z"),
        ],
        "edges": [
            ((1, 2), "SIMPLE"),
            ((1, 3), "SIMPLE"),
            ((1, 4), "SIMPLE"),
            ((5, 2), "SIMPLE"),
            ((5, 3), "SIMPLE"),
            ((6, 2), "SIMPLE"),
            ((6, 4), "SIMPLE"),
            ((7, 3), "SIMPLE"),
            ((7, 4), "SIMPLE"),
            ((8, 1), "SIMPLE"),
            ((9, 5), "SIMPLE"),
            ((10, 6), "SIMPLE"),
            ((11, 7), "SIMPLE"),
            ((2, 12), "SIMPLE"),
            ((3, 13), "SIMPLE"),
            ((4, 14), "SIMPLE"),
        ],
    }

    # CALL TO ALGORITHM
    nx_graph_3d, edge_paths, new_nx_graph = main(steane)
    t2 = time.time()

    # PRINTOUT OF RESULTS
    print("\nNodes:")
    for node_id, data in nx_graph_3d.nodes(data=True):
        print(f"  Node ID: {node_id}, Attributes: {data}")

    print("\nEdges:")
    for u, v, data in nx_graph_3d.edges(data=True):
        print(f" Edge: ({u}, {v}), Attributes: {data}")

    print("\nEdge paths:")
    for key, edge_path in edge_paths.items():
        print(f"  {key}: {edge_path['path_nodes']}")

    print("duration:", (t2 - t1)/60, "min")
