"""Transformations used to assist the primary graph managemer BFS algorithm.

Usage:
    Call any function/class from a separate script.

"""

import networkx as nx

from topologiq.core.graph_manager.query import get_node_degree
from topologiq.utils.classes import StandardBlock, StandardCoord


############
# NX GRAPH #
############
def prune_beams(nx_g: nx.Graph, taken: list[StandardCoord]) -> nx.Graph:
    """Remove beams that have already been broken.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        taken: A list of coordinates taken by any blocks placed as a result of previous operations.

    Returns:
        nx_g: Updated nx_graph with pruned beams.

    """

    try:
        for n_id in nx_g.nodes():
            new_beams = []
            new_beams_short = []
            if nx_g.nodes[n_id]["completed"] == []:
                pass
            elif nx_g.nodes[n_id]["completed"] >= get_node_degree(nx_g, n_id):
                nx_g.nodes[n_id]["beams"] = []
                nx_g.nodes[n_id]["beams_short"] = []
            else:
                old_beams = nx_g.nodes[n_id]["beams"]
                old_beams_short = nx_g.nodes[n_id]["beams_short"]

                if old_beams:
                    for single_beam in old_beams:
                        if not any([single_beam.contains(coord) for coord in taken]):
                            new_beams += [single_beam]
                    nx_g.nodes[n_id]["beams"] = new_beams

                if old_beams_short:
                    for single_beam_short in old_beams_short:
                        if not any([single_beam_short.contains(coord) for coord in taken]):
                            new_beams_short += [single_beam_short]
                    nx_g.nodes[n_id]["beams_short"] = new_beams_short

    except (IndexError, ValueError, LookupError, KeyError):
        pass

    return nx_g


##############
# CONVERSION #
##############
def reindex_path_dict(
    edge_paths: dict, fix_errors: bool = False
) -> tuple[dict[int, StandardBlock], dict[tuple[int, int], list[str]]]:
    """Distil an edge_path object into a list of nodes/blocks and edges/pipes for the space-time diagram.

    This function converts an edge_paths dictionary into a list of cubes and pipes in the final space-time diagram.
    The function can be called during the process to output a temporary snapshot of progress, or at the end
    to produce final results. If used to convert objects for visualisation, the `fix_errors` optional parameter
    should be used to avoid indexing errors. However, using the `fix_errors` parameter in final outputs
    might conceal errors one would want to see explicitly.

    Args:
        edge_paths: A dictionary containing a number of edge paths, i.e., full paths between two blocks, each path made of 3D blocks and pipes.
        fix_errors: When True, exclude any errors from edge_paths before processing.

    Returns:
        lat_nodes: The nodes/blocks of the resulting space-time diagram / lattice surgery (without redundant blocks)
        lat_edges: The edges/pipes of the resulting space-time diagram / lattice surgery (without redundant pipes)

    """

    # Exclude any errors in edge_paths
    if fix_errors is True:
        new_edge_paths = {
            key: path_data
            for key, path_data in edge_paths.items()
            if isinstance(path_data["src_tgt_ids"], tuple)
        }
        edge_paths = new_edge_paths

    max_id = 0
    idx_paths = {}
    for path in edge_paths.values():
        max_id = max(max_id, path["src_tgt_ids"][0], path["src_tgt_ids"][1])
    nxt_id = max_id + 1

    for path in edge_paths.values():
        idxd_path = {}
        key_1, key_2 = path["src_tgt_ids"]
        path_nodes = path["path_nodes"]

        idxd_path[key_1] = path_nodes[0]

        for i in range(1, len(path_nodes) - 1):
            mid_node = path_nodes[i]
            idxd_path[nxt_id] = mid_node
            nxt_id += 1

        if len(path_nodes) > 1:
            idxd_path[key_2] = path_nodes[-1]

        idx_paths[(key_1, key_2)] = idxd_path

    final_edges = {}
    for orig_key, path_id_value_map in idx_paths.items():
        n_ids = list(path_id_value_map.keys())
        b_ids = []
        e_ids = []

        for i in range(len(n_ids)):
            if i % 2 == 0:
                b_ids.append(n_ids[i])
            else:
                e_ids.append(n_ids[i])

        if len(b_ids) >= 2:
            for i in range(len(b_ids) - 1):
                n1 = b_ids[i]
                n2 = b_ids[i + 1]
                e_type = path_id_value_map[e_ids[i]][1]

                final_edges[(n1, n2)] = [e_type, orig_key]

    lat_nodes: dict[int, StandardBlock] = {}
    lat_edges: dict[tuple[int, int], list[str]] = {}
    for path in idx_paths.values():
        keys = list(path.keys())
        i = 0
        for key, info in path.items():
            if i % 2 == 0:
                lat_nodes[key] = info
            else:
                e_key = (keys[i - 1], keys[i + 1])
                e_type = info[1]
                lat_edges[e_key] = [e_type, final_edges[e_key][1]]
            i += 1

    return lat_nodes, lat_edges
