"""Util facilities to assist primary graph managemer BFS.

Usage:
    Call any function/class from a separate script.

"""

import random

import networkx as nx

from topologiq.utils.classes import StandardBlock, StandardCoord


#######################
# NX GRAPH OPERATIONS #
#######################
def find_first_id(nx_g: nx.Graph, deterministic: bool = False) -> int:
    """Pick a node for use as starting point by outer graph manager BFS.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        deterministic (optional): Boolean flag to change between deterministic and probabilistic mode.

    Returns:
        first_id: ID of node with highest closeness centrality or random ID from list of highest centrality.

    """

    # Terminate if graph is empty
    if not nx_g.nodes:
        raise ValueError("ERROR: nx_g.nodes() empty. Graph appears empty.")

    # If deterministic mode is ON, first ID is ID of first non-boundary node.
    if deterministic:
        all_node_ids = sorted(
            [node_id for node_id, node_info in nx_g.nodes(data=True) if node_info["type"] != "O"]
        )
        first_id = all_node_ids[0]

    # Otherwise, first ID chosen randomly from high-centrality nodes.
    else:
        max_degree = -1
        central_nodes: list[int] = []
        node_degrees = nx_g.degree

        if isinstance(node_degrees, int):
            raise ValueError("ERROR: nx_g.degree() returned int. Cannot determine first ID.")

        for node, degree in node_degrees:
            if degree > max_degree:
                max_degree = degree
                central_nodes = [node]
            elif degree == max_degree:
                central_nodes.append(node)

        # PICK A HIGHEST DEGREE NODE
        first_id: int = random.choice(central_nodes)

    return first_id


def get_node_degree(g: nx.Graph, node: int) -> int:
    """Get the degree (# of edges) of a given node.

    Args:
        g: an nx Graph.
        node: the node of interest.

    Returns:
        int: the degree for the node of interest, or 0 if graph has no edges.

    """

    # GET DEGREES FOR THE ENTIRE GRAPH
    degrees = g.degree

    # GET DEGREE FOR NODE OF INTEREST
    if not isinstance(degrees, int) and hasattr(degrees, "__getitem__"):
        return degrees[node]

    # IF DEGREES NOT A LIST, RETURN 0 (SINGLE NODE WON'T HAVE EDGES)
    return 0


######################
# BFS AUX OPERATIONS #
######################
def gen_tent_tgt_coords(
    src_c: StandardCoord,
    max_manhattan: int = 3,
    taken: list[StandardCoord] = [],
) -> list[StandardCoord]:
    """Generate a number of potential placement positions for target node.

    Args:
        src_c: The (x, y, z) coordinates for the originating block.
        max_manhattan: Max. (Manhattan) distance between origin and target blocks.
        taken: A list of coordinates already taken by previous operations.

    Returns:
        all_coords_at_distance: A list of tentative target coordinates that make good candidates for placing the target block.

    """

    # EXTRACT SOURCE COORDS
    sx, sy, sz = src_c
    tent_coords = {}

    # SINGLE MOVES
    tgts = [
        (sx + 3, sy, sz),
        (sx - 3, sy, sz),
        (sx, sy + 3, sz),
        (sx, sy - 3, sz),
        (sx, sy, sz + 3),
        (sx, sy, sz - 3),
    ]
    tent_coords[3] = [t for t in tgts if t not in taken]
    base_for_next_layer = [t for t in tgts]

    # MANHATTAN 6
    if max_manhattan > 3:
        tent_coords[6] = []
        for dx, dy, dz in [c for c in base_for_next_layer]:
            tgts = [
                (dx + 3, dy, dz),
                (dx - 3, dy, dz),
                (dx, dy + 3, dz),
                (dx, dy - 3, dz),
                (dx, dy, dz + 3),
                (dx, dy, dz - 3),
            ]
            tent_coords[6].extend([t for t in tgts if t not in taken and t != src_c])
            base_for_next_layer = [t for t in tgts]

    # MANHATTAN 9
    if max_manhattan > 6:
        tent_coords[9] = []
        for dx, dy, dz in [c for c in base_for_next_layer]:
            tgts = [
                (dx + 3, dy, dz),
                (dx - 3, dy, dz),
                (dx, dy + 3, dz),
                (dx, dy - 3, dz),
                (dx, dy, dz + 3),
                (dx, dy, dz - 3),
            ]
            tent_coords[9].extend([t for t in tgts if t not in taken and t != src_c])

    # > MANHATTAN 9
    if max_manhattan > 9:
        tgts = [
        (sx + max_manhattan, sy, sz),
        (sx - max_manhattan, sy, sz),
        (sx, sy + max_manhattan, sz),
        (sx, sy - max_manhattan, sz),
        (sx, sy, sz + max_manhattan),
        (sx, sy, sz - max_manhattan),
    ]
        tent_coords[max_manhattan] = [t for t in tgts if t not in taken]

    all_coords_at_distance = tent_coords[min(max_manhattan, 15)]
    return all_coords_at_distance


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
            if nx_g.nodes[n_id]["completed"] == []:
                pass
            elif nx_g.nodes[n_id]["completed"] >= get_node_degree(nx_g, n_id):
                nx_g.nodes[n_id]["beams"] = []
            else:
                old_beams = nx_g.nodes[n_id]["beams"]
                if old_beams:
                    for single_beam in old_beams:
                        if not any([single_beam.contains(coord) for coord in taken]):
                            new_beams += [single_beam]
                    nx_g.nodes[n_id]["beams"] = new_beams
    except (IndexError, ValueError, LookupError, KeyError):
        pass

    return nx_g


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


def get_bounding_box(
    taken: list[StandardCoord], second_pass: bool = False
) -> tuple[dict[str, dict[str, int]], int]:
    """Determine min/max coordinates for any second pass search.

    Args:
        taken: A list of all coordinates occupied by any previously-placed blocks/pipes.
        second_pass: A boolean flag to determine if search is a primary or `second_pass` search.

    Returns:
        bounding_box: A box made of min. and max. coordinates for each axis, which make a box
            declaring the space inside which the pathfinder is allowed to search for paths.
        max_span: the longest edge of the bounding box, equivalent to largest axes needed for box.

    """

    # Get the bounds of pre-existing blocks.
    bounds_x = [x for (x, _, _) in taken] if taken else [-1, 0, 1]
    bounds_y = [y for (_, y, _) in taken] if taken else [-1, 0, 1]
    bounds_z = [z for (_, _, z) in taken] if taken else [-1, 0, 1]

    # Add small leeway depending on type of search
    margin = 6 if second_pass else 12
    min_x, max_x = (min(bounds_x) - margin, max(bounds_x) + margin)
    min_y, max_y = (min(bounds_y) - margin, max(bounds_y) + margin)
    min_z, max_z = (min(bounds_z) - margin, max(bounds_z) + margin)
    bounding_box = {
        "x": {"min": min_x, "max": max_x},
        "y": {"min": min_y, "max": max_y},
        "z": {"min": min_z, "max": max_z},
    }

    # Calculate maximum span across all axes
    max_span = max(
        [
            abs((min_x + margin) - (max_x - margin)),
            abs((min_y + margin) - (max_y - margin)),
            abs((min_z + margin) - (max_z - margin)),
        ]
    )

    return bounding_box, max_span

# DEPRECATED AFTER CHANGING BEAMS TO START/END SEGMENTS
#def adjust_beam_len(node_beams: NodeBeams, taken: list[StandardCoord], second_pass: bool = False) -> NodeBeams:
        #"""Trims the beams to an optimal length.
        #Args:
            #node_beams: All the beams of a single cube.
            #taken: A list of all coordinates occupied by any previously-placed blocks/pipes.
            #second_pass: A boolean flag to determine if search is a primary or `second_pass` search.
        #Returns:
            #new_node_beams: A trimmed version of the incoming beams, with length optimal for current size of 3D space.
        #"""
        #_, max_span = get_bounding_box(taken, second_pass=second_pass)
        #max_span = 9 if max_span < 9 else 21 if max_span > 21 else max_span
        #new_node_beams = [beam[:max_span] for beam in node_beams]
        #return new_node_beams
