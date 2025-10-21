import random
import networkx as nx

from typing import Tuple, List

from topologiq.utils.classes import StandardCoord, NodeBeams, StandardBlock


#######################
# NX GRAPH OPERATIONS #
#######################
def find_first_id(nx_g: nx.Graph) -> int:
    """Pick a node for use as starting point for outer graph manager BFS.

    Args:
        - nx_g: an nx_graph.

    Returns:
        - first_id: ID of node with highest closeness centrality or random ID from list of highest centrality.

    """

    # TERMINATE IF THERE ARE NO NODES
    if not nx_g.nodes:
        raise ValueError("ERROR: nx_g.nodes() empty. Graph appears empty.")

    # LOOP OVER NODES FINDING NODES WITH HIGHEST DEGREE
    max_degree = -1
    central_nodes: List[int] = []

    node_degrees = nx_g.degree
    if isinstance(node_degrees, int):
        raise ValueError("ERROR: nx_g.degree() returned int. Cannot determine first ID.")
    else:
        for node, degree in node_degrees:
            if degree > max_degree:
                max_degree = degree
                central_nodes = [node]
            elif degree == max_degree:
                central_nodes.append(node)

    # PICK A HIGHEST DEGREE NODE
    first_id: int = random.choice(central_nodes)

    # RETURN FIRST NODE
    return first_id


def get_node_degree(g: nx.Graph, node: int) -> int:
    """Gets the degree (# of edges) of a given node.

    Args:
        - g: an nx Graph.
        - node: the node of interest.

    Returns:
        - int: the degree for the node of interest, or 0 if graph has no edges.

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
    taken: List[StandardCoord] = [],
) -> List[StandardCoord]:
    """Generates a number of potential placement positions for target node.

    Args:
        - src_c: (x, y, z) coordinates for the originating block.
        - max_manhattan: Max. (Manhattan) distance between origin and target blocks.
        - taken: a list of coordinates already taken by previous operations.

    Returns:
        - tent_coords: a list of tentative target coordinates that make good candidates for placing the target block.

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

    # MANHATTAN 6
    if max_manhattan > 3:
        tent_coords[6] = []
        for dx, dy, dz in [c for c in tent_coords[3]]:
            tgts = [
                (dx + 3, dy, dz),
                (dx - 3, dy, dz),
                (dx, dy + 3, dz),
                (dx, dy - 3, dz),
                (dx, dy, dz + 3),
                (dx, dy, dz - 3),
            ]

            tent_coords[6].extend(
                [
                    t
                    for t in tgts
                    if all(
                        [
                            abs(t[0]) - abs(sx) != 6,
                            abs(t[1]) - abs(sy) != 6,
                            abs(t[2]) - abs(sz) != 6,
                        ]
                    )
                    and sum(
                        [abs(t[0]) - abs(sx), abs(t[1]) - abs(sy), abs(t[2]) - abs(sz)]
                    )
                    == 6
                    and t not in taken
                ]
            )

    # MANHATTAN 9
    if max_manhattan > 6:
        tent_coords[9] = []
        for dx, dy, dz in [c for c in tent_coords[6]]:
            tgts = [
                (dx + 3, dy, dz),
                (dx - 3, dy, dz),
                (dx, dy + 3, dz),
                (dx, dy - 3, dz),
                (dx, dy, dz + 3),
                (dx, dy, dz - 3),
            ]

            tent_coords[9].extend(
                [
                    t
                    for t in tgts
                    if all(
                        [
                            abs(t[0]) - abs(sx) != 9,
                            abs(t[1]) - abs(sy) != 9,
                            abs(t[2]) - abs(sz) != 9,
                        ]
                    )
                    and sum(
                        [abs(t[0]) - abs(sx), abs(t[1]) - abs(sy), abs(t[2]) - abs(sz)]
                    )
                    == 9
                    and t not in taken
                ]
            )

    # RETURN ALL COORDS WITHIN DISTANCE
    return tent_coords[min(max_manhattan, 9)]


def prune_beams(
    nx_g: nx.Graph, all_beams: List[NodeBeams], taken: List[StandardCoord]
) -> Tuple[nx.Graph, List[NodeBeams]]:
    """Removes beams that have already been broken, as these are no longer indicative of anything.
    Args:
        - all_beams: list of coordinates taken by the beams of all blocks in original ZX-graph
        - taken: list of coordinates taken by any blocks/pipes placed as a result of previous operations.
    Returns:
        - new_beams: list of beams without any beams where any of the coordinates in the path of the beam overlap with taken coords.
    """

    try:
        new_beams = []
        for beams in all_beams:
            iter_beams = [
                beam for beam in beams if all([coord not in taken for coord in beam])
            ]
            if iter_beams:
                new_beams.append(iter_beams)
    except (IndexError, ValueError, LookupError, KeyError):
        new_beams = all_beams

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
                    for beam in old_beams:
                        if all([(c not in taken) for c in beam]):
                            new_beams.append(beam)

                    nx_g.nodes[n_id]["beams"] = new_beams
    except (IndexError, ValueError, LookupError, KeyError):
        nx_g = nx_g

    return nx_g, new_beams


def reindex_pth_dict(
    edge_pths: dict,
) -> Tuple[dict[int, StandardBlock], dict[Tuple[int, int], List[str]]]:
    """Distils an edge_pth object into a final list of nodes/blocks and edges/pipes for the space-time diagram.

    Args:
        - edge_pths: a dictionary containing a number of edge paths, i.e., full paths between two blocks, each path made of 3D blocks and pipes.

    Returns:
        - lat_nodes: the nodes/blocks of the resulting space-time diagram / lattice surgery (without redundant blocks)
        - lat_edges: the edges/pipes of the resulting space-time diagram / lattice surgery (without redundant pipes)

    """

    max_id = 0
    idx_pths = {}
    for pth in edge_pths.values():
        max_id = max(max_id, pth["src_tgt_ids"][0], pth["src_tgt_ids"][1])
    nxt_id = max_id + 1

    for pth in edge_pths.values():

        idxd_pth = {}
        key_1, key_2 = pth["src_tgt_ids"]
        pth_nodes = pth["pth_nodes"]

        idxd_pth[key_1] = pth_nodes[0]

        for i in range(1, len(pth_nodes) - 1):
            mid_node = pth_nodes[i]
            idxd_pth[nxt_id] = mid_node
            nxt_id += 1

        if len(pth_nodes) > 1:
            idxd_pth[key_2] = pth_nodes[-1]

        idx_pths[(key_1, key_2)] = idxd_pth

    final_edges = {}
    for orig_key, pth_id_value_map in idx_pths.items():
        n_ids = list(pth_id_value_map.keys())
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
                e_type = pth_id_value_map[e_ids[i]][1]

                final_edges[(n1, n2)] = [e_type, orig_key]

    lat_nodes: dict[int, StandardBlock] = {}
    lat_edges: dict[Tuple[int, int], List[str]] = {}
    for pth in idx_pths.values():
        keys = list(pth.keys())
        i = 0
        for key, info in pth.items():
            if i % 2 == 0:
                lat_nodes[key] = info
            else:
                e_key = (keys[i - 1], keys[i + 1])
                e_type = info[1]
                lat_edges[e_key] = [e_type, final_edges[e_key][1]]
            i += 1

    return lat_nodes, lat_edges


######################
# DEBUG OPERATIONS #
######################
# CONSIDER MOVE LOGGING AND ANIMATION HERE.