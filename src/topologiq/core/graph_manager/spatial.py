"""Utilities to assist space management in the main graph manager class.

Usage:
    Call any function/class from a separate script.

"""

import math
import random

import networkx as nx
import numpy as np

from topologiq.core.blocks import ZXBlockRegistry
from topologiq.core.pathfinder.symbolic import rotate_pipe
from topologiq.utils.classes import StandardCoord


##################
# GRAPH REWRITES #
##################
def max_four_edges_random(bgraph: nx.Graph) -> tuple[nx.Graph, dict[int, int]]:
    """Ensure nodes in graph have max 4 neighbours (randomised breakup strategy).

    Args:
        bgraph: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.

    Return:
        bgraph: The updated version of the incoming graph.

    """

    # Determine max degree
    new_ids = {}
    max_id = max(list(bgraph.nodes()))

    # Loop over max nodes and break as appropriate
    i = 0
    while i < 100:
        # List of high degree nodes
        all_nodes_loop = bgraph.nodes()
        nodes_with_more_than_four_edges = [n for n in all_nodes_loop if bgraph.degree(n) > 4]

        # Exit loop when no nodes with more than 4 edges
        if not nodes_with_more_than_four_edges:
            break

        node_to_sanitise = random.choice(nodes_with_more_than_four_edges)
        original_node_type = bgraph.nodes[node_to_sanitise]["zx_block"].zx_type

        # Add a twin
        max_id += 1
        twin_n_id = max_id

        new_ids[twin_n_id] = node_to_sanitise

        bgraph.add_node(
            twin_n_id,
            zx_block=ZXBlockRegistry.get_create(zx_type=original_node_type),
            coords=None,
            completions={
                "degree": None,
                "pending": None,
            },
        )

        bgraph.add_edge(
            node_to_sanitise,
            twin_n_id,
            edge_type="SIMPLE",
            start_coords=None,
            end_coords=None,
            kind=None,
        )

        # Distribute edges across twins
        neighs = [n for n in list(bgraph.neighbors(node_to_sanitise)) if n != twin_n_id]
        degree_to_shuffle = bgraph.degree(node_to_sanitise) // 2
        random.shuffle(neighs)

        shuffle_c = 0
        for neigh in neighs:
            if shuffle_c >= degree_to_shuffle or bgraph.degree(node_to_sanitise) <= 4:
                break
            if bgraph.has_edge(node_to_sanitise, neigh) and not bgraph.has_edge(twin_n_id, neigh):
                edge_data = bgraph.get_edge_data(node_to_sanitise, neigh)
                edge_type = edge_data.get("edge_type", None)
                bgraph.add_edge(
                    twin_n_id,
                    neigh,
                    edge_type=edge_type,
                    start_coords=None,
                    end_coords=None,
                    kind=None,
                )
                bgraph.remove_edge(node_to_sanitise, neigh)
                shuffle_c += 1

        # Update completion attributes once final nodes and edges are in graph
        for n_id in bgraph.nodes():
            bgraph.nodes[n_id]["completions"] = {
                "degree": bgraph.degree(n_id),
                "pending": bgraph.degree(n_id),
            }

    return bgraph, new_ids


def max_four_edges_single_spider_graph(bgraph: nx.Graph) -> tuple[nx.Graph, dict[int, int]]:
    """Ensure nodes in graph have max 4 neighbours (breakup strategy optimised for single spider graphs).

    Args:
        bgraph: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.

    Return:
        bgraph: The updated version of the incoming graph.

    """

    # Split spiders into color and boundary spiders
    new_ids = {}
    in_spiders = []
    in_boundaries = []
    in_edges = {(u, v): attrs["edge_type"] for u, v, attrs in bgraph.edges(data=True)}

    [
        in_boundaries.append((spider_id, zx_block.zx_type))
        if zx_block.zx_type == "O"
        else in_spiders.append((spider_id, zx_block.zx_type))
        for spider_id, zx_block in bgraph.nodes(data="zx_block")
    ]

    # Reject if not a single spider graph
    if len(in_spiders) > 1:
        raise ValueError("Graph tagged as single spider graph is NOT single spider graph.")

    # Get central spider ZX_type
    central_spider_id, central_spider_zx_type = in_spiders[0]

    # Get rid of original edges
    bgraph.remove_edges_from([(u, v) for u, v in in_edges.keys()])

    # Add optimal number of spiders
    # Color spiders
    n_spiders_required = max(1, math.ceil(2 + (len(in_boundaries) - 6) / 2))
    color_spider_ids = [central_spider_id]
    current_spider_id = central_spider_id
    for i in range(n_spiders_required - 1):
        nxt_available_id = max(list(bgraph.nodes())) + 1

        new_ids[nxt_available_id] = current_spider_id

        bgraph.add_node(
            nxt_available_id,
            zx_block=ZXBlockRegistry.get_create(zx_type=central_spider_zx_type),
            coords=None,
            completions={
                "degree": None,
                "pending": None,
            },
        )

        bgraph.add_edge(
            current_spider_id,
            nxt_available_id,
            edge_type="SIMPLE",
            start_coords=None,
            end_coords=None,
            kind=None,
        )

        color_spider_ids.append(nxt_available_id)
        current_spider_id = nxt_available_id

    # Boundaries up to max neighbours allowed per color spider
    for boundary_id, _ in in_boundaries:
        # Pick an available color spider
        current_color_spider_id = color_spider_ids[0]

        # Add edge
        assign_zx_type = in_edges[(central_spider_id, boundary_id)]
        bgraph.add_edge(
            current_color_spider_id,
            boundary_id,
            edge_type=assign_zx_type,
            start_coords=None,
            end_coords=None,
            kind=None,
        )

        # Remove color ID from available color spiders
        if bgraph.degree(current_color_spider_id) == 4:
            color_spider_ids.remove(current_color_spider_id)

    # Update completion attributes once final nodes and edges are in graph
    for n_id in bgraph.nodes():
        bgraph.nodes[n_id]["completions"] = {
            "degree": bgraph.degree(n_id),
            "pending": bgraph.degree(n_id),
        }

    return bgraph, new_ids


########################
# TENTATIVE COORDS GEN #
########################
def gen_tent_tgt_coords(
    src_c: StandardCoord,
    max_manhattan: int = 1,
    taken: set[StandardCoord] = [],
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
    base_for_next_layer = []
    tent_coords = {}

    # SINGLE MOVES
    tgts = [
        (sx + 1, sy, sz),
        (sx - 1, sy, sz),
        (sx, sy + 1, sz),
        (sx, sy - 1, sz),
        (sx, sy, sz + 1),
        (sx, sy, sz - 1),
    ]
    tent_coords[1] = [t for t in tgts if t not in taken]
    base_for_next_layer = [t for t in tgts]

    # MANHATTAN 6
    if max_manhattan > 1:
        tent_coords[2] = []
        for dx, dy, dz in [c for c in base_for_next_layer]:
            tgts = [
                (dx + 1, dy, dz),
                (dx - 1, dy, dz),
                (dx, dy + 1, dz),
                (dx, dy - 1, dz),
                (dx, dy, dz + 1),
                (dx, dy, dz - 1),
            ]
            tent_coords[2].extend([t for t in tgts if t not in taken and t != src_c])
            base_for_next_layer.extend([t for t in tgts])

    # MANHATTAN 9
    if max_manhattan > 2:
        tent_coords[3] = []
        for dx, dy, dz in [c for c in base_for_next_layer]:
            tgts = [
                (dx + 1, dy, dz),
                (dx - 1, dy, dz),
                (dx, dy + 1, dz),
                (dx, dy - 1, dz),
                (dx, dy, dz + 1),
                (dx, dy, dz - 1),
            ]
            tent_coords[3].extend([t for t in tgts if t not in taken and t != src_c])
            base_for_next_layer.extend([t for t in tgts])

    # > MANHATTAN 9
    if max_manhattan > 3:
        tent_coords[max_manhattan] = []
        num_loops = int(max_manhattan - 3)

        for _ in [i + 1 for i in range(num_loops)]:
            for dx, dy, dz in [c for c in base_for_next_layer]:
                tgts = [
                    (dx + 1, dy, dz),
                    (dx - 1, dy, dz),
                    (dx, dy + 1, dz),
                    (dx, dy - 1, dz),
                    (dx, dy, dz + 1),
                    (dx, dy, dz - 1),
                ]
                tent_coords[max_manhattan].extend(
                    [t for t in tgts if t not in taken and t != src_c]
                )
                base_for_next_layer.extend([t for t in tgts])

    all_coords_at_distance = tent_coords[min(max_manhattan, 3)]
    return all_coords_at_distance


##############
# INFERENCES #
##############
def pipe_kind_inference(
    bgraph: nx.Graph,
    u: int,
    v: int,
    is_hadamard: bool = False,
) -> tuple[StandardCoord, StandardCoord, str]:
    """Infer the pipe kind between two cubes.

    Args:
        bgraph: The BlockGraph currently being built.
        u: And ID to override the ID of the source cube.
        v: And ID to override the ID of the target cube.
        is_hadamard: Whether the pipe is a Hadamard pipe.

    """

    # Health checks
    if not bgraph.has_node(u) or not bgraph.has_node(v):
        raise ValueError(
            "ERROR. Cannot infer pipe kind. Source and target cubes must exist in blockgraph."
        )
    if not bgraph.has_edge(u, v) and not bgraph.has_edge(v, u):
        raise ValueError(
            "ERROR. Cannot infer pipe kind. Source and target must have a connecting edge."
        )
    if not bgraph.nodes[u]["coords"] or not bgraph.nodes[v]["coords"]:
        raise ValueError(
            "ERROR. Cannot infer pipe kind. Source and target must have a pre-defined position."
        )

    # Determine base kind
    base_id, end_id = (u, v) if bgraph.nodes[u]["coords"] < bgraph.nodes[v]["coords"] else (v, u)
    start_coords = bgraph.nodes[base_id]["coords"]
    end_coords = bgraph.nodes[end_id]["coords"]
    base_kind = bgraph.nodes[base_id]["zx_block"].kind

    # Determine pipe direction
    directional_axis = np.array(np.nonzero(np.array(start_coords) - np.array(end_coords))).item()

    # Exchange base_kind for kind of immediately adjacent block if base cube is a port
    if base_kind in ["OOO", "YYO", "YYI", "YYM", "TTO", "TTI", "TTM"]:
        if is_hadamard:
            move = np.array(end_coords) - np.array(start_coords)
            base_kind = rotate_pipe(bgraph.nodes[end_id]["zx_block"].kind, move)
        else:
            base_kind = bgraph.nodes[end_id]["zx_block"].kind

    # Write foundational pipe kind
    pipe_kind = base_kind[:directional_axis] + "O" + base_kind[directional_axis + 1 :]

    # Append Hadamard symbol if applicable
    if is_hadamard:
        pipe_kind += "H"

    return start_coords, end_coords, pipe_kind
