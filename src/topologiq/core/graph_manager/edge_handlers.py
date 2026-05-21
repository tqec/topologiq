"""Creation/generation utils to assist the primary graph managemer BFS.

Usage:
    Call any function/class from a separate script.

"""

import networkx as nx

from topologiq.core.blocks import ZXBlock
from topologiq.core.graph_manager.spatial import pipe_kind_inference
from topologiq.utils.classes import StandardCoord


################
# QUEUE PRECAL #
################
def queue_precalc(
    bgraph: nx.Graph, first_id: int, graph_traverse_strategy: str
) -> list[tuple[int, int]]:
    """Build queue using strategy defined in kwargs.

    Args:
        bgraph: The BlockGraph currently being built.
        first_id: The spider ID at which to begin traversing (needed for some algorithms)
        graph_traverse_strategy: The graph traversing strategy.

    Returns:
        edge_queue: A list of (u, v) tuples to use as queue.

    """

    # Standard edge BFS queues
    standard_edges = list(nx.bfs_edges(bgraph, first_id))
    edge_bfs = list(nx.edge_bfs(bgraph, first_id))
    cross_edges = [e for e in edge_bfs if e not in standard_edges]

    # Fill edge_queue
    edge_queue: list[tuple[int, int]] = []

    # Standard edge BFS
    if graph_traverse_strategy == "bfs":
        edge_queue = edge_bfs

    # Standard edge BFS + cross edge priority
    if graph_traverse_strategy == "bfs-cross":
        visited = []
        for u, v in standard_edges:
            if (u, v) not in edge_queue:
                [
                    edge_queue.append((uu, vv))
                    for uu, vv in cross_edges
                    if (uu, vv) not in edge_queue and uu in visited and vv in visited
                ]
                edge_queue.append((u, v))
                visited.extend([u, v])

    return edge_queue


##############
# TRANSFORMS #
##############
def add_path_to_bgraph(
    bgraph: nx.Graph,
    taken: set[StandardCoord],
    full_path: list[tuple[StandardCoord, ZXBlock]],
    curr_src_id: int,
    curr_tgt_id: int,
    is_hadamard: bool,
    cross_edge: bool,
) -> tuple[nx.Graph, set[StandardCoord]]:
    """Add a winner path to the main blockgraph.

    Args:
        bgraph: The BlockGraph currently being built.
        taken: The set of taken coordinates.
        full_path: The full path being added.
        curr_src_id: The ID of the current source cube.
        curr_tgt_id: The ID of the current target cube.
        is_hadamard: True if the current edge is a Hadamard in the input ZX graph.
        cross_edge: True if the current edge is a cross-edge (as opposed to a standard edge).

    Returns:
        bgraph: An updated BlockGraph containing the new path.
        taken: An updated set of taken coordinates.


    """

    # Health checks for source and target
    src_coords, src_zx_block = full_path.pop(0)
    tgt_coords, tgt_zx_block = full_path.pop(-1)
    if (
        bgraph.nodes[curr_src_id]["coords"] != src_coords
        or bgraph.nodes[curr_src_id]["zx_block"] != src_zx_block
    ):
        raise ValueError(
            "ERROR. Cannot add standard edge. Source must always match a pre-existing cube."
        )
    if cross_edge:
        if (
            bgraph.nodes[curr_tgt_id]["coords"] != tgt_coords
            or bgraph.nodes[curr_tgt_id]["zx_block"] != tgt_zx_block
        ):
            raise ValueError(
                "ERROR. Cannot add cross-edge. Target of a cross edge must match a pre-existing cube."
            )

    # Update source
    bgraph.nodes[curr_src_id]["completions"]["pending"] -= 1

    # Handle any intermediate cubes before target
    # Since both source and target have been popped,
    # if winner path has cubes left, there are intermediate cubes
    intermediate_cubes_present = False
    if full_path:
        # Update intermediate cube flag
        intermediate_cubes_present = True

        # In-loop trackers
        first_in_sequence = True
        prev_id = None

        # Add intermediate cubes one by one
        for coords, zx_block in full_path:
            # Calculate intermediate node ID
            n_id = max(list(bgraph.nodes())) + 1

            # Add intermediate node
            bgraph.add_node(
                n_id,
                zx_block=zx_block,
                coords=coords,
                completions={
                    "degree": 2,
                    "pending": 0,
                },
            )

            # Add intermediate cube coords to taken
            taken.add(coords)

            # Add edge as appropriate
            if first_in_sequence:
                # Make the foundational connection
                bgraph.add_edge(
                    curr_src_id,
                    n_id,
                    edge_type=bgraph.edges[(curr_src_id, curr_tgt_id)]["edge_type"],
                    start_coords=None,
                    end_coords=None,
                    kind=None,
                )

                # Remove flag for first pipe in sequence
                first_in_sequence = False

                # Infer pipe kind
                start_coords, end_coords, pipe_kind = pipe_kind_inference(
                    bgraph, curr_src_id, n_id, is_hadamard=is_hadamard
                )

                # Complete edge attributes
                bgraph.edges[(curr_src_id, n_id)]["start_coords"] = start_coords
                bgraph.edges[(curr_src_id, n_id)]["end_coords"] = end_coords
                bgraph.edges[(curr_src_id, n_id)]["kind"] = pipe_kind

            else:
                # Make the foundational connection
                bgraph.add_edge(
                    prev_id,
                    n_id,
                    edge_type="SIMPLE",
                    start_coords=None,
                    end_coords=None,
                    kind=None,
                )

                # Infer pipe kind
                start_coords, end_coords, pipe_kind = pipe_kind_inference(bgraph, prev_id, n_id)

                # Complete edge attributes
                bgraph.edges[(prev_id, n_id)]["start_coords"] = start_coords
                bgraph.edges[(prev_id, n_id)]["end_coords"] = end_coords
                bgraph.edges[(prev_id, n_id)]["kind"] = pipe_kind

            # Update previous node ID
            prev_id = n_id

    # Handle target (in either case, there is already a node for target)
    # Replace source-target edge if intermediate nodes were added
    if intermediate_cubes_present:
        # Add edge from last intermediate cube to target
        bgraph.add_edge(
            prev_id,
            curr_tgt_id,
            edge_type="SIMPLE",
            start_coords=None,
            end_coords=None,
            kind=None,
        )
        # Remove original source-target edge
        bgraph.remove_edge(curr_src_id, curr_tgt_id)

    # Update target attributes
    bgraph.nodes[curr_tgt_id]["zx_block"] = tgt_zx_block
    bgraph.nodes[curr_tgt_id]["coords"] = tgt_coords
    bgraph.nodes[curr_tgt_id]["completions"]["pending"] -= 1

    # Add target to taken
    taken.add(tgt_coords)

    # Infer pipe kind and other attributes
    if intermediate_cubes_present:
        start_coords, end_coords, pipe_kind = pipe_kind_inference(bgraph, prev_id, curr_tgt_id)
        bgraph.edges[(prev_id, curr_tgt_id)]["start_coords"] = start_coords
        bgraph.edges[(prev_id, curr_tgt_id)]["end_coords"] = end_coords
        bgraph.edges[(prev_id, curr_tgt_id)]["kind"] = pipe_kind
    else:
        start_coords, end_coords, pipe_kind = pipe_kind_inference(
            bgraph, curr_src_id, curr_tgt_id, is_hadamard=is_hadamard
        )
        bgraph.edges[(curr_src_id, curr_tgt_id)]["start_coords"] = start_coords
        bgraph.edges[(curr_src_id, curr_tgt_id)]["end_coords"] = end_coords
        bgraph.edges[(curr_src_id, curr_tgt_id)]["kind"] = pipe_kind

    return bgraph, taken
