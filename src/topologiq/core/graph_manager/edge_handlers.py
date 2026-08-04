"""Creation/generation utils to assist the primary graph managemer BFS.

Usage:
    Call any function/class from a separate script.

"""

import networkx as nx
import numpy as np

from topologiq.core.blocks import ZXBlock
from topologiq.core.graph_manager.spatial import pipe_kind_inference
from topologiq.utils.classes import StandardCoord


################
# QUEUE PRECAL #
################
def queue_precalc(
    bgraph: nx.Graph,
    first_id: int,
    rows: dict[int, int],
    qubits: dict[int, int],
    inputs: list[int],
    outputs: list[int],
    graph_traverse_strategy: str,
    other_first_ids: list[int] = [],
    s_gates: dict[int, list[tuple[int, int] | None]] | None = None,
    t_gates: dict[int, list[tuple[int, int] | None]] | None = None,
) -> list[tuple[int, int]]:
    """Build queue using strategy defined in kwargs.

    Args:
        bgraph: The BlockGraph currently being built.
        first_id: The spider ID at which to begin traversing (needed for some algorithms).
        rows: Spiders organised by row.
        qubits: Spiders organised by qubit.
        inputs: The inputs of the base graph.
        outputs: The outputs of the base graph.
        graph_traverse_strategy: The graph traversing strategy.
        other_first_ids (optional): Other IDS to use as basis for queue (in BFS-layers strategy).
        s_gates (optional): A dictionary containing all IDs and edges in S-gate patterns.
        t_gates (optional): A dictionary containing all IDs and edges in T-gate patterns.

    Returns:
        edge_queue: A list of (u, v) tuples to use as queue.

    """
    # Init empty edge queue
    edge_queue: list[tuple[int, int]] = []

    # Standard edge BFS
    if graph_traverse_strategy == "bfs":
        edge_queue = list(nx.edge_bfs(bgraph, first_id))

    # Standard edge BFS + cross edge priority
    if graph_traverse_strategy == "bfs-cross":
        edge_queue = _queue_bfs_cross(bgraph, first_id)

    # BFS with priority cross-edges strategy and boundaries last
    if graph_traverse_strategy == "bfs-cross-boundaries-last":
        edge_queue = _queue_bfs_cross_boundaries_last(bgraph, first_id)

    if graph_traverse_strategy == "bfs-cycles":
        edge_queue = _queue_bfs_cycles(bgraph, first_id)

    # Layered BFS (by rows)
    if graph_traverse_strategy == "bfs-rows":
        edge_queue = _queue_bfs_rows(
            bgraph, first_id, rows, inputs, other_first_ids=other_first_ids, t_gates=t_gates
        )

    # Layered BFS (by T-gate sections)
    if graph_traverse_strategy == "bfs-layers":
        edge_queue = _queue_bfs_layers(
            bgraph,
            first_id,
            qubits,
            rows,
            inputs,
            s_gates=s_gates,
            t_gates=t_gates,
        )

    # Layered BFS (by T-gates)
    if graph_traverse_strategy == "tfs":
        edge_queue = _queue_tfs(bgraph, first_id, inputs, s_gates=s_gates, t_gates=t_gates)

    # An almost-no-longer-BFS that lays out the central qubit before traversing CNOTs
    if graph_traverse_strategy == "bfs-cnots":
        edge_queue = _queue_bfs_cnots(
            bgraph, rows, qubits, inputs, other_first_ids=other_first_ids, t_gates=t_gates
        )

    # An almost-no-longer-BFS that lays out the central qubit before traversing CNOT cycles
    if graph_traverse_strategy == "bfs-cnot-cycles":
        edge_queue = _queue_bfs_cnot_cycles(
            bgraph, rows, qubits, inputs, other_first_ids=other_first_ids, t_gates=t_gates
        )

    # A T-gate First Search (TFS) approach that visits T-gates before expanding using CNOTs
    if graph_traverse_strategy == "tfs-cnots":
        edge_queue = _queue_tfs_cnots(
            bgraph,
            rows,
            qubits,
            inputs,
            other_first_ids=other_first_ids,
            s_gates=s_gates,
            t_gates=t_gates,
        )

    if not edge_queue:
        raise ValueError("No valid edge queue strategy was found or queue generation failed.")

    return edge_queue


####################
# QUEUE RATIONALES #
####################
def _queue_bfs_cross(
    bgraph: nx.Graph,
    first_id: int,
    edge_queue: list[tuple[int, int]] | None = None,
    standard_edges: list[tuple[int, int]] | None = None,
    cross_edges: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Build queue using a BFS with priority cross-edges strategy.

    Args:
        bgraph: The BlockGraph currently being built.
        first_id: The spider ID at which to begin traversing (needed for some algorithms).
        edge_queue (optional): A pre-existing edge-queue with some edges already completed.
        standard_edges (optional): Pre-built NX BFS (avoids duplication if running in a different hander).
        cross_edges (optional): Pre-built list of cross edges (avoids duplication if running in a different hander).

    Returns:
        edge_queue: A list of (u, v) tuples to use as queue.

    """

    # Instantiate empty edge queue if not given
    edge_queue = edge_queue if edge_queue else []

    # Get native NX queues to use as basis for construction
    standard_edges = standard_edges if standard_edges else list(nx.bfs_edges(bgraph, first_id))

    if not cross_edges:
        edge_bfs = list(nx.edge_bfs(bgraph, first_id))
        cross_edges = [e for e in edge_bfs if e not in standard_edges]
    else:
        cross_edges = []

    visited = []
    for u, v in standard_edges:
        if (u, v) not in edge_queue and (v, u) not in edge_queue:
            [
                edge_queue.append((uu, vv))
                for uu, vv in cross_edges
                if (
                    (uu, vv) not in edge_queue
                    and (vv, uu) not in edge_queue
                    and uu in visited
                    and vv in visited
                )
            ]
            edge_queue.append((u, v))
            visited.extend([u, v])

    return edge_queue


def _queue_bfs_cross_boundaries_last(bgraph: nx.Graph, first_id: int) -> list[tuple[int, int]]:
    """Build queue using a BFS with priority cross-edges strategy and boundaries last.

    Args:
        bgraph: The BlockGraph currently being built.
        first_id: The spider ID at which to begin traversing (needed for some algorithms).

    Returns:
        edge_queue: A list of (u, v) tuples to use as queue.

    """

    # Get native NX queues
    standard_edges = list(nx.bfs_edges(bgraph, first_id))
    edge_bfs = list(nx.edge_bfs(bgraph, first_id))
    cross_edges = [e for e in edge_bfs if e not in standard_edges]

    # Shuffle edges into a ckear standard -> cross -> boundary edges
    boundary_edges = [
        (u, v)
        for u, v in standard_edges
        if "O" in [bgraph.nodes[u]["zx_block"].zx_type, bgraph.nodes[v]["zx_block"].zx_type]
    ]
    boundary_spiders = [i for i in bgraph.nodes() if bgraph.nodes[i]["zx_block"].zx_type == "O"]
    alt_edge_queue = [
        (u, v)
        for u, v in standard_edges
        if (u not in boundary_spiders and v not in boundary_spiders)
    ]
    return [*alt_edge_queue, *cross_edges, *boundary_edges]


def _queue_bfs_rows(
    bgraph: nx.Graph,
    first_id: int,
    rows: dict[int, int],
    inputs: list[int],
    other_first_ids: list[int] = [],
    t_gates: dict[int, list[tuple[int, int] | None]] = {},
) -> list[tuple[int, int]]:
    """Build queue using a row by row graph traverse strategy.

    Args:
        bgraph: The BlockGraph currently being built.
        first_id: The spider ID at which to begin traversing (needed for some algorithms)
        rows: Spiders organised by row.
        inputs: The inputs of the base graph.
        other_first_ids (optional): Other IDS to use as basis for queue (in BFS-layers strategy)
        t_gates (optional): A dictionary containing the edges of each T-gate pattern in base graph.

    Returns:
        edge_queue: A list of (u, v) tuples to use as queue.

    """

    # Faux first spiders
    faux_input_input_edges = []
    prev_id = first_id

    for other_id in other_first_ids[1:]:
        faux_input_input_edges.append((prev_id, other_id))
        bgraph.add_edge(
            prev_id,
            other_id,
            edge_type="SIMPLE",
            start_coords=None,
            end_coords=None,
            kind=None,
        )
        prev_id = other_id
    edge_queue = faux_input_input_edges

    # Calculate all layers starting at inputs
    spiders_by_row: dict[int, set[int]] = {}
    ids_in_t_patterns: set[int] = set()
    for pat in t_gates.values():
        ids_in_t_patterns.update(tuple(x for edge in pat for x in edge))

    for spider_id, row_number in rows.items():
        if spider_id in inputs or spider_id in ids_in_t_patterns:
            continue
        if row_number not in spiders_by_row:
            spiders_by_row[row_number] = set([spider_id])
        else:
            spiders_by_row[row_number].add(spider_id)

    # Visit the full breadth of row
    visited = set(inputs)

    for row_number, spider_ids_in_row in spiders_by_row.items():
        # Reset list of same row edges (CNOT/CZ/T-gadgets)
        same_row_edges = []

        # Visit each neighbour of each spider in row
        for spider_id in spider_ids_in_row:
            # Safekeeper for edges to S- and T-gates patterns
            # Note. Not all S- and T-gates will be handled here,
            #   as this depends on the exact position given to the
            #   base of the S- or T-gate pattern.
            hold_edges = []

            # Loop over all neighbours of current spider
            neighs = bgraph.neighbors(spider_id)
            for neigh_id in neighs:
                # If edge to current neighbour not already in queue
                if (neigh_id, spider_id) not in edge_queue and (
                    spider_id,
                    neigh_id,
                ) not in edge_queue:
                    # Edges to spiders in previous rows
                    # Covers all spiders in same qubit, most S-gates,
                    # and potentially some T-gates.
                    if rows[neigh_id] < rows[spider_id]:
                        # Most edges will be to previous spider in same qubits
                        # in which case the neighbour has been visited
                        if neigh_id in visited:
                            edge_queue.append((neigh_id, spider_id))
                            visited.add(spider_id)
                        # Edges to S- and T-gates must wait until spider is visited at least once
                        else:
                            hold_edges.append((spider_id, neigh_id))
                    # (Rare) S-gates placed on subsequent rows
                    elif bgraph.nodes[neigh_id]["zx_block"].zx_type in ["Y", "T"]:
                        hold_edges.append((spider_id, neigh_id))

                    # Handle CNOT/CZ
                    if rows[neigh_id] == rows[spider_id]:
                        # Check that cross edge is not being duplicated
                        if tuple(sorted([spider_id, neigh_id])) not in same_row_edges:
                            # Add the actual edge to list of same row edges
                            same_row_edges.append(tuple(sorted([spider_id, neigh_id])))
                            # Add any associated T-pattern to list of same row edges
                            if neigh_id in t_gates:
                                same_row_edges.extend(t_gates[neigh_id])
                                (visited.update([u, v]) for u, v in t_gates[neigh_id])
            # Current spider has been visited at this point if it has neighbours,
            # so it is possible to add any S- or T-gates encountered
            if hold_edges:
                edge_queue.extend(hold_edges)
                [visited.update([u, v]) for u, v in hold_edges]

        # Add list of same row edges to edge queue
        edge_queue.extend(same_row_edges)
        # Add the full T-pattern if circuit had T-gates
        if t_gates:
            alt_edge_queue: list[tuple[int, int]] = []
            for u, v in edge_queue:
                alt_edge_queue.append((u, v))
                if v in t_gates and t_gates[v][0] not in edge_queue:
                    alt_edge_queue.extend(t_gates[v])
                    [visited.update([u, v]) for u, v in t_gates[v]]
            edge_queue = alt_edge_queue

    return edge_queue


def _queue_bfs_layers(
    bgraph: nx.Graph,
    first_id: int,
    qubits: dict[int, int],
    rows: dict[int, int],
    inputs: list[int],
    s_gates: dict[int, list[tuple[int, int] | None]] | None = None,
    t_gates: dict[int, list[tuple[int, int] | None]] | None = None,
) -> list[tuple[int, int]]:
    """Build queue using a row by row graph traverse strategy.

    Args:
        bgraph: The BlockGraph currently being built.
        first_id: The spider ID at which to begin traversing (needed for some algorithms)
        qubits: Spiders with their respective qubit.
        rows: Spiders with their respective row.
        inputs: The inputs of the base graph.
        other_first_ids (optional): Other IDS to use as basis for queue (in BFS-layers strategy).
        s_gates (optional): A dictionary containing the edges of each S-gate pattern in base graph.
        t_gates (optional): A dictionary containing the edges of each T-gate pattern in base graph.

    Returns:
        edge_queue: A list of (u, v) tuples to use as queue.

    """

    # Init empty edge queue and visited with first id
    edge_queue = []
    visited = set([first_id])
    visited_edges = set()

    inverted_qubits = {}
    for k, v in qubits.items():
        if v not in inverted_qubits:
            inverted_qubits[v] = [k]
        else:
            inverted_qubits[v].append(k)
    qubits_by_len = sorted(
        [(k, sorted(v)) for k, v in inverted_qubits.items()], key=lambda x: len(x), reverse=True
    )

    # Start by visiting all inputs
    if len(inputs) > 1:
        for in_id in inputs:
            if in_id not in visited:
                # Get path to current input
                path_to_in_id = nx.shortest_path(bgraph, source=first_id, target=in_id)
                u_v_to_t = [
                    (path_to_in_id[i - 1], s_id) for i, s_id in enumerate(path_to_in_id) if i != 0
                ]

                # Add any edges in path not already in edge_queue
                for u, v in u_v_to_t:
                    if ((u, v) not in visited_edges) and ((v, u) not in visited_edges):
                        edge_queue.append((u, v))
                        visited_edges.add((u, v))
                        visited.add(v)

        # Traverse longest qubit
        u_v_to_t = [
            (qubits_by_len[0][1][i - 1], s_id) for i, s_id in enumerate(qubits_by_len[0][1]) if i != 0
        ]
        for u, v in u_v_to_t:
            if ((u, v) not in visited_edges) and ((v, u) not in visited_edges):
                edge_queue.append((u, v))
                visited_edges.add((u, v))
                visited.add(v)

        # Priority T-gates
        if t_gates:
            for t_pattern in t_gates.values():
                # If first edge in a T-pattern is in visited, all T-pattern is in visited
                if t_pattern[0] not in visited_edges:
                    # Get path to first spider in T-pattern
                    _, path_to_t = nx.multi_source_dijkstra(bgraph, visited, target=t_pattern[0][0])
                    u_v_to_t = [(path_to_t[i - 1], s_id) for i, s_id in enumerate(path_to_t) if i != 0]

                    # Add any edges in path not already in edge_queue
                    for u, v in u_v_to_t:
                        if ((u, v) not in visited_edges) and ((v, u) not in visited_edges):
                            edge_queue.append((u, v))
                            visited_edges.add((u, v))
                            visited.add(v)

                    # Add the T-pattern
                    edge_queue.extend(t_pattern)
                    visited_edges.update(t_pattern)

        # Priority S-gates
        if s_gates:
            for s_pattern in s_gates.values():
                # If first edge in a S-pattern is in visited, all S-pattern is in visited
                if s_pattern[0] not in visited_edges:
                    # Get path to first spider in T-pattern
                    _, path_to_t = nx.multi_source_dijkstra(bgraph, visited, target=s_pattern[0][0])
                    u_v_to_t = [(path_to_t[i - 1], s_id) for i, s_id in enumerate(path_to_t) if i != 0]

                    # Add any edges in path not already in edge_queue
                    for u, v in u_v_to_t:
                        if ((u, v) not in visited_edges) and ((v, u) not in visited_edges):
                            edge_queue.append((u, v))
                            visited_edges.add((u, v))
                            visited.add(v)

                    # Add the S-pattern
                    edge_queue.extend(s_pattern)
                    visited_edges.update(s_pattern)

        # CNOTs
        cross_edges = []
        for spider_id in qubits_by_len[0][1]:
            neighs = bgraph.neighbors(spider_id)
            for neigh_id in neighs:
                if (
                rows[neigh_id] == rows[spider_id]
                and (spider_id, neigh_id) not in edge_queue
                and (neigh_id, spider_id) not in edge_queue
                ):
                    cross_edges.append((spider_id, neigh_id))
                    visited.add(neigh_id)
                    visited_edges.add((spider_id, neigh_id))
        edge_queue.extend(list(cross_edges))

    # Complete remainder using a NX BFS strategy with cross edge priority
    edge_queue = _queue_bfs_cross(bgraph, first_id, edge_queue=edge_queue)

    return edge_queue


def _queue_bfs_cycles(bgraph: nx.Graph, first_id: int) -> list[tuple[int, int]]:
    """Build queue using a BFS with priority cross-edges strategy.

    Args:
        bgraph: The BlockGraph currently being built.
        first_id: The spider ID at which to begin traversing (needed for some algorithms).

    Returns:
        edge_queue: A list of (u, v) tuples to use as queue.

    """

    def _distance_to_cycle(cycle, path_lens):
        d = [path_lens[n] for n in cycle if n in path_lens]
        return min(d) if d else np.info

    # Get NX cycles
    nx_cycles_init = nx.cycle_basis(bgraph, first_id)

    # Fallback to standard BFS if circuit has no cycles
    if not nx_cycles_init:
        print("Circuit has no cycles, returning a standard BFS")
        return list(nx.edge_bfs(bgraph, first_id))

    # Initialise key objects
    edge_queue: list[tuple[int, int]] = []
    visited: set[int] = set([first_id])
    visited_edges: set[tuple[int, int]] = set()
    cycle_leader_id = None

    # Get native NX queue as basis for construction
    edge_bfs = list(nx.edge_bfs(bgraph, first_id))

    # Check that first ID is in first cycle
    if first_id not in nx_cycles_init[0]:
        path_lens = nx.single_source_shortest_path_length(bgraph, first_id)
        nx_cycles = sorted(nx_cycles_init, key=lambda x: (_distance_to_cycle(x, path_lens), len(x)))
    else:
        nx_cycles = nx_cycles_init

    # Iteratively add cycles to edge_queue
    for cycle in nx_cycles:
        # Start by defining most NB! spider in cycle
        if cycle_leader_id:
            new_cycle_leader_id = None
            max_degree_in_round = 0
            for cycle_member_id in cycle:
                if cycle_member_id in visited:
                    cycle_member_degree = bgraph.degree[cycle_member_id]
                    if not new_cycle_leader_id or cycle_member_degree > max_degree_in_round:
                        new_cycle_leader_id = cycle_member_id
            cycle_leader_id = new_cycle_leader_id
        else:
            cycle_leader_id = first_id

        # If no cycle leader is found cycle is disconnected and needs to be connected
        # This is expensive but one wouldn't want to use this queue strategy for
        # circuits with many disconnected cycles.
        if not cycle_leader_id:
            shortest_path = None
            for cycle_member_id in cycle:
                shortest_paths_to_cycle_member = nx.single_target_shortest_path(
                    bgraph, cycle_member_id
                )

                iter_min_len = np.inf
                for k, v in shortest_paths_to_cycle_member.items():
                    if k not in visited:
                        continue

                    path_len = len(v)
                    if path_len == 1:
                        continue

                    if path_len < iter_min_len:
                        iter_min_len = path_len
                        iter_shortest_path = v

                if not shortest_path and not iter_shortest_path:
                    continue

                if iter_shortest_path:
                    if not shortest_path or (len(iter_shortest_path) < len(shortest_path)):
                        shortest_path = iter_shortest_path

            if shortest_path:
                cycle_leader_id = shortest_path[-1]

            first = True
            for i, spider_id in enumerate(shortest_path):
                if first:
                    first = False
                    continue
                if ((shortest_path[i - 1], spider_id) not in visited_edges) and (
                    (spider_id, shortest_path[i - 1]) not in visited_edges
                ):
                    edge_queue.append((shortest_path[i - 1], spider_id))
                    visited_edges.add((shortest_path[i - 1], spider_id))
                    visited.add(spider_id)

        # BFS through current cycle
        cycle_subgraph = bgraph.subgraph(cycle)
        cycle_bfs = nx.edge_bfs(cycle_subgraph, cycle_leader_id)

        # Add cycle BFS to edge queue
        for u, v in cycle_bfs:
            if ((u, v) not in visited_edges) and ((v, u) not in visited_edges):
                edge_queue.append((u, v))
                visited_edges.add((u, v))
                visited.add(v)

    # Loop over all edges of graph picking up non-completed spiders
    hold_these_edges = []
    for u, v in edge_bfs:
        if (u, v) not in visited_edges and (v, u) not in visited_edges:
            if u in visited:
                edge_queue.append((u, v))
                visited_edges.add((u, v))
                if v not in visited:
                    visited.add(v)
            elif v in visited:
                edge_queue.append((v, u))
                visited_edges.add((v, u))
                if u not in visited:
                    visited.add(u)
            else:
                hold_these_edges.append((v, u))

    if hold_these_edges:
        hold_these_edges_copy = hold_these_edges.copy()
        while hold_these_edges_copy:
            for u, v in hold_these_edges:
                if u in visited:
                    edge_queue.append((u, v))
                    visited_edges.add((u, v))
                    visited.add(v)
                    hold_these_edges_copy.remove((u, v))
                elif v in visited:
                    edge_queue.append((v, u))
                    visited_edges.add((v, u))
                    visited.add(u)
                    hold_these_edges_copy.remove((u, v))

    return edge_queue


def _queue_bfs_cnots(
    bgraph: nx.Graph,
    rows: dict[int, int],
    qubits: dict[int, int],
    inputs: list[int],
    other_first_ids: list[int] = [],
    t_gates: dict[int, list[tuple[int, int] | None]] = {},
) -> list[tuple[int, int]]:
    """Build queue using strategy defined in kwargs.

    Args:
        bgraph: The BlockGraph currently being built.
        first_id: The spider ID at which to begin traversing (needed for some algorithms)
        rows: Spiders organised by row.
        qubits: Spiders organised by qubit.
        inputs: The inputs of the base graph.
        graph_traverse_strategy: The graph traversing strategy.
        other_first_ids (optional): Other IDS to use as basis for queue (in BFS-layers strategy)
        t_gates (optional): A dictionary containing the edges of each T-gate pattern in base graph.

    Returns:
        edge_queue: A list of (u, v) tuples to use as queue.

    """

    # Strategy only works if first_spider is an input
    if not inputs:
        raise ValueError(
            "CNOT cycles graph traversing requires the base graph to have at least one input"
        )

    # Initialise edge queue
    edge_queue: list[tuple[int, int]] = []

    # Extract T-gate patterns if applicable
    t_gate_pats_ids = set()
    for edges_in_t_pat in t_gates.values():
        if edges_in_t_pat:
            t_gate_pats_ids.update([i for e in edges_in_t_pat for i in e])

    # Organise spiders by qubit
    spiders_by_qubit_row: dict[int, list[tuple[int, int]]] = {}
    for spider_id, qubit_number in qubits.items():
        if qubit_number not in spiders_by_qubit_row:
            spiders_by_qubit_row[qubit_number] = [spider_id]
        else:
            spiders_by_qubit_row[qubit_number].append(spider_id)

    # Sort qubit layers
    spiders_by_qubit_row_sorted = {
        k: sorted(v, key=lambda x: rows[x]) for k, v in spiders_by_qubit_row.items()
    }

    # Traverse the full central qubit
    visited = set([other_first_ids[0]])
    visited_edges = set()
    cross_edges = []
    central_qubit = qubits[other_first_ids[0]]
    for spider_id in spiders_by_qubit_row_sorted[central_qubit]:
        neighs = bgraph.neighbors(spider_id)
        for neigh_id in neighs:
            if rows[neigh_id] < rows[spider_id]:
                if neigh_id in visited and (neigh_id, spider_id) not in edge_queue:
                    edge_queue.append((neigh_id, spider_id))
                    visited.add(spider_id)
                    visited_edges.add((neigh_id, spider_id))
            if (
                rows[neigh_id] == rows[spider_id]
                and (spider_id, neigh_id) not in edge_queue
                and (neigh_id, spider_id) not in edge_queue
            ):
                cross_edges.append((spider_id, neigh_id))
                visited.add(neigh_id)
                visited_edges.add((spider_id, neigh_id))
    edge_queue.extend(list(cross_edges))

    # Complete remainder using a more standard BFS strategy
    edge_queue = _bfs_remainder(bgraph, visited, visited_edges, edge_queue, qubits, t_gates)

    return edge_queue


def _queue_bfs_cnot_cycles(
    bgraph: nx.Graph,
    rows: dict[int, int],
    qubits: dict[int, int],
    inputs: list[int],
    other_first_ids: list[int] = [],
    t_gates: dict[int, list[tuple[int, int] | None]] = {},
) -> list[tuple[int, int]]:
    """Build queue using strategy defined in kwargs.

    Args:
        bgraph: The BlockGraph currently being built.
        first_id: The spider ID at which to begin traversing (needed for some algorithms)
        rows: Spiders organised by row.
        qubits: Spiders organised by qubit.
        inputs: The inputs of the base graph.
        graph_traverse_strategy: The graph traversing strategy.
        other_first_ids (optional): Other IDS to use as basis for queue (in BFS-layers strategy)
        t_gates (optional): A dictionary containing the edges of each T-gate pattern in base graph.

    Returns:
        edge_queue: A list of (u, v) tuples to use as queue.

    """

    # Strategy only works if first_spider is an input
    if not inputs:
        raise ValueError(
            "CNOT cycles graph traversing requires the base graph to have at least one input"
        )

    # Initialise key objects
    edge_queue: list[tuple[int, int]] = []
    visited = set([other_first_ids[0]])
    visited_edges: set[tuple[int, int]] = set()
    central_qubit = qubits[other_first_ids[0]]

    central_qubit_cycles: list[tuple[list[int], int]] = []
    outer_qubits_cycles: list[tuple[list[int], int]] = []

    # Extract T-gate patterns is applicable
    t_gate_pats_ids = set()
    for edges_in_t_pat in t_gates.values():
        if edges_in_t_pat:
            t_gate_pats_ids.update([i for e in edges_in_t_pat for i in e])

    # Organise spiders by qubit
    spiders_by_qubit_row: dict[int, list[tuple[int, int]]] = {}
    for spider_id, qubit_number in qubits.items():
        if qubit_number not in spiders_by_qubit_row:
            spiders_by_qubit_row[qubit_number] = [spider_id]
        else:
            spiders_by_qubit_row[qubit_number].append(spider_id)

    # Sort qubit layers
    spiders_by_qubit_row_sorted = {
        k: sorted(v, key=lambda x: rows[x]) for k, v in spiders_by_qubit_row.items()
    }

    # Traverse the full central qubit
    for spider_id in spiders_by_qubit_row_sorted[central_qubit]:
        neighs = bgraph.neighbors(spider_id)
        for neigh_id in neighs:
            if rows[neigh_id] < rows[spider_id]:
                if neigh_id in visited and (neigh_id, spider_id) not in edge_queue:
                    edge_queue.append((neigh_id, spider_id))
                    visited.add(spider_id)
                    visited_edges.add((neigh_id, spider_id))

    nx_cycles_init: list[list[int]] = nx.simple_cycles(bgraph)
    for cycle in nx_cycles_init:
        if any([i in visited for i in cycle]):
            central_qubit_cycles.append((cycle, len(cycle)))
        else:
            outer_qubits_cycles.append((cycle, len(cycle)))
    central_qubit_cycles = sorted(central_qubit_cycles, key=lambda x: x[1], reverse=True)
    outer_qubits_cycles = sorted(outer_qubits_cycles, key=lambda x: x[1], reverse=True)

    # Incorporate cycles into edge_queue
    for cycle_group in [central_qubit_cycles, outer_qubits_cycles]:
        for cycle, _ in cycle_group:
            # Find a starting point in cycle
            cycle_leader_id = None
            for cycle_member_id in cycle:
                if cycle_member_id in visited:
                    cycle_leader_id = cycle_member_id
                    break

            # Rotate cycle so items are in correct neigh-to-neigh sequence
            if cycle_leader_id:
                idx = cycle.index(cycle_leader_id)
            # Skip if no spider in cycle has been visited yet
            else:
                continue

            # Process if there is a valid idx
            rotated_cycle = cycle[idx:] + cycle[:idx]

            # Add spiders sequentially if not already in queue
            first = True
            for i, spider_id in enumerate(rotated_cycle):
                if first:
                    first = False
                    continue
                if ((rotated_cycle[i - 1], spider_id) not in visited_edges) and (
                    (spider_id, rotated_cycle[i - 1]) not in visited_edges
                ):
                    edge_queue.append((rotated_cycle[i - 1], spider_id))
                    visited_edges.add((rotated_cycle[i - 1], spider_id))
                    visited.add(spider_id)
            if (rotated_cycle[-1], rotated_cycle[0]) not in visited_edges and (
                rotated_cycle[0],
                rotated_cycle[-1],
            ) not in visited_edges:
                edge_queue.append((rotated_cycle[-1], rotated_cycle[0]))
                visited_edges.add((rotated_cycle[-1], rotated_cycle[0]))
                visited.add(rotated_cycle[0])

    # Complete remainder using a more standard BFS strategy
    edge_queue = _bfs_remainder(bgraph, visited, visited_edges, edge_queue, qubits, t_gates)

    return edge_queue


def _queue_tfs_cnots(
    bgraph: nx.Graph,
    rows: dict[int, int],
    qubits: dict[int, int],
    inputs: list[int],
    other_first_ids: list[int] = [],
    s_gates: dict[int, list[tuple[int, int] | None]] | None = None,
    t_gates: dict[int, list[tuple[int, int] | None]] | None = None,
) -> list[tuple[int, int]]:
    """Build queue using strategy defined in kwargs.

    Args:
        bgraph: The BlockGraph currently being built.
        first_id: The spider ID at which to begin traversing (needed for some algorithms)
        rows: Spiders organised by row.
        qubits: Spiders organised by qubit.
        inputs: The inputs of the base graph.
        graph_traverse_strategy: The graph traversing strategy.
        other_first_ids (optional): Other IDS to use as basis for queue (in BFS-layers strategy)
        s_gates (optional): A dictionary containing all IDs and edges in S-gate patterns.
        t_gates (optional): A dictionary containing all IDs and edges in T-gate patterns.

    Returns:
        edge_queue: A list of (u, v) tuples to use as queue.

    """

    # Strategy only works if first_spider is an input
    if not inputs:
        raise ValueError(
            "CNOT cycles graph traversing requires the base graph to have at least one input"
        )

    # Initialise key objects
    edge_queue: list[tuple[int, int]] = []
    visited = set([other_first_ids[0]])
    visited_edges: set[tuple[int, int]] = set()
    central_qubit = qubits[other_first_ids[0]]

    # Traverse central qubit
    spiders_by_qubit_row: dict[int, list[tuple[int, int]]] = {}
    for spider_id, qubit_number in qubits.items():
        if qubit_number not in spiders_by_qubit_row:
            spiders_by_qubit_row[qubit_number] = [spider_id]
        else:
            spiders_by_qubit_row[qubit_number].append(spider_id)

    spiders_by_qubit_row_sorted = {
        k: sorted(v, key=lambda x: rows[x]) for k, v in spiders_by_qubit_row.items()
    }

    hold_cnots = []
    for spider_id in spiders_by_qubit_row_sorted[central_qubit]:
        neighs = bgraph.neighbors(spider_id)
        for neigh_id in neighs:
            if rows[neigh_id] < rows[spider_id]:
                if neigh_id in visited and (neigh_id, spider_id) not in edge_queue:
                    edge_queue.append((neigh_id, spider_id))
                    visited.add(spider_id)
                    visited_edges.add((neigh_id, spider_id))
            if rows[neigh_id] == rows[spider_id]:
                if (spider_id, neigh_id) not in edge_queue and (
                    neigh_id,
                    spider_id,
                ) not in edge_queue:
                    hold_cnots.append((spider_id, neigh_id))

    # Resolve T-gates
    if t_gates:
        for t_pattern in t_gates.values():
            # If first edge in a T-pattern is in visited, all T-pattern is in visited
            if t_pattern[0] not in visited_edges:
                # Get path to first spider in T-pattern
                _, path_to_t = nx.multi_source_dijkstra(bgraph, visited, target=t_pattern[0][0])
                u_v_to_t = [(path_to_t[i - 1], s_id) for i, s_id in enumerate(path_to_t) if i != 0]

                # Add any edges in path not already in edge_queue
                for u, v in u_v_to_t:
                    if ((u, v) not in visited_edges) and ((v, u) not in visited_edges):
                        edge_queue.append((u, v))
                        visited_edges.add((u, v))
                        visited.add(v)

                # Add the T-pattern
                edge_queue.extend(t_pattern)
                visited_edges.update(t_pattern)

    # Resolve S-gates
    if s_gates:
        for s_pattern in s_gates.values():
            # If first edge in a T-pattern is in visited, all T-pattern is in visited
            if s_pattern[0] not in visited_edges:
                # Get path to first spider in T-pattern
                _, path_to_t = nx.multi_source_dijkstra(bgraph, visited, target=s_pattern[0][0])
                u_v_to_t = [(path_to_t[i - 1], s_id) for i, s_id in enumerate(path_to_t) if i != 0]

                # Add any edges in path not already in edge_queue
                for u, v in u_v_to_t:
                    if ((u, v) not in visited_edges) and ((v, u) not in visited_edges):
                        edge_queue.append((u, v))
                        visited_edges.add((u, v))
                        visited.add(v)

                # Add the T-pattern
                edge_queue.extend(s_pattern)
                visited_edges.update(s_pattern)

    # Add any remaining CNOTs from central qubit
    if hold_cnots:
        for u, v in hold_cnots:
            if ((u, v) not in visited_edges) and ((v, u) not in visited_edges):
                edge_queue.append((u, v))
                visited.add(v)
                visited_edges.add((u, v))

    edge_queue = _bfs_remainder(bgraph, visited, visited_edges, edge_queue, qubits)

    return edge_queue


def _queue_tfs(
    bgraph: nx.Graph,
    first_id: int,
    inputs: list[int],
    s_gates: dict[int, list[tuple[int, int] | None]] | None = None,
    t_gates: dict[int, list[tuple[int, int] | None]] | None = None,
) -> list[tuple[int, int]]:
    """Build queue using strategy defined in kwargs.

    Args:
        bgraph: The BlockGraph currently being built.
        first_id: The spider ID at which to begin traversing (needed for some algorithms)
        inputs: The inputs of the base graph.
        s_gates (optional): A dictionary containing all IDs and edges in S-gate patterns.
        t_gates (optional): A dictionary containing all IDs and edges in T-gate patterns.

    Returns:
        edge_queue: A list of (u, v) tuples to use as queue.

    """

    # Strategy only works if first_spider is an input
    if not inputs:
        raise ValueError(
            "CNOT cycles graph traversing requires the base graph to have at least one input"
        )

    # Initialise key objects
    edge_queue: list[tuple[int, int]] = []
    visited = set([first_id])
    visited_edges: set[tuple[int, int]] = set()

    # Resolve T-gates
    if t_gates:
        for t_pattern in t_gates.values():
            # If first edge in a T-pattern is in visited, all T-pattern is in visited
            if t_pattern[0] not in visited_edges:
                # Get path to first spider in T-pattern
                _, path_to_t = nx.multi_source_dijkstra(bgraph, visited, target=t_pattern[0][0])
                u_v_to_t = [(path_to_t[i - 1], s_id) for i, s_id in enumerate(path_to_t) if i != 0]

                # Add any edges in path not already in edge_queue
                for u, v in u_v_to_t:
                    if ((u, v) not in visited_edges) and ((v, u) not in visited_edges):
                        edge_queue.append((u, v))
                        visited_edges.add((u, v))
                        visited.add(v)

                # Add the T-pattern
                edge_queue.extend(t_pattern)
                visited_edges.update(t_pattern)

    # Resolve S-gates first
    if s_gates:
        for s_pattern in s_gates.values():
            # If first edge in a S-pattern is in visited, all S-pattern is in visited
            if s_pattern[0] not in visited_edges:
                # Get path to first spider in T-pattern
                _, path_to_t = nx.multi_source_dijkstra(bgraph, visited, target=s_pattern[0][0])
                u_v_to_t = [(path_to_t[i - 1], s_id) for i, s_id in enumerate(path_to_t) if i != 0]

                # Add any edges in path not already in edge_queue
                for u, v in u_v_to_t:
                    if ((u, v) not in visited_edges) and ((v, u) not in visited_edges):
                        edge_queue.append((u, v))
                        visited_edges.add((u, v))
                        visited.add(v)

                # Add the S-pattern
                edge_queue.extend(s_pattern)
                visited_edges.update(s_pattern)

    # Complete remainder using a NX BFS strategy with cross edge priority
    edge_queue = _queue_bfs_cross(bgraph, first_id, edge_queue=edge_queue)

    return edge_queue


def _bfs_remainder(
    bgraph: nx.Graph,
    visited: set,
    visited_edges: set[tuple[int, int]],
    edge_queue: list[tuple[int, int]],
    qubits: dict[int, int],
    t_gates: dict[int, list[tuple[int, int] | None]] = {},
):
    """Traverse rest of the graph using BFS."""

    num_t_gate_edges = (len(t_gates) * 4) if t_gates else 0
    while len(edge_queue) < (len(bgraph.edges()) - num_t_gate_edges):
        add_to_visited = []
        cross_edges = []
        for spider_id in visited:
            neighs = bgraph.neighbors(spider_id)
            if spider_id in t_gates:
                continue
            for neigh_id in neighs:
                if qubits[spider_id] == qubits[neigh_id]:
                    if (spider_id in visited or neigh_id in visited) and (
                        (spider_id, neigh_id) not in visited_edges
                        and (neigh_id, spider_id) not in visited_edges
                    ):
                        edge_queue.append((spider_id, neigh_id))
                        visited_edges.add((spider_id, neigh_id))
                        if spider_id not in visited:
                            add_to_visited.append(spider_id)
                        if neigh_id not in visited:
                            add_to_visited.append(neigh_id)
                elif (spider_id, neigh_id) not in visited_edges and (
                    neigh_id,
                    spider_id,
                ) not in visited_edges:
                    cross_edges.append((spider_id, neigh_id))
                    visited_edges.add((spider_id, neigh_id))
                    add_to_visited.append(neigh_id)

        edge_queue.extend(list(cross_edges))
        visited.update(add_to_visited)

    # Add T-gates
    if t_gates:
        alt_edge_queue: list[tuple[int, int]] = []
        for u, v in edge_queue:
            alt_edge_queue.append((u, v))
            if v in t_gates and t_gates[v][0] not in visited_edges:
                alt_edge_queue.extend(t_gates[v])
                visited_edges.update([(u, v) for u, v in t_gates[v]])
        edge_queue = alt_edge_queue

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
) -> tuple[nx.Graph, set[StandardCoord], set[int]]:
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
        intermediate_ids: A set containing all IDs in the new path.


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
    intermediate_ids = set()
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
            intermediate_ids.update([n_id])

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
    if tgt_zx_block.zx_type == "T":
        taken.update([(tgt_coords[0], tgt_coords[1], tgt_coords[2] - i) for i in range(1, 4)])

    # Check for faux edges in case they're present in path and kill them
    if (
        bgraph.nodes[curr_src_id]["zx_block"].zx_type == "O"
        and bgraph.nodes[curr_tgt_id]["zx_block"].zx_type == "O"
    ):
        bgraph.remove_edge(curr_src_id, curr_tgt_id)

    # Infer pipe kind and other attributes
    else:
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

    return bgraph, taken, intermediate_ids
