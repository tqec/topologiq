"""Hyperparameters for use by the algorithm.

Usage:
    Import hyperparameters before calling Topologiq on a circuit,
        and give them as **kwargs to Topologiq's main runner.

Example:
    The following demonstrates how you would call and use hyperparameters.

    ::

        # Import the hyperparameters.
        from topologiq.run_hyperparams import LENGTH_OF_BEAMS, VALUE_FUNCTION_HYPERPARAMS
        # ...

        # Assemble hyperparameters as kwargs
        kwargs = {
            "weights": VALUE_FUNCTION_HYPERPARAMS,
            "first_id_strategy": FIRST_ID_STRATEGY,
            "beams_len_short": BEAMS_SHORT_LEN,
            "seed": SEED,
            "animate": ANIMATE,
            "max_attempts": MAX_ATTEMPTS,
            "stop_on_first_success": STOP_ON_FIRST_SUCCESS,
            "min_succ_rate": MIN_SUCC_RATE,
            "strip_ports": STRIP_PORTS,
            "hide_ports": HIDE_PORTS,
            "log_stats": LOG_STATS,
            "log_stats_id": None,
            "debug": DEBUG,
        }
        # ...

        # Give hyperparameters to Topologiq as **kwargs
        if circuit_name and simple_graph["nodes"] and simple_graph["edges"]:
            _, _, _, _ = runner(
                simple_graph,
                circuit_name,
                fig_data=None,
                first_cube=(None, None),
                **kwargs,
            )

Notes:
    It follows from the example above that it is also possible to give Topologiq
        an entirely different set of hyperparameters.

"""

# Weights for the main value function to choose best path
# Hyperparams A tuple of integers where 1st item weighs the length of path and 2nd the number of beams broken by path.
# Z_stretch: Int or None, to favour paths that move along Z axis and apply graph bounds.
# Gravity: Int or None, to favour paths that end closer to a specific point in graph
VALUE_FUNCTION_HYPERPARAMS = (-1, -1)
Z_STRETCH = 0
GRAVITY = 0

# Strategy for selecting the ID of the first spider processed by the algorithm
# - first-spider: Select lowest ID non-boundary spider (typically first spider on first qubit)
# - random: Pick randomly from all non-boundary spiders
# - centrality-random: Pick randomly from a list of central spiders
# - centrality-majority: Use a majority vote from several centrality measures
# - central-qubit: Lowest ID in most central qubit (defined as the qubit with most CNOTs)
# - central-in-first-cycle: Pick the central spider from the first cycle in graph (per nx.cycle_basis).
FIRST_ID_STRATEGY = "first-spider"

# Strategy for graph traversing
# - bfs: Standard BFS tree search including cross edges
# - bfs-cross: Standard BFS giving priority to cross edges
# - bfs-cross-boundaries-last: Standard BFS giving priority to cross edges and holding boundaries for the end
# - bfs-cycles: BFS per cycles (per nx.cycle_basis) with bridge recovery subroutine (to join disconnected cycles) and boundary handling at the end
# - bfs-rows: BFS per (ZX) rows in the graph, one row at a time
# - bfs-cnots: BFS using central qubit and graph CNOTs as pillars (almost no longer a BFS but let's say it is).
# - tfs-cnots: Combines BFS cnots with priority queuing of all edges needed to complete T-gates before traversing rest of the graph.
# - tfs: Starts at any given node and finds shortest paths between visited spiders and all T-gates, then traverses the graph BFS.
GRAPH_TRAVERSE_MODE = "bfs-cross"

# Length of short beams
# (Long beams are always np.inf)
BEAMS_SHORT_LEN = 2

# Single seed to use across any randomised operations
SEED = None

# Max. number of runs for any given circuit
MAX_ATTEMPTS = 1

# Stop on first successful outcome for a given circuit or force multiple runs for same circuit
STOP_ON_FIRST_SUCCESS = True

# Force pathfinder to return more paths
MIN_SUCC_RATE = 60

# Eliminate boundary spiders from ZX graph before processing
STRIP_PORTS = False

# Hide boundary spiders/ports in any 3D visualisations
HIDE_PORTS = False

# Trigger automated performance metrics logs
LOG_STATS = False

# Turn debug mode on, with increasing level of stringency: 0 -> 4
DEBUG = 0

# Override value for first cube placement
FIRST_CUBE = (None, None)

# Default vis options
ANIMATE = None

# Whether to check and add twins
TWINS = True

# Whether to trigger any post-processing
POST_PROCESS = False
