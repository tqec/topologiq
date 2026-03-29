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
            "vis_options": (None, None),
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

# Weights for the main value function to choose best of several valid paths (length of path, beams broken by path)
VALUE_FUNCTION_HYPERPARAMS = (-1, -1)

# Strategy for selecting the ID of the first spider processed by the algorithm
# centrality_majority: Use a majority vote from several centrality measures
# centrality_random: Pick randomly from a list of central spiders
# first_spider: Select lowest ID non-boundary spider (typically first spider on first qubit)
FIRST_ID_STRATEGY = "first_spider"

# Deterministic or randomised running mode
BEAMS_SHORT_LEN = 7

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
