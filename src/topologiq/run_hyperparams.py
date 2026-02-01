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
            "deterministic": False,
            "seed": None,
        }
        # ...

        # Give hyperparameters to Topologiq as **kwargs
        if circuit_name and simple_graph["nodes"] and simple_graph["edges"]:
            _, _, _, _ = runner(
                simple_graph,
                circuit_name,
                min_succ_rate=min_pathfinder_success_rate,
                strip_ports=strip_ports,
                hide_ports=hide_ports,
                max_attempts=num_attempts,
                stop_on_first_success=stop_on_first_success,
                vis_options=(vis_0, vis_1),
                log_stats=log_stats,
                debug=debug,
                fig_data=fig_data,
                **kwargs,
            )

Notes:
    It follows from the example above that it is also possible to give Topologiq
        an entirely different set of hyperparameters.

"""

# Weights for the main value function to choose best of several valid paths (length of path, beams broken by path)
VALUE_FUNCTION_HYPERPARAMS = (-1, -1)

# Deterministic or randomised running mode
DETERMINISTIC = False

# Deterministic or randomised running mode
BEAMS_SHORT_LEN = 9

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

