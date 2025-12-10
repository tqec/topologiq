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
        kwargs: dict[str, tuple[int, int] | int] = {
            "weights": VALUE_FUNCTION_HYPERPARAMS,
            "length_of_beams": LENGTH_OF_BEAMS,
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

# Length of beams
LENGTH_OF_BEAMS = 99
