"""Example using a random PyZX graphs of arbitrary dimensions.

This script contains an example of how to use Topologiq to perform algorithmic lattice
surgery on a random PyZX graphs. The script is meant as an example of what Topologiq can do
but can also be used as the base for running automated randomised tests.

Usage:
    Run script as given.

Notes:
    While we have identified improvements that might allow Topologiq to handle large graphs,
        this is not yet possible. You will start to see some attempts fail at around 50 spiders,
        which can be recovered by asking Topologiq to increase the number of attempts per graph.
        Graphs over 100 spiders might fail entirely irrespective of number of attempts.

"""

import random

import matplotlib.figure
from pyzx.graph.base import BaseGraph
from pyzx.graph.graph_s import GraphS

from topologiq.assets.pyzx_graphs import random_graph
from topologiq.run_hyperparams import LENGTH_OF_BEAMS, VALUE_FUNCTION_HYPERPARAMS
from topologiq.scripts.runner import runner
from topologiq.utils.interop_pyzx import pyzx_g_to_simple_g


####################
# MAIN RUN MANAGER #
####################
def run_random(
    pyzx_graph: BaseGraph | GraphS,
    fig_data: matplotlib.figure.Figure,
    m_times: int,
    vis_options: tuple[str | None, str | None] = (None, None),
    stop_on_first_success: bool = False,
    log_stats: bool = False,
    debug: int = 0
):
    """Run a single random PyZX graph m_times.

    Args:
        pyzx_graph: The random PyZX graph.
        m_times: How many times to repeat-run the circuit.
        vis_options (optional): Visualisation settings provided as a tuple.
        stop_on_first_success: Whether to stop after the first successful attempt.
        log_stats (optional): If True, triggers automated stats logging to CSV files in `./benchmark/data`.
        debug (optional): Debug mode (0: off, 1: graph manager, 2: pathfinder, 3: pathfinder w. discarded paths).
        fig_data (optional): The visualisation of the input ZX graph (to overlay it over other visualisations).

    """

    # Assemble kwargs
    kwargs: dict[str, tuple[int, int] | int] = {
        "weights": VALUE_FUNCTION_HYPERPARAMS,
        "length_of_beams": LENGTH_OF_BEAMS,
    }

    # call Topologiq on graph if graph is available
    if pyzx_graph is not None:

        # Convert graph to simple graph
        simple_graph = pyzx_g_to_simple_g(pyzx_graph)

        # Call Topologiq on `simple_graph` of circuit
        # Default values for key parameters
        if circuit_name and simple_graph["nodes"] and simple_graph["edges"]:
            _, _, _, _ = runner(
                simple_graph,
                circuit_name,
                max_attempts=m_times,
                stop_on_first_success=stop_on_first_success,
                vis_options=vis_options,
                log_stats=log_stats,
                debug=debug,
                fig_data=fig_data,
                **kwargs
            )

    # Explain why graph wouldn't be available and close shop.
    else:
        print("Try again. Valid graph not available.\nPyZX sometimes generates graphs with disconnected subgraphs, which are incompatible with Topologiq and need to be discarded.")


# ...
if __name__ == "__main__":

    # Topologiq generation parameters
    vis_options = ("final", None)
    stop_on_first_success = True
    log_stats = False
    debug = 1

    # Parameters for random generation of input graph
    seed = 1
    random.seed(seed)
    qubit_n = 5
    depth = 15

    # Get a valid random PyZX circuit graph
    circuit_name = f"random_{seed}_{qubit_n}_{depth}"
    graph_type = "cnot"
    pyzx_graph, fig_data = random_graph(qubit_n, depth, graph_type=graph_type, draw_graph=True)

    # Build and run Topologiq on random graph
    m_times = 1  # Number of times to repeat the run of single random graph
    run_random(pyzx_graph, fig_data, m_times, vis_options=vis_options, stop_on_first_success=stop_on_first_success, log_stats=log_stats, debug=debug)
