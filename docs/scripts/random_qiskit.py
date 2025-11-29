"""
Run random PyZX graphs of arbitrary dimensions, an arbitrary number of times.

Script loops over `n` random PyZX graphs, running each `m` times and logging
statistics for all runs.

Usage:
    
Notes:

"""

from typing import Tuple

from topologiq.utils.interop_pyzx import pyzx_g_to_simple_g
from topologiq.run_hyperparams import VALUE_FUNCTION_HYPERPARAMS, LENGTH_OF_BEAMS
from topologiq.assets.graphs.pyzx_graphs import random_graph
from topologiq.scripts.runner import runner

####################
# MAIN RUN MANAGER #
####################
def run_random(
    m_times: int,
    qubit_range: Tuple[int, int],
    depth_range: Tuple[int, int],
    vis_options: Tuple[str | None, str | None] = (None, None),
    stop_on_first_success: bool = False,
    log_stats: bool = False,
    debug: int = 0
):
    """Runs a single random PyZX graph m_times.

    Args:
        m_times: How many times to repeat-run the circuit.
        vis_options (optional): Visualisation settings provided as a Tuple.
        log_stats (optional): If True, triggers automated stats logging to CSV files in `.assets/stats/`.
        debug (optional): Debug mode (0: off, 1: graph manager, 2: pathfinder, 3: pathfinder w. discarded paths).
        fig_data (optional): The visualisation of the input ZX graph (to overlay it over other visualisations).
    """

    # Assemble kwargs
    kwargs: dict[str, Tuple[int, int] | int] = {
        "weights": VALUE_FUNCTION_HYPERPARAMS,
        "length_of_beams": LENGTH_OF_BEAMS,
    }

    # Get a random PyZX circuit
    circuit_name: str = f"random_{0}"
    pyzx_graph, fig_data = random_graph(qubit_range, depth_range, draw_graph=True)
    
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
        print("Try again. Valid graph not available.\n\
                PyZX sometimes generates graphs with disconnected subgraphs,\
                which are incompatible with Topologiq and need to be discarted.")


if __name__ == "__main__":
    
    vis_options = ("final", None)
    stop_on_first_success = True
    log_stats = False
    debug = 0

    m_times = 1  # Number of times to repeat the run of single random graph
    qubit_range = (2, 7)  # Min, max number of qubits.
    depth_range = (5, 15) # Min, max depth.
    run_random(m_times, qubit_range, depth_range, vis_options=vis_options, stop_on_first_success=stop_on_first_success, log_stats=log_stats, debug=debug)
