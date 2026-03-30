"""Example of how to use Topologiq to perform LS on random PyZX graphs.

This script contains an example of how to use Topologiq to perform algorithmic lattice
surgery (LS) on random PyZX graphs.

Usage:
    Run script as given.

Notes:
    While we have identified improvements that might allow Topologiq to handle large graphs,
        this is not yet possible. You will start to see some attempts fail at around 50 spiders,
        which can be recovered by asking Topologiq to increase the number of attempts per graph.
        Graphs over 100 spiders might fail entirely irrespective of number of attempts.

"""

import matplotlib.figure
from pyzx.graph.base import BaseGraph
from pyzx.graph.graph_s import GraphS

from topologiq.assets.pyzx_graphs import random_graph
from topologiq.core.graph_manager.graph_manager import runner
from topologiq.input.pyzx import pyzx_g_to_simple_g
from topologiq.kwargs import VALUE_FUNCTION_HYPERPARAMS


####################
# MAIN RUN MANAGER #
####################
def run_random(pyzx_graph: BaseGraph | GraphS, fig_data: matplotlib.figure.Figure, **kwargs):
    """Run a single random PyZX graph m_times.

    Args:
        pyzx_graph: The random PyZX graph.
        fig_data (optional): The visualisation of the input ZX graph (to overlay it over other visualisations).
        **kwargs: See `./kwargs.py` for a comprehensive breakdown.
            NB! If an arbitrary kwarg is not given explicitly, it is created against defaults on `./src/topologiq/kwargs.py`.
            NB! By extension, it only makes sense to give the specific kwargs where user wants to deviate from defaults.

    """

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
                fig_data=fig_data,
                **kwargs,
            )

    # Explain why graph wouldn't be available and close shop.
    else:
        print(
            "Try again. Valid graph not available.\nPyZX sometimes generates graphs with disconnected subgraphs, which are incompatible with Topologiq and need to be discarded."
        )


# ...
if __name__ == "__main__":
    # Topologiq generation parameters
    kwargs = {
        "weights": VALUE_FUNCTION_HYPERPARAMS,
        "first_id_strategy": "first_spider",
        "seed": 42,
        "vis_options": ("final", None),
        "max_attempts": 1,
        "stop_on_first_success": True,
        "debug": 1,
        "log_stats": True,
    }

    # General description of circuit
    qubit_n = 5
    depth = 10
    circuit_name = f"random_{kwargs['seed'] if kwargs.get('seed') else 'noseed'}_{qubit_n}_{depth}"

    # Get a valid random PyZX circuit graph
    graph_type = "cnot"  # Tells PyZX to generate a circuit based on CNOTS
    pyzx_graph, fig_data = random_graph(
        qubit_n, depth, graph_type=graph_type, draw_graph=True, **kwargs
    )

    # Build and run Topologiq on random graph
    run_random(
        pyzx_graph,
        fig_data,
        **kwargs,
    )
