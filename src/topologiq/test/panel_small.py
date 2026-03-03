"""Test panel containing a series of pre-existing circuit examples.

This script tests Topologiq performance using a number of pre-existing circuits
available in `src/topologiq/assets`. The script is not yet tied to an automated
testing pipeline, but is meant to eventually be used for the said purpose.

Usage:
    Run script as given.

"""

from matplotlib.figure import Figure

from topologiq.assets import pyzx_graphs, simple_graphs
from topologiq.core.graph_manager.graph_manager import runner
from topologiq.input.pyzx import pyzx_g_to_simple_g
from topologiq.utils.classes import SimpleDictGraph


####################
# MAIN RUN MANAGER #
####################
def run_panel_of_small_circuits():
    """Run Topologiq on a series of pre-defined circuit.

    This function loops over circuits available in `./assets/graphs/simple_graphs.py` and
    `./assets/graphs/pyzx_graphs.py`. It tests each circuit using all `first_id_strategy`
    selections available.

    """

    # Preliminaries
    fig_data: Figure | None = (
        None  # Placeholder, overwrite with a Matplotlib ZX graph visualisation to overlay on 3D visualisations
    )

    # Commonly modifiable kwargs
    # Note! Not a comprehensive list of kwargs. Topologiq has an internal function to assemble full kwargs.
    # Only included those that can be helpful to adjust to perform quick tests manually.
    visualisation_mode: str | None = None  # Change to final for a single visualisation in the end
    animation_mode: str | None = None  # Change to trigger an animation of the entire process
    kwargs = {
        "vis_options": (visualisation_mode, animation_mode),
        "seed": None,  # (None | int) Change to use a specific random seed across the entire algorithm
        "stop_on_first_success": False,  # (bool) Change to force multiple runs for same circuit
        "log_stats": True,  # (bool) Change to trigger automated performance metrics logs
        "debug": 0,  # (int: 0, 1, 2, 3) Change to turn debug mode on, with increasing level of stringency
    }

    # List of circuits to test
    all_pyzx_circuits = ["cnot", "cnots", "simple_mess"]
    all_simple_graph_circuits = [
        "steane",
        "steane_obfs",
        "hadamard_line",
        "hadamard_bend",
        "hadamard_mess",
    ]

    # List of available strategies for choosing a first ID
    all_first_id_strategies = ["first_spider", "centrality_majority", "centrality_random"]

    for circuit_name in all_pyzx_circuits + all_simple_graph_circuits:
        for strategy in all_first_id_strategies:
            # Test 10 times for deterministic approaches and 100 for probabilistic ones
            kwargs["max_attempts"] = 100 if "random" in strategy else 10

            # Prepare simple graph
            simple_graph: SimpleDictGraph = {"nodes": [], "edges": []}

            # Look for name of a "simple" or "non-descript" graph
            if circuit_name in all_simple_graph_circuits:
                simple_graph = getattr(simple_graphs, circuit_name)

            # Look for name of a PyZX graph
            if circuit_name in all_pyzx_circuits:
                pyzx_function = getattr(pyzx_graphs, circuit_name)
                pyzx_graph, fig_data = pyzx_function(draw_graph=False)
                simple_graph = pyzx_g_to_simple_g(pyzx_graph)

            # Call Topologiq on `simple_graph` of circuit
            if circuit_name and simple_graph["nodes"] and simple_graph["edges"]:
                _, _, _, _ = runner(
                    simple_graph,
                    circuit_name,
                    fig_data=fig_data,
                    **kwargs,
                )


if __name__ == "__main__":
    run_panel_of_small_circuits()
