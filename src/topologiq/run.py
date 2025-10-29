"""
Run Topologiq from the command line using pre-defined example circuits.

Script looks for a PyZX graph if called using '--pyzx'
or a native `simple_graph` if called using '--graph'.

Usage:
    run.py --graph:<name_of_graph> [options]
    run.py --pyzx:<name_of_graph>  [options]
    run.py (-h | --help)
    run.py --version

Options:
    -h, --help              Show this help message and exit.
    --vis:<final|detail>    Visualise only the final space-time diagram.
    --repeat:<n>            Repeat for <n> (integer) times.
    --animate:<GIF|MP4>     Create animation of the entire algorithmic process in GIF or MP4 format.
    --strip_boundaries      Eliminate boundary nodes from graph before running algorithm
    --hide_boundaries       Keep boundary nodes in graph but do not show corresponding cubes in visualisations.
    --log_stats             Log key performance metrics for the run.
    --debug                 Run in debug mode (enables verbose logging and more detailed visualisations).
     

Notes:
    If command uses '--graph', example circuit must exist in `./assets/graphs/simple_graphs.py`.
    If command uses '--pyzx', example circuit must exist in `./assets/graphs/pyzx_graphs.py`.
    MP4 animations require FFmpeg (the actual thing, not just the Python wrapper).
    Examples of how to run this file using combined options are available in the README.
"""

import sys
from typing import Tuple
from matplotlib.figure import Figure

from topologiq.scripts.runner import runner
from topologiq.utils.interop_pyzx import pyzx_g_to_simple_g
from topologiq.utils.simple_grapher import simple_graph_vis
from topologiq.run_hyperparams import VALUE_FUNCTION_HYPERPARAMS, LENGTH_OF_BEAMS
from topologiq.utils.classes import SimpleDictGraph
from topologiq.assets.graphs import simple_graphs, pyzx_graphs


####################
# MAIN RUN MANAGER #
####################
def run():
    """Execute command-line instruction to run Topologiq on a pre-defined circuit.

    This function serves as main entry point for command-line operations. It relies 
    on arguments passed via the command-line and requires calling a pre-defined example
    circuit available from either `./assets/graphs/simple_graphs.py`
    or `./assets/graphs/pyzx_graphs.py`.

    Outputs:
        The purpose of this function is to produce visualisations and file-based outputs. 
        Depending on the options given via the command line, the function will:
            * Save a TXT file with detailed information about the run.
            * (Optional) Pop up an interactive 3D viewer to examine process and/or outcome.
            * (Optional) Save a GIF or MP4 animation.

    Returns:
        None: This function does not produce objects for programmatic use.

    Notes:
        To run Topologiq for programmatic use (i.e., call it from another script), see `./scripts/runner.py`.
    """

    # Misc options
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__)
        sys.exit(0)

    # Assemble kwargs
    kwargs: dict[str, Tuple[int, int] | int] = {
        "weights": VALUE_FUNCTION_HYPERPARAMS,
        "length_of_beams": LENGTH_OF_BEAMS,
    }

    # Get the circuit
    circuit_name: str = None
    simple_graph: SimpleDictGraph = {"nodes": [], "edges": []}

    # Default values for key parameters
    num_attempts: int = 10
    vis_0: str | None = None
    vis_1: str | None = None
    stop_on_first_success: bool = True
    min_pthfinder_success_rate: int = 60
    strip_ports: bool = False
    hide_ports: bool = False
    log_stats: bool = False
    debug: bool = False
    fig_data: Figure | None = None

    # Handle any arguments passed via the command
    for arg in sys.argv:

        # Visualisation settings
        if arg == "--strip_boundaries":
            strip_ports = True

        if arg == "--hide_boundaries":
            hide_ports = True

        # Look for name of a "simple" or "non-descript" graph
        if arg.startswith("--graph:"):
            circuit_name = arg.replace("--graph:", "")
            simple_graph = getattr(simple_graphs, circuit_name)
            fig_data = simple_graph_vis(simple_graph)

        # Look for name of a PyZX graph
        if arg.startswith("--pyzx:"):
            circuit_name = arg.replace("--pyzx:", "")
            pyzx_function = getattr(pyzx_graphs, circuit_name)
            pyzx_graph, fig_data = pyzx_function(draw_graph=True)
            simple_graph = pyzx_g_to_simple_g(pyzx_graph)

        # Look for visualisation options
        if arg.startswith("--vis:"):
            vis_0 = arg.replace("--vis:", "")

        # Look for animation options
        if arg.startswith("--animate:"):
            vis_1 = arg.replace("--animate:", "")

        # Look for log_stats to file flag
        if arg.startswith("--log_stats"):
            log_stats = True

        # Look for number of repetitions parameter
        if arg.startswith("--repeat:"):
            num_attempts = int(arg.replace("--repeat:", ""))
            stop_on_first_success = False

            # Look for number of repetitions parameter
        if arg.startswith("--debug"):
            debug = True

    # Call Topologiq on `simple_graph` of circuit
    if circuit_name and simple_graph["nodes"] and simple_graph["edges"]:
        _, _, _, _ = runner(
            simple_graph,
            circuit_name,
            min_succ_rate=min_pthfinder_success_rate,
            strip_ports=strip_ports,
            hide_ports=hide_ports,
            max_attempts=num_attempts,
            stop_on_first_success=stop_on_first_success,
            vis_options=(vis_0, vis_1),
            log_stats=log_stats,
            debug=debug,
            fig_data=fig_data,
            **kwargs
        )


if __name__ == "__main__":
    run()
