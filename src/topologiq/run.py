"""Run Topologiq from the command line using pre-defined example circuits.

Script looks for a PyZX graph if called using '--pyzx'
or a native `simple_graph` if called using '--graph'.

Usage:
    run.py --graph:<name_of_graph> [options]
    run.py --pyzx:<name_of_graph>  [options]
    run.py (-h | --help)
    run.py --version

Options:
    -h, --help              Show this help message and exit.
    --vis:<final|detail>    Visualise the "final" output or "detail" visualisations for each edge.
    --repeat:<n>            Repeat for <n> (integer) times.
    --animate:<GIF|MP4>     Create animation of the entire algorithmic process in GIF or MP4 format.
    --strip_boundaries      Eliminate boundary nodes from graph before running algorithm
    --hide_boundaries       Keep boundary nodes in graph but do not show corresponding cubes in visualisations.
    --log_stats             Log key performance metrics for the run.
    --debug                 Run in debug mode (enables verbose logging and more detailed visualisations).
    --first_id:             Select strategy to use when defining the ID and kind of the first cube placed in 3D space.
                                first_spider: lowest ID of non-boundary spiders.
                                centrality_majority: majority vote from applicable centrality measures.
                                centrality_random (default): random pick from a list of central nodes.

Notes:
    If command uses '--graph', example circuit must exist in `./assets/graphs/simple_graphs.py`.
    If command uses '--pyzx', example circuit must exist in `./assets/graphs/pyzx_graphs.py`.
    MP4 animations require FFmpeg (the actual thing, not just the Python wrapper).
    Examples of how to run this file using combined options are available in the README.

"""

import sys

from matplotlib.figure import Figure

from topologiq.assets import pyzx_graphs, simple_graphs
from topologiq.core.graph_manager.graph_manager import runner
from topologiq.input.pyzx import pyzx_g_to_simple_g
from topologiq.utils.classes import SimpleDictGraph
from topologiq.vis.zx import simple_graph_vis


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

    # Key default parameters
    visualisation_mode: str | None = None  # Change to final for a single visualisation in the end
    animation_mode: str | None = None  # Change to trigger an animation of the entire process
    fig_data: Figure | None = None  # Placeholder, overwrite with a Matplotlib ZX graph visualisation to overlay on 3D visualisations

    # Assemble key kwargs
    # Note! Not a comprehensive list of kwargs.
    # Only included those that can be adjusted from terminal and a couple other that are commonly adjusted.
    # Topologiq has an internal function to assemble full kwargs.
    kwargs = {
        "first_id_strategy": "centrality_random",  # (bool) Change between several strategies for selecting the ID of the first spider
        "seed": None,  # (None | int) Change to use a specific random seed across the entire algorithm
        "max_attempts": 1,  # (int) Change to limit the max number of runs for any given circuit
        "stop_on_first_success": True,  # (bool) Change to force multiple runs for same circuit
        "strip_ports": False,  # (bool) Change to eliminate boundary spiders from ZX graph before processing
        "hide_ports": False,  # (bool) Change to hide boundary spiders/ports in any 3D visualisations
        "log_stats": False,  # (bool) Change to trigger automated performance metrics logs
        "debug": 0,  # (int: 0, 1, 2, 3) Change to turn debug mode on, with increasing level of stringency
    }


    # Handle any arguments passed via the command
    circuit_name: str = None
    simple_graph: SimpleDictGraph = {"nodes": [], "edges": []}
    for arg in sys.argv:
        # Visualisation settings
        if arg == "--strip_boundaries":
            kwargs["strip_ports"] = True

        if arg == "--hide_boundaries":
            kwargs["hide_ports"] = True

        # Look for name of a "simple" or "non-descript" graph
        if arg.startswith("--graph:"):
            circuit_name = arg.replace("--graph:", "")
            simple_graph = getattr(simple_graphs, circuit_name)
            fig_data = simple_graph_vis(simple_graph)

        # Look for name of a PyZX graph
        if arg.startswith("--pyzx:"):
            circuit_name = arg.replace("--pyzx:", "")
            pyzx_function = getattr(pyzx_graphs, circuit_name)
            if "random" not in circuit_name:
                pyzx_graph, fig_data = pyzx_function(draw_graph=True)
            else:
                qubit_range = (2, 7)
                depth_range = (3, 10)
                pyzx_graph, fig_data = pyzx_function(qubit_range, depth_range, draw_graph=True)
            simple_graph = pyzx_g_to_simple_g(pyzx_graph)

        # Look for visualisation options
        if arg.startswith("--vis:"):
            visualisation_mode = arg.replace("--vis:", "")
        if arg.startswith("--animate:"):
            animation_mode = arg.replace("--animate:", "")
        kwargs["vis_options"] = (visualisation_mode, animation_mode)

        # Look for log_stats to file flag
        if arg.startswith("--log_stats"):
            kwargs["log_stats"] = True

        # Look for number of repetitions
        if arg.startswith("--repeat:"):
            kwargs["max_attempts"] = int(arg.replace("--repeat:", ""))
            kwargs["stop_on_first_success"] = False

        # Look for debug mode
        if arg.startswith("--debug"):
            kwargs["debug"] = int(arg.replace("--debug:", ""))

        # Look for debug mode
        if arg.startswith("--first_id"):
            kwargs["first_id_strategy"] = arg.replace("--first_id:", "")

    # Call Topologiq on `simple_graph` of circuit
    if circuit_name and simple_graph["nodes"] and simple_graph["edges"]:
        _, _, _, _ = runner(
            simple_graph,
            circuit_name,
            fig_data=fig_data,
            **kwargs,
        )


if __name__ == "__main__":
    run()
