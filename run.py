# This file is the main runner script for examples ran via the command line.
#
# It picks up the arguments from the command line and
# calls the algorithm on the specified circuit with hyperparameters from run_hyperparams.py.
#
# The file is compatible with any graph in the folder `./assets/graphs`,
# but the graph needs to exist in either `assets/graphs/simple_graphs.py` or `assets/graphs/pyzx_graphs.py`
#

import sys
from scripts.runner import runner
from assets.graphs import simple_graphs
from utils.interop_pyzx import pyzx_g_to_simple_g
from assets.graphs import pyzx_graphs
from utils.classes import SimpleDictGraph
from run_hyperparams import VALUE_FUNCTION_HYPERPARAMS, LENGTH_OF_BEAMS


####################
# MAIN RUN MANAGER #
####################
def run():
    """Runs algorithm when prompted from command line, using args given in command.

    Args (function does not take args directly, but args below can be given to it via the command):
        - `--graph:<name_of_graph>`: name of the graph in `./assets/graphs/simple_graphs.py` to run.
        - `--pyzx:<name_of_pyzx_graph>`: name of the graph in `./assets/graphs/pyzx_graphs.py` to run.
        - `--strip_boundaries`: instructs the algorithm to eliminate any boundary nodes and their corresponding edges
                (without it, nodes are factored into the process and shown on visualisation).
        - `--hide_boundaries`: instructs the algorithm to use boundary nodes but do not display them in visualisation
                (without it, nodes are factored into the process and shown on visualisation).

    Returns:
        - n/a. All outputs below result from separate functions at different parts of the process.
            - Pops up an interactive 3D viewer for user to examine the space-time diagram produced as output.
            - Saves a TXT file with detailed results to `.outputs/txt/` folder.
            - Saves a GIF or MP4 animation to `.outputs/media/` folder.
            - IMPORTANT! This is NOT the function you need to use any of the above programmatically – see `./scripts/runner.py` for that.

    """

    # ASSEMBLE KWARGS
    kwargs = {
        "weights": VALUE_FUNCTION_HYPERPARAMS,
        "length_of_beams": LENGTH_OF_BEAMS,
    }

    # GET CIRCUIT
    c_name: str | None = None
    c_g_dict: SimpleDictGraph = {"nodes": [], "edges": []}

    # READ AND HANDLE ARGUMENTS
    num_attempts = 0
    stop_on_first_success = True
    vis_0, vis_1 = (None, None)
    strip_ports: bool = False
    hide_ports: bool = False
    log_stats: bool = False
    for arg in sys.argv:

        # Visualisation settings
        if arg == "--strip_boundaries":
            strip_ports = True

        if arg == "--hide_boundaries":
            hide_ports = True

        # Look for name of a "simple" or "non-descript" graph
        if arg.startswith("--graph:"):
            c_name = arg.replace("--graph:", "")
            c_g_dict = getattr(simple_graphs, c_name)

        # Look for name of a PyZX graph
        if arg.startswith("--pyzx:"):
            c_name = arg.replace("--pyzx:", "")
            pyzx_function = getattr(pyzx_graphs, c_name)
            g = pyzx_function(draw_graph=True)
            c_g_dict = pyzx_g_to_simple_g(g)
        
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
        else:
            num_attempts = 10

    # CALL ALGORITHM ON CIRCUIT
    if c_name and c_g_dict["nodes"] and c_g_dict["edges"]:
        _, _, _, _ = runner(
            c_g_dict,
            c_name,
            strip_ports=strip_ports,
            hide_ports=hide_ports,
            max_attempts=num_attempts,
            stop_on_first_success=stop_on_first_success,
            visualise=(vis_0, vis_1),
            log_stats=log_stats,
            **kwargs
        )


if __name__ == "__main__":
    run()
