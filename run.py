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
from utils.interop_pyzx import get_simple_graph_from_pyzx
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
    circuit_name: str | None = None
    circuit_graph_dict: SimpleDictGraph = {"nodes": [], "edges": []}

    # READ AND HANDLE ARGUMENTS
    strip_boundaries: bool = False
    hide_boundaries: bool = False
    for arg in sys.argv:

        # Visualisation settings
        if arg == "--strip_boundaries":
            strip_boundaries = True

        if arg == "--hide_boundaries":
            hide_boundaries = True

        # Look for name of a "simple" or "non-descript" graph
        if arg.startswith("--graph:"):
            circuit_name = arg.replace("--graph:", "")
            circuit_graph_dict = getattr(simple_graphs, circuit_name)

        # Look for name of a PyZX graph
        if arg.startswith("--pyzx:"):
            circuit_name = arg.replace("--pyzx:", "")
            pyzx_function = getattr(pyzx_graphs, circuit_name)
            g = pyzx_function(draw_graph=True)
            circuit_graph_dict = get_simple_graph_from_pyzx(g)

    # CALL ALGORITHM ON CIRCUIT
    if circuit_name and circuit_graph_dict["nodes"] and circuit_graph_dict["edges"]:
        _, _, _, _ = runner(
            circuit_graph_dict,
            circuit_name,
            strip_boundaries=strip_boundaries,
            hide_boundaries=hide_boundaries,
            visualise=("final", "MP4"),
            **kwargs
        )


if __name__ == "__main__":
    run()
