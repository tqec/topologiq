import sys
import json
from typing import cast
from scripts.runner import runner
from assets.graphs import simple_graphs
from utils.interop_pyzx import get_simple_graph_from_pyzx
from assets.graphs import pyzx_graphs
from utils.classes import SimpleDictGraph
from run_hyperparams import (
    VALUE_FUNCTION_HYPERPARAMS,
    LENGTH_OF_BEAMS,
    MAX_PATHFINDER_SEARCH_SPACE,
)


####################
# MAIN RUN MANAGER #
####################
def run():

    # ASSEMBLE KWARGS

    kwargs = {
        "weights": VALUE_FUNCTION_HYPERPARAMS,
        "length_of_beams": LENGTH_OF_BEAMS,
        "max_search_space": MAX_PATHFINDER_SEARCH_SPACE,
    }

    # GET CIRCUIT
    circuit_name: str | None = None
    circuit_graph_dict: SimpleDictGraph = {"nodes": [], "edges": []}
    for arg in sys.argv:

        # Named "simple" graph
        if arg.startswith("--graph:"):

            circuit_name = arg.replace("--graph:", "")
            circuit_graph_dict = getattr(simple_graphs, circuit_name)

        # From PyZX
        if arg.startswith("--pyzx:"):
            circuit_name = arg.replace("--pyzx:", "")
            pyzx_function = getattr(pyzx_graphs, circuit_name)
            g = pyzx_function(draw_graph=False)
            circuit_graph_dict = get_simple_graph_from_pyzx(g)

    # CALL ALGORITHM ON CIRCUIT
    if circuit_name and circuit_graph_dict["nodes"] and circuit_graph_dict["edges"]:
        runner(circuit_graph_dict, circuit_name, **kwargs)


if __name__ == "__main__":
    run()
