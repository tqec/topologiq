import sys
import json
from typing import cast
from scripts.runner import runner
from assets.graphs import simple_graphs
from utils.interop_pyzx import get_simple_graph_from_pyzx
from assets.graphs.pyzx_graphs import cnot
from utils.classes import SimpleDictGraph, GraphEdge
from run_hyper_params import (
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
        # From arguments in command
        if arg.startswith("--simple:"):

            circuit_name = arg.replace("--simple:", "")
            circuit_graph_dict = getattr(simple_graphs, circuit_name)

        # Using a pre-determined PyZX script
        if arg.startswith("--pyzx:"):
            circuit_name = arg.replace("--pyzx:", "")

            if circuit_name == "cnot":
                g = cnot()
                circuit_graph_dict = get_simple_graph_from_pyzx(g)

    # CALL ALGORITHM ON CIRCUIT
    if circuit_name and circuit_graph_dict["nodes"] and circuit_graph_dict["edges"]:
        runner(circuit_graph_dict, circuit_name, **kwargs)


def get_circuit_from_json(path_to_json: str) -> SimpleDictGraph:

    circuit_graph_dict: SimpleDictGraph = {"nodes": [], "edges": []}

    with open(path_to_json, "r") as f:
        json_string = f.read()
        f.close()

    data = json.loads(json_string)

    for n in data["nodes"]:
        circuit_graph_dict["nodes"].append(tuple(n))

    for e in data["edges"]:
        e_tuple = cast(GraphEdge, tuple([tuple(e[0]), e[1]]))
        circuit_graph_dict["edges"].append(e_tuple)

    return circuit_graph_dict


if __name__ == "__main__":
    run()
