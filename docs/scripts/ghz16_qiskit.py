"""
Example of how to use Topologiq to perform lattice surgery on a
16-qubit GHZ circuit produced using Qiskit. 
"""

import random
import pyzx as zx

from typing import List, Tuple, Union
from pyzx.graph.base import BaseGraph
from qiskit import qasm2

from topologiq.utils.interop_pyzx import pyzx_g_to_simple_g
from topologiq.utils.classes import SimpleDictGraph, StandardBlock
from topologiq.scripts.runner import runner
from topologiq.assets.graphs.qiskit import ghz_qiskit


def ghz16_to_qasm(circuit_name: str) -> str:
    
    # CREATE THE FOUNDATIONAL CIRCUIT
    qc = ghz_qiskit(16)

    # PRINT CIRCUIT FOR INSPECTION
    print(f"\n======> Foundational {circuit_name.upper()}:\n")
    print(qc)

    # CONVERT CIRCUIT TO QASM
    qasm_str = qasm2.dumps(qc)
    print("\n======> QASM string of circuit:\n")
    print(qasm_str)

    # RETURN QASM STRING
    return qasm_str


def qasm_to_pyzx(qasm_str:str) -> BaseGraph:
    
    # QASM TO PYZX
    zx_circuit = zx.Circuit.from_qasm(qasm_str)

    # PYZX CIRCUIT TO PYZX GRAPH
    zx_graph = zx_circuit.to_graph()

    # DRAW INITIAL GRAPH
    zx.draw(zx_graph, labels = True)

    return zx_graph


def pyzx_reduce(zx_graph: BaseGraph) -> BaseGraph:

    # MAKE COPY TO WORK WITH COPY
    zx_graph_copy = zx_graph
    # APPLY STATES AND POST-SELECT
    # States
    num_apply_state = zx_graph_copy.num_inputs()
    zx_graph_copy.apply_state('0' * num_apply_state)

    # No post-select needed
    # Here only to signal place where operation would fall
    zx_graph_copy.apply_effect('////////////////')

    # REDUCE
    zx.full_reduce(zx_graph_copy)

    # DRAW REDUCED GRAPH
    zx.draw(zx_graph_copy, labels = True)

    # RETURN REDUCED GRAPH
    return zx_graph_copy

def zx_graph_to_simple_graph(zx_graph: BaseGraph) -> SimpleDictGraph:
    
    # CONVERT TO TOPOLOGIQ'S NATIVE FORMAR
    simple_graph = pyzx_g_to_simple_g(zx_graph)

    #PRINT SIMPLE GRAPH
    print("\n======> Initial simple graph:")
    print(simple_graph)

    # RETURN SIMPLE GRAPH
    return simple_graph

def run_topologiq(simple_graph: SimpleDictGraph, circuit_name:str) -> Tuple[Union[None, dict[int, StandardBlock]], Union[None, dict[Tuple[int, int], List[str]]]]:
    
    # PARAMS & HYPERPARAMS
    vis = "final"  # Calls 3D visualisation at the end. `None` to deactivate.
    anim = None  # Best to avoid in a public notebook. Animation support depends a lot on the machine and rights.

    VALUE_FUNCTION_HYPERPARAMS = (
        -1,  # Weight for length of path
        -1,  # Weight for number of "beams" broken by path
    )

    kwargs = {
        "weights": VALUE_FUNCTION_HYPERPARAMS,
        "length_of_beams": 9,
    }

    # Run topologiq
    random.seed(11)
    print("\n======> Now calling Topologiq:")
    _, _, lattice_nodes, lattice_edges = runner(
        simple_graph,  # The simple_graph to be processed by Topologiq
        circuit_name,  # Name of the circuit
        min_succ_rate = 80,  # Runtime saving parameter (min % of total possible paths per edge)
        strip_ports = False,  # Remove open boundaries from an incoming graph
        hide_ports = False,  # Leave open boundaries in graph object but hide in visualisations
        max_attempts = 10,  # Maximum # of attempts to find a successful solution
        stop_on_first_success = True,  # Exit when any attempt is successful (False useful for automating stats)
        vis_options = (vis, anim),  # (Visualisation mode, Animation mode)
        log_stats = False,  # Automatically log stats for all runs (requires writing privileges)
        debug = False,  # Enter debug mode (additional detail in visualisation)
        fig_data = None,  # Matplotlib object containing input ZX graph (to overlay over visualisations)
        **kwargs,  # {Weights for value function, Length of beams}
    )

    return lattice_nodes, lattice_edges

if __name__ == "__main__":
    circuit_name = "ghz16"
    qasm_str = ghz16_to_qasm(circuit_name)
    zx_graph_init = qasm_to_pyzx(qasm_str)
    zx_graph_reduced = pyzx_reduce(zx_graph_init)
    simple_graph = zx_graph_to_simple_graph(zx_graph_reduced)
    lattice_nodes, lattice_edges = run_topologiq(simple_graph, circuit_name)