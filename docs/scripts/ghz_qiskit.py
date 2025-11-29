""" Perform lattice surgery on a 16-qubit GHZ circuit produced using Qiskit."""
import random
import pyzx as zx

from typing import List, Tuple, Union
from pyzx.graph.base import BaseGraph
from qiskit import qasm2
from qiskit.circuit import QuantumCircuit

from topologiq.utils.interop_pyzx import pyzx_g_to_simple_g
from topologiq.utils.classes import StandardBlock
from topologiq.scripts.runner import runner


def ghz_to_qasm(n_qubits: int, circuit_name: str) -> str:
    # Foundational circuit
    qc: QuantumCircuit = QuantumCircuit(n_qubits, name=circuit_name)
    
    # GHZ encoding
    qc.h(0)
    for i in range(n_qubits-1):
        qc.cx(i, i+1)
    print(f"\n======> Foundational {circuit_name.upper()}:\n", qc)

    # Convert to QASM
    qasm_str = qasm2.dumps(qc)
    print("\n======> QASM string of circuit:\n", qasm_str)

    return qasm_str

def qasm_to_pyzx(qasm_str:str) -> BaseGraph:
    # QASM --> PyZX circuit --> PyZX graph
    zx_circuit = zx.Circuit.from_qasm(qasm_str)
    zx_graph = zx_circuit.to_graph()

    # Draw
    zx.draw(zx_graph, labels = True)

    return zx_graph


def pyzx_reduce(zx_graph: BaseGraph) -> BaseGraph:

    # Work with copy
    zx_graph_copy = zx_graph

    # Apply states
    num_apply_state = zx_graph_copy.num_inputs()
    zx_graph_copy.apply_state('0' * num_apply_state)

    # No post-select needed (here to signal where operation would fall if needed)
    zx_graph_copy.apply_effect('////////////////')

    # Reduce & draw
    zx.full_reduce(zx_graph_copy)
    zx.draw(zx_graph_copy, labels = True)

    return zx_graph_copy


def run_topologiq(
    zx_graph: BaseGraph,
    circuit_name:str
) -> Tuple[
    Union[None, dict[int, StandardBlock]],
    Union[None, dict[Tuple[int, int], List[str]]]
]:

    # Call Topologiq
    print("\n======> Now calling Topologiq:")
    random.seed(11)
    simple_graph = pyzx_g_to_simple_g(zx_graph)  # PyZX graph --> Topologiq's native format
    _, _, lattice_nodes, lattice_edges = runner(
        simple_graph,  # The simple_graph to be processed by Topologiq
        circuit_name,  # Name of the circuit
        stop_on_first_success=True,  # Exit when any attempt is successful
        vis_options=("final", None),  # (Visualisation mode, Animation mode)
    )

    return lattice_nodes, lattice_edges


if __name__ == "__main__":
    circuit_name = "ghz16"
    n_qubits = 16
    qasm_str = ghz_to_qasm(n_qubits, circuit_name)
    zx_graph_init = qasm_to_pyzx(qasm_str)
    zx_graph_reduced = pyzx_reduce(zx_graph_init)
    lattice_nodes, lattice_edges = run_topologiq(zx_graph_reduced, circuit_name)