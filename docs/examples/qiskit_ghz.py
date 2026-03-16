"""Example of how to use Topologiq to perform LS on a 16-qubit GHZ designed in Qiskit.

This script contains an example of how to use Topologiq to perform algorithmic lattice
surgery (LS) on a 16-qubit GHZ circuit originally designed in Qiskit.

Usage:
    Run script as given.

Notes:
    There is a critical step not shown in visualisations. The reduced PyZX graph contains
        one colour spider an many boundaries, which violates the max_edges == 4 constraint
        needed to perform lattice surgery based on the surface code. Topologiq has in-built
        subroutines to handle these situations. In particular, for single_spider graphs,
        Topologiq calculates the optimal number of spiders needed to have a graph where all
        spiders have the maximum number of edges allowed but not more than possible.

"""

import random

import pyzx as zx
from pyzx.graph.base import BaseGraph
from qiskit.circuit import QuantumCircuit

from topologiq.core.graph_manager.graph_manager import runner
from topologiq.input.pyzx_manager import ZXGraphManager, pyzx_g_to_simple_g
from topologiq.input.qbraid_manager import CircuitManager
from topologiq.utils.classes import StandardBlock


def ghz_encoding(n_qubits: int, circuit_name: str, draw_circuit: bool = False) -> str:
    """Create a GHZ circuit with n-qubits.

    Args:
        n_qubits: The number of qubits for the GHZ.
        circuit_name: The name of the circuit.
        draw_circuit: Whether to pop-up PyZX graph visualisation or not.

    """
    # Foundational circuit
    qc: QuantumCircuit = QuantumCircuit(n_qubits, name=circuit_name)

    # GHZ encoding
    qc.reset(0)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.reset(i + 1)
        qc.cx(i, i + 1)

    if draw_circuit:
        print(f"\n======> QISKIT circuit: {circuit_name.upper()}.\n", qc)

    return qc


def run_topologiq(
    zx_graph: BaseGraph, circuit_name: str
) -> tuple[
    dict[int, StandardBlock] | None,
    dict[tuple[int, int], list[str]] | None,
]:
    """Call Topologiq on an arbitrary PyZX graph.

    Args:
        zx_graph: The input ZX graph, given as a PyZX graph.
        circuit_name: The name of the circuit.

    """

    # Add kwargs for visualisation as desired in this particular example
    # Only add kwargs when you want to deviate from default. Others will be autocompleted on run.
    kwargs = {"vis_options": ("final", None)}  # (Visualisation mode, Animation mode)

    # Add a seed for replicability, or comment out if desired
    random.seed(11)

    print("\n======> Now calling Topologiq:")
    simple_graph = pyzx_g_to_simple_g(zx_graph)  # PyZX graph --> Topologiq's native format
    _, _, lattice_nodes, lattice_edges = runner(
        simple_graph,  # The simple_graph to be processed by Topologiq
        circuit_name,  # Name of the circuit
        stop_on_first_success=True,  # Exit when any attempt is successful
        **kwargs,
    )

    return lattice_nodes, lattice_edges


# ...
if __name__ == "__main__":
    # Create circuit
    # Example uses in-script circuit for clarity,
    # but it is also possible to import it from wherever.
    n_qubits = 16
    circuit_name = f"ghz_{n_qubits}"
    ghz_circuit = ghz_encoding(n_qubits, circuit_name)

    # qBraid -> QASM
    qbraid_circuit_manager = CircuitManager()
    qasm_str = qbraid_circuit_manager.add_qiskit_circuit(ghz_circuit, key=circuit_name)

    # QASM -> PyZX
    zx_graph_manager = ZXGraphManager()
    aug_zx = zx_graph_manager.add_graph_from_qasm(qasm_str=qasm_str, graph_key=circuit_name)

    # Draw ZX graph
    zx.draw(aug_zx.zx_graph)
    zx.draw(aug_zx.zx_graph_reduced)

    # Run Topologiq
    lattice_nodes, lattice_edges = run_topologiq(aug_zx.zx_graph_reduced, circuit_name)
