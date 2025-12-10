"""Example using a 16-qubit GHZ circuit produced using Qiskit.

This script contains an example of how to use Topologiq to perform algorithmic lattice
surgery on a 16-qubit GHZ circuit designed using Qiskit.

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
from qiskit import qasm2
from qiskit.circuit import QuantumCircuit

from topologiq.scripts.runner import runner
from topologiq.utils.classes import StandardBlock
from topologiq.utils.interop_pyzx import pyzx_g_to_simple_g


def ghz_to_qasm(n_qubits: int, circuit_name: str) -> str:
    """Create a n-qubits GHZ and convert it to QASM.

    Args:
        n_qubits: The number of qubits for the GHZ.
        circuit_name: The name of the circuit.

    """
    # Foundational circuit
    qc: QuantumCircuit = QuantumCircuit(n_qubits, name=circuit_name)

    # GHZ encoding
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    print(f"\n======> Foundational {circuit_name.upper()}:\n", qc)

    # Convert to QASM
    qasm_str = qasm2.dumps(qc)
    print("\n======> QASM string of circuit:\n", qasm_str)

    return qasm_str


def qasm_to_pyzx(qasm_str: str) -> BaseGraph:
    """Import a circuit from QASM and convert it to a PyZX graph.

    Args:
        qasm_str: A quantum circuit encoded as a QASM string.

    """
    # QASM --> PyZX circuit --> PyZX graph
    zx_circuit = zx.Circuit.from_qasm(qasm_str)
    zx_graph = zx_circuit.to_graph()

    # Draw
    zx.draw(zx_graph, labels=True)

    return zx_graph


def pyzx_reduce(zx_graph: BaseGraph) -> BaseGraph:
    """Reduce a PyZX graph after applying states to all inputs.

    Args:
        zx_graph: The input ZX graph, given as a PyZX graph.

    """

    # Work with copy
    zx_graph_copy = zx_graph

    # Apply states
    num_apply_state = zx_graph_copy.num_inputs()
    zx_graph_copy.apply_state("0" * num_apply_state)

    # No post-selection needed (but here to signal where to do it if ever necessary)
    zx_graph_copy.apply_effect("////////////////")

    # Reduce & draw
    zx.full_reduce(zx_graph_copy)
    zx.draw(zx_graph_copy, labels=True)

    return zx_graph_copy


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


# ...
if __name__ == "__main__":
    circuit_name = "ghz16"
    n_qubits = 16
    qasm_str = ghz_to_qasm(n_qubits, circuit_name)
    zx_graph_init = qasm_to_pyzx(qasm_str)
    zx_graph_reduced = pyzx_reduce(zx_graph_init)
    lattice_nodes, lattice_edges = run_topologiq(zx_graph_reduced, circuit_name)
