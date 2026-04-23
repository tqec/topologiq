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

import pyzx as zx
from qiskit.circuit import QuantumCircuit

from topologiq.input.pyzx_manager import ZXGraphManager
from topologiq.input.qbraid_manager import CircuitManager


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


# ...
if __name__ == "__main__":
    # Create circuit or import it from somewhere
    n_qubits = 16
    circuit_name = f"ghz_{n_qubits}"
    ghz_circuit = ghz_encoding(n_qubits, circuit_name)

    # qBraid -> QASM
    qbraid_circuit_manager = CircuitManager()
    qasm_str = qbraid_circuit_manager.add_qiskit_circuit(ghz_circuit, key=circuit_name)

    # QASM -> PyZX
    zx_graph_manager = ZXGraphManager()
    aug_zx = zx_graph_manager.add_graph_from_qasm(qasm_str=qasm_str, graph_key=circuit_name)

    # Get inputs
    print(aug_zx.zx_graph_reduced.inputs(), aug_zx.zx_graph_reduced.outputs())

    # Draw ZX graph
    zx.draw(aug_zx.zx_graph, labels=True)
    zx.draw(aug_zx.zx_graph_reduced, labels=True)

    # Run Topologiq
    lattice_nodes, lattice_edges = aug_zx.get_blockgraph(
        circuit_name=circuit_name, use_reduced=True, final_vis=True
    )

    aug_zx_out = zx_graph_manager.add_graph_from_blockgraph(
        blockgraph_cubes=lattice_nodes,
        blockgraph_pipes=lattice_edges,
        graph_key=f"{circuit_name}_out",
        other=aug_zx,
    )

    #aug_zx.check_equality(aug_zx_out)
