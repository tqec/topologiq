"""Example of how to use with a Qiskit circuit.

For documentation purposes, the script performs algorithmic lattice surgery on both the
full (unreduced) and the reduced version of the circuit. In real terms,
one would only undertake LS on one of them, probably the reduced version.

Usage:
    Run script as given.

"""

import pyzx as zx
from qiskit.circuit import QuantumCircuit

from topologiq.input.circuit_manager import CircuitManager
from topologiq.input.zx_manager import ZXGraphManager


############
# ENCODING #
############
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


#######
# RUN #
#######
if __name__ == "__main__":
    # Create circuit or import it from somewhere
    n_qubits = 16
    circuit_name = f"ghz_{n_qubits}"
    ghz_circuit = ghz_encoding(n_qubits, circuit_name)

    # qBraid -> QASM
    qbraid_circuit_manager = CircuitManager()
    qasm_str = qbraid_circuit_manager.add_qiskit_circuit(ghz_circuit, key=circuit_name)

    # QASM -> ZX manager
    in_zx_graph_manager = ZXGraphManager()
    augmented_zx_graph_in = in_zx_graph_manager.add_graph_from_qasm(
        qasm_str=qasm_str, graph_key=circuit_name
    )
    zx.draw(augmented_zx_graph_in.zx_graph, labels=True)

    # Run Topologiq on full (unreduced graph)
    bgraph_manager_full = augmented_zx_graph_in.get_blockgraph()
    bgraph_manager_full.draw_blockgraph()

    # Run Topologiq on reduced graph (Augmented ZX Graph always contains the reduced version of the graph)
    zx.draw(augmented_zx_graph_in.zx_graph, labels=True)
    zx.draw(augmented_zx_graph_in.zx_graph_reduced, labels=True)
    bgraph_manager_reduced = augmented_zx_graph_in.get_blockgraph(use_reduced=True)
    bgraph_manager_reduced.draw_blockgraph()

    # You can also confirm equality using the Augmented ZX Graph
    out_zx_graph_manager = ZXGraphManager()
    augmented_zx_graph_out = out_zx_graph_manager.add_graph_from_blockgraph(
        bgraph_manager_reduced, graph_key="ghz_out"
    )
    equality = augmented_zx_graph_in.check_equality(augmented_zx_graph_out)
    print(equality)
