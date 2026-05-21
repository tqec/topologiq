"""Example of how to use Topologiq to perform algorithmic lattice surgery (LS) on a Qiskit circuit.

This script contains an example of how to use Topologiq to perform algorithmic LS on a
16-qubit GHZ circuit originally designed in Qiskit. Please note, for documentation purposes,
the LS is performed on both the full (unreduced) and the reduced version of the circuit.
In real terms, one would only produce one LS.

Usage:
    Run script as given.

"""

import pyzx as zx
from qiskit.circuit import QuantumCircuit

from topologiq.input.pyzx_manager import ZXGraphManager
from topologiq.input.qbraid_manager import CircuitManager


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


############
# MAIN RUN #
############
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

    # Note that you can also confirm equality using the Augmented ZX Graph
    out_zx_graph_manager = ZXGraphManager()
    augmented_zx_graph_out = out_zx_graph_manager.add_graph_from_blockgraph(
        bgraph_manager_reduced, graph_key="ghz_out"
    )
    zx.draw(augmented_zx_graph_out.zx_graph)

    equality = augmented_zx_graph_in.check_equality(augmented_zx_graph_out)
    print(equality)
