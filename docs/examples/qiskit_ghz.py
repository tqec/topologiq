"""Example of how to use with a Qiskit circuit.

For documentation purposes, the script performs algorithmic lattice surgery on both the
full (unreduced) and the reduced version of the circuit. In real terms,
one would only undertake LS on one of them, probably the reduced version.

Usage:
    Run script as given.

"""

import pyzx as zx
from qiskit.circuit import QuantumCircuit

from topologiq.core.graph_manager.graph_manager import BlockGraphManager
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

    # ORIGINAL GRAPH: QASM -> ZX manager
    print("\n################\n# GHZ CANONICAL #\n################")
    in_zx_graph_manager = ZXGraphManager()
    aug_zx = in_zx_graph_manager.add_graph_from_qasm(
        qasm_str=qasm_str, graph_key=circuit_name
    )
    zx.draw(aug_zx.zx_graph, labels=True)

    # ORIGINAL GRAPH: AugmentedZXGraph -> BlockGraph
    bgraph_manager = BlockGraphManager(aug_zx)
    bgraph_manager.build()
    bgraph_manager.draw_blockgraph()

    # REDUCED GRAPH: QASM -> ZX manager
    print("\n###############\n# GHZ REDUCED #\n###############")
    ghz_reduced = aug_zx.zx_graph.copy()
    zx.full_reduce(ghz_reduced)
    aug_zx_reduced = in_zx_graph_manager.add_graph_from_pyzx(
        ghz_reduced, graph_key=f"{circuit_name}_reduced"
    )
    zx.draw(aug_zx_reduced.zx_graph, labels=True)

    # REDUCED GRAPH: AugmentedZXGraph -> BlockGraph
    bgraph_manager_reduced = BlockGraphManager(aug_zx_reduced)
    bgraph_manager_reduced.build()
    bgraph_manager_reduced.draw_blockgraph()

    # You can also confirm equality of either surgery using the Augmented ZX Graph
    print("\n################\n# VERIFICATION #\n################")
    print("=> GHZ canonical")
    zx_out = bgraph_manager.to_zx_graph()
    equality = aug_zx.check_equality(zx_out)

    print("\n=> GHZ reduced")
    zx_out_reduced = bgraph_manager_reduced.to_zx_graph()
    equality = aug_zx.check_equality(zx_out_reduced)
