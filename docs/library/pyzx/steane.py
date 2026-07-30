"""PyZX graphs to use in examples, demonstrations and testing.

Usage:
    Call any graph from a separate script.

"""

import pyzx as zx
from pyzx.graph.base import BaseGraph
from pyzx.graph.graph_s import GraphS


############
# ENCODING #
############
def steane() -> BaseGraph | GraphS:
    """Return a PyZX graph of a Steane code.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.
        use_bialgebra: Whether to apply the bialgebra rule to graph or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """

    # Foundational circuit
    pyzx_circuit = zx.Circuit(10)

    ancilla_qubits = [0, 1, 2]
    qubits = [[3, 4, 5, 6], [3, 4, 7, 8], [3, 5, 7, 9]]
    for i, ancilla_qubit in enumerate(ancilla_qubits):
        pyzx_circuit.add_gate("HAD", ancilla_qubit)
        for qubit in qubits[i]:
            pyzx_circuit.add_gate("CNOT", ancilla_qubit, qubit)
        pyzx_circuit.add_gate("HAD", ancilla_qubit)
    pyzx_graph = pyzx_circuit.to_graph()

    # States & effects
    num_apply_state = pyzx_graph.num_inputs()
    pyzx_graph.apply_state("0" * num_apply_state)
    pyzx_graph.apply_effect("000///////")

    return pyzx_graph


########
# CALL #
########
if __name__ == "__main__":
    steane = steane()
