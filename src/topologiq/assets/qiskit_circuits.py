"""Qiskit circuits to use in examples, demonstrations and testing."""

from pathlib import Path

from qiskit import qpy
from qiskit.circuit import QuantumCircuit


def ghz(n_qubits: int, circuit_name: str, draw_circuit: bool = False) -> str:
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


def save_to_qpy(qc: QuantumCircuit, path_to_output: Path | None = None):
    """Save Qiskit circuit to Qiskit's .qpy format.

    Args:
        path_to_output: path to desired .qpy output file.
        qc: Qiskit encoding for the desired circuit

    """

    if not path_to_output:
        path_to_output = Path(__file__).resolve().parent / "qiskit_example_circuit.qpy"

    with open(path_to_output, "wb") as f:
        qpy.dump(qc, f)
