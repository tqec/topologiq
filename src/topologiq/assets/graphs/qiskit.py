"""
Saves Qiskit circuits into Qiskit's .qpy native format. 

Usage:
- Call the circuit encoding if you only need the circuit. 
- Save circuit to Qiskit's .qpy file using `save_to_qpy()` and load circuit from file.
"""

from pathlib import Path
from qiskit.circuit import QuantumCircuit
from qiskit import qpy


def ghz_qiskit(n_qubits: int) -> QuantumCircuit:
    """Return a Qiskit encoding for a n-qubit GHZ.

    Args
        n_qubits: number of qubits for the GHZ. 

    Return:
        qc: Qiskit encoding for the n-qubit GHZ
    """

    # CIRCUIT NAME
    circuit_name = "ghz16"

    # ENCODING
    qc: QuantumCircuit = QuantumCircuit(16, name=circuit_name)
    qc.h(0)
    for i in range(n_qubits-1):
        qc.cx(i, i+1)

    return qc


def save_to_qpy(path_to_output: Path, qc: QuantumCircuit):
    """Save a Qiskit circuit to Qiskit's native .qpy format.

    Args: 
        path_to_output: path to desired .qpy output file.
        qc: Qiskit encoding for the desired circuit

    Returns
        n/a. Function saves to file. 
    """

    with open(path_to_output, 'wb') as f:
        qpy.dump(qc, f)
