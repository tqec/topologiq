"""Save Qiskit circuits into Qiskit's .qpy native format.

Usage:
    Call `save_to_qpy` from a separate circuit with a path for the desired .qpy file and a qiskit circuit encoding.

"""

from pathlib import Path

from qiskit import qpy
from qiskit.circuit import QuantumCircuit


def save_to_qpy(path_to_output: Path, qc: QuantumCircuit):
    """Save a Qiskit circuit to Qiskit's native .qpy format.

    Args:
        path_to_output: path to desired .qpy output file.
        qc: Qiskit encoding for the desired circuit

    Returns:
        n/a. Function saves to file.

    """

    with open(path_to_output, 'wb') as f:
        qpy.dump(qc, f)
