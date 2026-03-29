"""Qrisp circuits to use in examples, demonstrations and testing."""

from qrisp import QuantumBool, cx, h, measure, reset


def steane_qrisp():
    """Create a Steane code encoding using Qrisp.

    Returns:
        qc: the Qrisp encoding for the Steane code.

    """
    # Ancilla & data qubits
    num_ancilla = 3
    num_data = 7

    ancilla_qubits = [QuantumBool() for _ in range(num_ancilla)]
    data_qubits = [QuantumBool() for _ in range(num_data)]

    for q in range(num_ancilla):
        reset(ancilla_qubits[q])

    for q in range(num_data):
        reset(data_qubits[q])

    sequences = [[0,1,2,3], [0,1,4,5], [0,2,4,6]]
    for a_q in range(num_ancilla):
        h(ancilla_qubits[a_q])
        for d_q in sequences[a_q]:
            cx(ancilla_qubits[a_q], data_qubits[d_q])
        h(ancilla_qubits[a_q])
        measure(ancilla_qubits[a_q])

    return ancilla_qubits + data_qubits


if __name__ == "__main__":
    qc = steane_qrisp()[0].qs.to_qiskit()
    print(qc)
