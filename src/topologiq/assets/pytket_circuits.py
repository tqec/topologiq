"""pytket circuits to use in examples, demonstrations and testing."""

from pytket import Circuit


def cnots_pytket(circuit_name):
    """Return a pytket circuit composed of a number of CNOTs."""

    circ = Circuit(2, name=f"{circuit_name}_pytket")
    circ.X(0)
    circ.CX(0, 1)
    circ.Z(1)

    return circ


circuit = cnots_pytket("cnot")

