"""Circuit ingestion module.

This module provides a unified interface for ingesting quantum circuits from
various frameworks (Qiskit, PyTKET, Qrisp) using the qBraid transpilation engine.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

from pathlib import Path
from typing import Any

import qbraid
from qbraid.programs import QPROGRAM
from qbraid.visualization import circuit_drawer
from qiskit import qpy


###########################
# CIRCUIT SOURCE MANAGER  #
###########################
class CircuitManager:
    """Manage and automatically save circuits as QASM artifacts."""

    def __init__(self, primary_key: str = "input"):
        """Initialise class with dual Augmented qBraid and QASM circuit collections."""
        self.primary_key = primary_key
        self._collection: dict[str, AugmentedQBCircuit] = {}
        # We keep this for quick string-based access if needed elsewhere
        self._qasm_collection: dict[str, str] = {}

    def _process(self, circuit_obj: QPROGRAM, key: str, is_primary: bool) -> str:
        """Wrap, store, and delegate artifact creation."""
        aug_qb = AugmentedQBCircuit(circuit_obj)
        self._collection[key] = aug_qb

        # Mirror the QASM string for registry access
        self._qasm_collection[key] = aug_qb.qasm

        if is_primary:
            self.primary_key = key

        return aug_qb.qasm

    def add_qiskit_circuit(self, circuit: Any, key: str = "input", is_primary: bool = True) -> str:
        """Add Qiskit circuit, save .qasm file, and return the qasm_str."""
        return self._process(circuit, key, is_primary)

    def add_pytket_circuit(self, circuit: Any, key: str = "input", is_primary: bool = True) -> str:
        """Add PyTKET circuit, save .qasm file, and return the qasm_str."""
        return self._process(circuit, key, is_primary)

    def add_qrisp_circuit(self, circuit: Any, key: str = "input", is_primary: bool = True) -> str:
        """Add Qrisp circuit, save .qasm file, and return the qasm_str."""
        return self._process(circuit, key, is_primary)

    def add_custom_circuit(
        self, circuit: QPROGRAM, key: str = "input", is_primary: bool = True
    ) -> str:
        """Add any qBraid-supported circuit (Qiskit, pytket, Qrisp, QASM)."""
        return self._process(circuit, key, is_primary)

    def add_circuit_from_file(
        self, path: str | Path, key: str = "input", is_primary: bool = True
    ) -> str:
        """Load from file, save .qasm artifact, and return the qasm_str."""

        # Ensure path is Path to facilitate extension management
        path_obj = Path(path)

        # Handle qiskit files (.qpy) explicitly
        if path_obj.suffix.lower() == ".qpy":
            with open(path_obj, "rb") as f:
                loaded_circuit = qpy.load(f)[0]  # Extract the first circuit from the QPY list

        # Handle QASM files explicitly
        elif path_obj.suffix.lower() == ".qasm":
            # Load_program returns an OpenQasm2Program object
            qasm_content = path_obj.read_text()
            wrapped_program = qbraid.load_program(qasm_content)

            # Extract the raw string from the wrapper to satisfy the transpiler
            loaded_circuit = wrapped_program.program

        # For .qasm, .json (Cirq), etc., use the standard qBraid loader
        else:
            loaded_circuit = qbraid.load_program(str(path_obj))

        return self._process(loaded_circuit, key, is_primary)

    def draw(self, key: str | None = None) -> Any:
        """Delegate drawing to the specific AugmentedQBCircuit object."""
        target_key = key if key else self.primary_key
        if target_key not in self._collection:
            raise KeyError(f"Circuit key '{target_key}' not found.")

        # Use the object's internal draw method
        return self._collection[target_key].draw()


###########################
# AUGMENTED QB CIRCUIT    #
###########################
class AugmentedQBCircuit:
    """Normalize quantum circuits into QASM using qBraid."""

    def __init__(self, source_circuit: QPROGRAM):
        """Initialise and immediately generate the normalized QASM truth."""
        self.source = source_circuit
        self._qasm: str = qbraid.transpile(self.source, "qasm2")

    @property
    def qasm(self) -> str:
        """Access the cached QASM string."""
        return self._qasm

    def draw(self, output: str = "text") -> Any:
        """Render the circuit from the normalized QASM string."""
        try:
            # We draw from the string to use qBraid's internal QASM renderer
            # fold=-1 ensures a single horizontal layer for ASCII smoke tests.
            vis = circuit_drawer(self._qasm, output=output, fold=-1)
            return vis
        except Exception as e:
            print(f"Drawing failed for {output}: {e}")
            print("\n--- Raw QASM Output ---")
            print(self._qasm)

    def get_native_type(self) -> str:
        """Return the original framework type name."""
        return f"{type(self.source).__module__}.{type(self.source).__name__}"
