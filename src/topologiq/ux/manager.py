"""UX manager.

The Central Nervous System for Topologiq UX. Handles Data (Controller) and UI Orchestration (Manager).

NB! Methods use asyncio because the goal is ultimately to stream data live.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

import asyncio
import subprocess
from typing import Any

import pyzx as zx
import qbraid
from PySide6.QtCore import QObject, Signal

from topologiq.input.pyzx_manager import ZXGraphManager
from topologiq.input.qbraid_manager import CircuitManager


class UXManager(QObject):
    """Data (Controller) and UI Orchestration (Manager) class."""

    # UI State Signals
    status_changed = Signal(str)
    processing_state_changed = Signal(bool)
    section_changed = Signal(str)

    # Global Data Signals
    qb_circuit_ready = Signal(str, str)  # Raw QASM, ASCII Diagram
    zx_input_ready = Signal(object)  # Now sends the NX graph for the Global Drawer

    # Compilation Result Signals
    blockgraph_ready = Signal(object, object)  # Cubes, Pipes (dict format for BGraphCanvas)
    zx_output_ready = Signal(object)  # The NX graph derived from the blockgraph

    # Verification Signals
    ready_for_equality_verification = Signal(bool)
    equality_verification = Signal(bool)

    def __init__(self):  # noqa: D107
        super().__init__()
        self.circuit_manager = CircuitManager()
        self.zx_manager = ZXGraphManager()
        self._active_proc: subprocess.Popen | None = None
        self._process_count = 0

        self._data_store: dict[str, Any] = {
            "circuit_raw": "",
            "augmented_qb_circuit": None,
            "augmented_zx_graph_in": None,
            "lattice_surgery": (),
            "augmented_zx_graph_out": None,
            "graphs_match": False,
        }

    @property
    def is_processing(self) -> bool:  # noqa: D102
        return self._process_count > 0

    def _set_processing(self, active: bool, message: str):
        """Standardised reference counting for UI locking."""
        if active:
            self._process_count += 1
        else:
            self._process_count = max(0, self._process_count - 1)

        self.processing_state_changed.emit(self.is_processing)
        self.status_changed.emit(message)

    async def handle_load_source_circuit(
        self,
        source_design: str,
        mode: str,
        var_name: str = "circuit",
        switch_to_transform: bool = False,
    ):
        """In-process ingestion using exec for rapid MVP development."""
        if self.is_processing and not switch_to_transform:
            return

        self._set_processing(True, f"Executing {mode.upper()} source...")

        # Local state for this ingestion cycle
        aug_zx_to_emit = None
        is_native_pyzx = False

        try:
            self._data_store["circuit_raw"] = source_design

            if mode == "python":
                # 1. Prepare a fresh execution context
                # We pre-inject zx and qbraid so the user doesn't strictly
                # need to import them in the editor for the script to run.
                context = {"__name__": "__main__", "zx": zx, "qbraid": qbraid}

                # 2. Execute user code in a thread to keep the UI responsive
                def _execute():
                    # We use exec for multi-line support
                    exec(source_design, context)  # noqa: S102
                    return context.get(var_name)

                target = await asyncio.to_thread(_execute)

                if target is None:
                    raise LookupError(f"Variable '{var_name}' not found in the script.")

                # 3. PATH A: NATIVE PyZX (In-Memory)
                # We check the live object type directly
                if isinstance(target, zx.graph.base.BaseGraph) or hasattr(target, "to_json"):
                    is_native_pyzx = True
                    self.status_changed.emit("Integrating live PyZX graph...")

                    # Pass the LIVE object. It retains all metadata/NetworkX pointers.
                    aug_zx_to_emit = self.zx_manager.add_graph_from_pyzx(target, use_primary=True)
                    self._data_store["augmented_zx_graph_in"] = aug_zx_to_emit

                    # Sync Design Pane text area with a QASM representation
                    try:
                        self.qb_circuit_ready.emit(zx.to_qasm(target), "[Native PyZX Graph]")
                    except Exception:
                        self.qb_circuit_ready.emit("// Topology-only graph", "[Native PyZX Graph]")

                # 4. PATH B: qBraid Fallback
                else:
                    # Transpile the live object (Qiskit, Cirq, etc.) via qBraid
                    qasm_str = qbraid.transpiler.transpile(target, "qasm2")
                    self.circuit_manager.add_custom_circuit(qasm_str)

            else:
                # Direct QASM string ingestion (non-python mode)
                self.circuit_manager.add_custom_circuit(source_design)

            # --- 5. qBraid Path Finalization ---
            if not is_native_pyzx:
                aug_qb = self.circuit_manager._collection[self.circuit_manager.primary_key]
                self._data_store["augmented_qb_circuit"] = aug_qb
                self.qb_circuit_ready.emit(str(aug_qb.qasm), str(aug_qb.draw()))

                if switch_to_transform:
                    # Transpile our internal qBraid circuit to ZX
                    aug_zx_to_emit = await asyncio.to_thread(
                        self.zx_manager.add_graph_from_qasm, qasm_str=aug_qb.qasm, use_primary=True
                    )
                    self._data_store["augmented_zx_graph_in"] = aug_zx_to_emit

            # --- 6. ATOMIC UI SWITCH ---
            # We only transition to the Canvas if all steps above succeeded
            if switch_to_transform and aug_zx_to_emit:
                self.section_changed.emit("DESIGN")  # Ensure the correct tab is active
                await asyncio.sleep(0.1)  # UI stability buffer
                self.zx_input_ready.emit(aug_zx_to_emit)
                self.status_changed.emit("Graph ingested and visualized successfully.")
            else:
                self.status_changed.emit("Code executed successfully.")

        except Exception as e:
            # Catch logic errors, syntax errors, or type errors
            self.status_changed.emit(f"Execution Error: {e!s}")
        finally:
            self._set_processing(False, "Ready")

    async def handle_lattice_surgery(self, use_reduced: bool = False):
        """Execute 3D Lattice Surgery and generate verification graph."""
        if self.is_processing:
            return

        # Pre-flight check: look for the graph in the data store
        aug_zx_in = self._data_store.get("augmented_zx_graph_in")

        if not aug_zx_in:
            # Emit error and stay in DESIGN tab
            self.status_changed.emit("LATTICE SURGERY ERROR: No input ZX graph found in store.")
            return

        # If we have a graph, start processing and switch pane
        self._set_processing(True, "Performing Lattice Surgery...")
        self.section_changed.emit("COMPILE")

        try:
            # 1. Surgery (Nodes/Edges for Blockgraph)
            cubes, pipes = await asyncio.to_thread(
                aug_zx_in.get_blockgraph, use_reduced=use_reduced
            )
            self._data_store["lattice_surgery"] = (cubes, pipes)
            self.blockgraph_ready.emit(cubes, pipes)

            # 2. Reverse Transpilation (Blocks -> ZX for verification)
            aug_zx_out = await asyncio.to_thread(
                self.zx_manager.add_graph_from_blockgraph,
                blockgraph_cubes=cubes,
                blockgraph_pipes=pipes,
                graph_key="output",
                other=aug_zx_in,
            )
            self._data_store["augmented_zx_graph_out"] = aug_zx_out

            # Emit the NetworkX graph for the secondary visualizer in COMPILE
            self.zx_output_ready.emit(aug_zx_out)

            self.ready_for_equality_verification.emit(True)
            self.status_changed.emit("Lattice surgery complete.")

        except Exception as e:
            self.status_changed.emit(f"Surgery Error: {e}")
        finally:
            self._set_processing(False, "Ready")

    async def handle_equality_verification(
        self, graph_key_in: str = "input", graph_key_out: str = "output"
    ):
        """Check if compiled blockgraph matches original design."""
        self.status_changed.emit("Verifying equality...")
        try:
            aug_zx_in = self.zx_manager.get_graph(graph_key=graph_key_in)
            aug_zx_out = self.zx_manager.get_graph(graph_key=graph_key_out)

            match = await asyncio.to_thread(aug_zx_in.check_equality, aug_zx_out)
            self._data_store["graphs_match"] = match
            self.equality_verification.emit(match)
            self.status_changed.emit("Verification complete.")
        except Exception as e:
            self.status_changed.emit(f"Verification Error: {e}")

    def emergency_stop(self):  # noqa: D102
        if self._active_proc and self._active_proc.poll() is None:
            self._active_proc.kill()
            self._active_proc = None
            self.status_changed.emit("PROCESS TERMINATED")

    def get_data(self, key: str) -> Any:  # noqa: D102
        return self._data_store.get(key)
