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

from topologiq.input.pyzx_manager import AugmentedZXGraph, ZXGraphManager
from topologiq.input.qbraid_manager import CircuitManager


class UXManager(QObject):
    """Data (Controller) and UI Orchestration (Manager) class."""

    # UI State Signals
    status_changed = Signal(str)
    processing_state_changed = Signal(bool)
    section_changed = Signal(str)

    # Global Data Signals
    qb_circuit_ready = Signal(str, str)  # Raw QASM, ASCII Diagram
    zx_input_ready = Signal(object)  # Carries AugmentedZXGraph

    # Compilation Result Signals
    blockgraph_ready = Signal(str)  # Carries graph key
    zx_output_ready = Signal(str)  # Carries graph key

    # Verification Signals
    verification_ready = Signal(str, bool)  # Carries graph key and result

    def __init__(self):
        """Initialise UX manager."""
        super().__init__()

        # Init circuit and ZX graph managers
        self.circuit_manager = CircuitManager()
        self.zx_manager_in = ZXGraphManager()
        self.zx_manager_out = ZXGraphManager()

        # Init data store
        self._data_store = self._init_store()

        # Init process tracker
        self._background_tasks = set()
        self._session_id = 0
        self._active_proc: subprocess.Popen | None = None
        self._process_count = 0

    def _init_store(self):
        """Standardized empty store structure."""
        return {
            "augmented_zx_graph_in": {},  # {key: AugZX}
            "lattice_surgery": {},  # {key: (cubes, pipes)}
            "augmented_zx_graph_out": {},  # {key: AugZX}
            "graphs_match": {},  # {key: bool}
            "circuit_raw": "",
        }

    def clear_session(self):
        """Reset sub-managers and data store."""

        # Re-init circuit and ZX graph managers
        self.circuit_manager = CircuitManager()
        self.zx_manager_in = ZXGraphManager()
        self.zx_manager_out = ZXGraphManager()

        # Re-init data store
        self._data_store = self._init_store()

        # Re-init session
        self._session_id += 1
        self._active_proc = None
        self._process_count = 0

        # Update UX message
        self.status_changed.emit(f"New input => new session (ID: {self._session_id})")

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
        print(f"DEBUG: Load Request Received. Processing Count: {self._process_count}")
        if self.is_processing and not switch_to_transform:
            print("DEBUG: LOAD BLOCKED - Manager is busy.")
            return

        self.clear_session()
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
                if isinstance(target, zx.graph.base.BaseGraph):
                    is_native_pyzx = True
                    self.status_changed.emit("Integrating live PyZX graph...")

                    # Pass the LIVE object. It retains all metadata/NetworkX pointers.
                    aug_zx_to_emit = self.zx_manager_in.add_graph_from_pyzx(target, graph_key=var_name)

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

                # We ALWAYS generate the ZX graph now so we can run surgery
                aug_zx_to_emit = await asyncio.to_thread(
                    self.zx_manager_in.add_graph_from_qasm, qasm_str=aug_qb.qasm, graph_key=var_name
                )

            # --- 6. ATOMIC UI SWITCH ---
            # We only transition to the Canvas if all steps above succeeded
            if aug_zx_to_emit:
                # 1. Update the Keyed Store
                self._data_store["augmented_zx_graph_in"][var_name] = aug_zx_to_emit

                # 2. Trigger Surgery with GC Protection
                task = asyncio.create_task(self.handle_silent_surgery(var_name, aug_zx_to_emit))

                # Store the reference
                self._background_tasks.add(task)

                # Ensure the task cleans itself up from the set when done
                task.add_done_callback(self._background_tasks.discard)

                # 3. Handle UI Transition
                if switch_to_transform:
                    self.section_changed.emit("DESIGN")
                    await asyncio.sleep(0.1)
                    self.zx_input_ready.emit(aug_zx_to_emit)
                    self.status_changed.emit(f"Ingested '{var_name}' and moved to Design.")
                else:
                    self.status_changed.emit(
                        f"Ingested '{var_name}'. Surgery running in background."
                    )
            else:
                self.status_changed.emit("Code executed, but no ZX graph was produced.")

        except Exception as e:
            # Catch logic errors, syntax errors, or type errors
            self.status_changed.emit(f"Execution Error: {e!s}")
        finally:
            self._set_processing(False, "Ready")

    async def handle_silent_surgery(self, graph_key: str, aug_zx_in: AugmentedZXGraph):
        """Handle surgery -> transpile -> verify pipeline in silence."""

        # Retrieve session ID and turn processing on
        local_session = self._session_id
        self._set_processing(True, f"Compiling {graph_key}...")

        try:
            # 1. Store Input
            self._data_store["augmented_zx_graph_in"][graph_key] = aug_zx_in

            # 2. Surgery (Full version only)
            cubes, pipes = await asyncio.to_thread(aug_zx_in.get_blockgraph)

            # SESSION GUARD: Check if we are still in the same session before writing
            if local_session != self._session_id:
                return

            self._data_store["lattice_surgery"][graph_key] = (cubes, pipes)
            self.blockgraph_ready.emit(graph_key)

            # 3. Reverse Transpilation (Output Graph)
            aug_zx_out = await asyncio.to_thread(
                self.zx_manager_out.add_graph_from_blockgraph,
                blockgraph_cubes=cubes,
                blockgraph_pipes=pipes,
                graph_key=f"{graph_key}",
                other=aug_zx_in,
            )

            if local_session != self._session_id:
                return
            self._data_store["augmented_zx_graph_out"][graph_key] = aug_zx_out
            self.zx_output_ready.emit(graph_key)

            # 4. Equality Check
            print(f"DEBUG: about to undertake equality check of {aug_zx_in.zx_graph_reduced} and {aug_zx_out.zx_graph_reduced}")
            match = await asyncio.to_thread(aug_zx_in.check_equality, aug_zx_out)
            self._data_store["graphs_match"][graph_key] = match
            print(f"DEBUG: match emitted to store is: {graph_key} , {match}")
            self.verification_ready.emit(graph_key, match)

        except Exception as e:
            self.status_changed.emit(f"Pipeline Error [{graph_key}]: {e}")
        finally:
            self._set_processing(False, "Ready")

    def emergency_stop(self):  # noqa: D102
        if self._active_proc and self._active_proc.poll() is None:
            self._active_proc.kill()
            self._active_proc = None
            self.status_changed.emit("PROCESS TERMINATED")

    def get_data(self, key: str) -> Any:  # noqa: D102
        return self._data_store.get(key)
