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
import os
import subprocess
import sys
import tempfile
from typing import Any

from PySide6.QtCore import QObject, Signal

from topologiq.input.pyzx_manager import ZXGraphManager
from topologiq.input.qbraid_manager import AugmentedQBCircuit, CircuitManager
from topologiq.ux.utils import GLUE_CODE


class UXManager(QObject):
    """Data (Controller) and UI Orchestration (Manager) class."""

    # SIGNALS: notify PySide6 UI when to update
    status_changed = Signal(str)  # For the status bar / progress messages
    processing_state_changed = Signal(bool)  # Disable buttons during heavy lifting
    section_changed = Signal(str)  # Switch tabs

    # Data signals for the visualizers
    qb_circuit_ready = Signal(str, str)  # Raw QASM update and ASCII diagram
    zx_input_ready = Signal(object)  # ZX visual data (matplotlib figure)
    blockgraph_ready = Signal(object)  # TBD
    zx_output_ready = Signal(dict)  # Visual data for Vedo
    ready_for_equality_verification = Signal(bool)  # Green light for equivalence checks
    equality_verification = Signal(bool)  # Green light for equivalence checks

    def __init__(self):
        """Initialise with QB and ZX circuit managers and data store."""
        super().__init__()

        # Internal Controllers
        self.circuit_manager = CircuitManager()
        self.zx_manager = ZXGraphManager()

        # Internal Data Store
        self._data_store: dict[str, Any] = {
            "circuit_raw": "",
            "augmented_qb_circuit": None,
            "augmented_zx_graph_in": None,
            "lattice_surgery": (),
            "augmented_zx_graph_out": None,
            "graphs_match": False,
        }

        self.is_processing = False

    # METHODS: interface with Topologiq
    async def handle_load_source_circuit(
        self,
        source_design: str,
        mode: str,
        var_name: str = "circuit",  # Defaulting to your test variable
        switch_to_transform: bool = False,
    ):
        """Ingest code and convert into an augmented qBraid circuit via Subprocess."""

        if self.is_processing:
            return

        self._set_processing(True, f"Ingesting {mode.upper()} source...")

        try:
            self._data_store["circuit_raw"] = source_design

            if mode == "python":
                # 1. Create a secure temporary script
                with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
                    tmp_path = tmp.name

                    # 2. Prepare the user's code with 4-space indentation
                    indented_source = "\n".join(
                        f"    {line}" for line in source_design.splitlines()
                    )

                    # 3. Format the Glue Code (assuming GLUE_CODE is defined globally/imported)
                    final_script = GLUE_CODE.format(
                        source_design=indented_source, var_name=var_name
                    )
                    tmp.write(final_script)

                try:
                    # 4. Run the script in a separate OS process
                    # We use asyncio.to_thread to keep the UI responsive during the wait
                    proc = await asyncio.to_thread(
                        subprocess.run,
                        [sys.executable, tmp_path],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )

                    # --- DEBUG BRIDGE: Essential for finding "Silent" failures ---
                    print(f"\n--- SUBPROCESS STDOUT ---\n{proc.stdout}")
                    print(f"--- SUBPROCESS STDERR ---\n{proc.stderr}")

                    # 2. Check for System/OS level crashes first
                    if proc.returncode != 0:
                        # Extract the actual Python error from stderr
                        error_msg = proc.stderr.strip().split('\n')[-1]
                        raise RuntimeError(f"Python Runtime Error: {error_msg}")

                    # 5. Parse the isolated output for the QASM markers
                    if "---BEGIN_QASM---" in proc.stdout:
                        qasm_data = (
                            proc.stdout.split("---BEGIN_QASM---")[1]
                            .split("---END_QASM---")[0]
                            .strip()
                        )
                        self.circuit_manager.add_custom_circuit(qasm_data)
                    elif "ERROR:" in proc.stderr:
                        # This catches the ERROR print we put in the GLUE_CODE
                        specific_error = proc.stderr.split("ERROR:")[-1].strip()
                        raise ValueError(f"Circuit Extraction Failed: {specific_error}")
                    else:
                        raise LookupError(f"Variable '{var_name}' not found in your script.")

                except subprocess.TimeoutExpired:
                    raise RuntimeError("Script execution timed out (5s limit). Check for infinite loops!")

                finally:
                    # Cleanup the temporary file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            else:
                # QASM mode (Standard string ingestion)
                self.circuit_manager.add_custom_circuit(source_design)

            # --- COMMON PATH: Update Data Store and UI ---
            aug_qb = self.circuit_manager._collection[self.circuit_manager.primary_key]
            self._data_store["augmented_qb_circuit"] = aug_qb

            # Notify UI (Ensure explicit string conversion for PySide6)
            self.qb_circuit_ready.emit(str(aug_qb.qasm), str(aug_qb.draw()))

            if switch_to_transform:
                await self.handle_qb_to_zx_transform(aug_qb)
            else:
                self.section_changed.emit("DESIGN")
                self.status_changed.emit("Circuit loaded via Air-Gapped Python.")

        except subprocess.TimeoutExpired:
            self.status_changed.emit("ERROR: Script timed out (Infinite loop?)")
        except Exception as e:
            # Mirror the error to the terminal for easy copy-pasting
            print(f"\n[MANAGER ERROR]: {e!s}\n")
            self.status_changed.emit(f"ERROR: {e!s}")

        finally:
            self.is_processing = False
            self.processing_state_changed.emit(False)

    async def handle_qb_to_zx_transform(self, aug_qb_circuit: AugmentedQBCircuit):
        """Ingest an augmented qBraid circuit and prepare the initial augmented ZX Graph."""

        # Do NOT do anything if something else is happening
        if self.is_processing:
            return

        # Turn processing ON
        self._set_processing(True, "Transpiling: qBraid -> PyZX...")

        # QB -> ZX
        try:
            # Convert and add to data store
            aug_zx_in = self.zx_manager.add_graph_from_qasm(
                qasm_str=aug_qb_circuit.qasm, use_primary=True
            )
            self._data_store["augmented_zx_graph_in"] = aug_zx_in

            # 3. Notify UI
            self.zx_input_ready.emit(aug_zx_in.get_native_visualisation())
            self.section_changed.emit("TRANSFORM")
            self.status_changed.emit("qBraid -> PyZX transpilation complete.")

        except Exception as e:
            self.status_changed.emit(f"Error: {e!s}")

        finally:
            self.is_processing = False
            self.processing_state_changed.emit(False)

    async def handle_lattice_surgery(self):
        """Execute the heavy 3D transformation logic."""

        # Do NOT do anything if something else is happening
        if self.is_processing:
            return

        # Turn processing ON
        self._set_processing(True, "Performing Lattice Surgery...")

        try:
            # Simulate the surgery logic from your Controller
            cubes, pipes = await asyncio.to_thread(
                self.zx_manager.perform_lattice_surgery, use_primary=True
            )
            self._data_store["lattice_surgery"] = (cubes, pipes)

            # Distill output and check match
            zx_in = self.zx_manager.get_graph(use_primary=True)
            aug_zx_out = await asyncio.to_thread(
                self.zx_manager.add_graph_from_blockgraph, cubes, pipes, other=zx_in
            )
            self._data_store["augmented_zx_graph_out"] = aug_zx_out

            # Notify UI
            zx_out_visual_payload = {
                "full": aug_zx_out.get_visual_data(use_reduced=False),
                "reduced": aug_zx_out.get_visual_data(use_reduced=True),
            }
            self.blockgraph_ready.emit(zx_out_visual_payload)
            self.zx_output_ready.emit(zx_out_visual_payload)
            self.ready_for_equality_verification.emit(True)
            self.status_changed.emit("Lattice surgery complete.")

        except Exception as e:
            self.status_changed.emit(f"ERROR. Lattice surgery did not complete: {e!s}")

        finally:
            self.is_processing = False
            self.processing_state_changed.emit(False)

    async def handle_equality_verification(
        self, graph_key_in: str = "input", graph_key_out: str = "output"
    ):
        """Execute the heavy 3D transformation logic."""

        # Change processing status without blocking other processing
        self.status_changed.emit("Verifying equality in background...")

        try:
            # Get input and output ZX as NX graphs
            aug_zx_in = self.zx_manager.get_graph(graph_key=graph_key_in)
            aug_zx_out = self.zx_manager.get_graph(graph_key=graph_key_out)

            # Check equivalence
            graphs_match = await asyncio.to_thread(aug_zx_in.check_equality, aug_zx_out)
            self._data_store["graphs_match"] = graphs_match

            # Notify UI
            self.equality_verification.emit(graphs_match)
            self.status_changed.emit("Equivalence check complete.")

        except Exception as e:
            self.status_changed.emit(f"ERROR. Equivalence checks errored out: {e!s}")

    # HELPERS
    def _set_processing(self, active: bool, message: str):
        """Set processing state and notify UI."""
        self.is_processing = active
        self.processing_state_changed.emit(active)
        self.status_changed.emit(message)

    def get_data(self, key: str) -> Any:
        """Retrieve from data store."""
        return self._data_store.get(key)
