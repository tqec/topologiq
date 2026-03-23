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
from topologiq.ux.utils.glue_code import GLUE_CODE


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
    blockgraph_ready = Signal(dict, dict)  # Cubes, Pipes (dict format for BGraphCanvas)
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
        """Ingest code and convert into qBraid circuit via Subprocess sandbox."""
        if self.is_processing and not switch_to_transform:
            return

        if mode == "python" and not var_name:
            self.status_changed.emit("ERROR: No variable name provided.")
            return

        self._set_processing(True, f"Ingesting {mode.upper()} source...")

        try:
            self._data_store["circuit_raw"] = source_design

            if mode == "python":
                with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
                    tmp_path = tmp.name
                    indented_source = "\n".join(
                        f"    {line}" for line in source_design.splitlines()
                    )
                    final_script = GLUE_CODE.format(
                        source_design=indented_source, var_name=var_name
                    )
                    tmp.write(final_script)

                try:
                    self._active_proc = subprocess.Popen(  # noqa: S603
                        [sys.executable, tmp_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        shell=False,  # Explicitly disable shell to prevent command injection
                        start_new_session=True,  # Prevents CTRL+C in terminal from killing the GUI
                    )
                    # Offload the blocking communicate call
                    stdout, stderr = await asyncio.to_thread(
                        self._active_proc.communicate, timeout=5
                    )

                    if self._active_proc.returncode != 0:
                        raise RuntimeError(f"Python Error: {stderr.strip().splitlines()[-1]}")

                    if "---BEGIN_QASM---" in stdout:
                        qasm_data = (
                            stdout.split("---BEGIN_QASM---")[1].split("---END_QASM---")[0].strip()
                        )
                        self.circuit_manager.add_custom_circuit(qasm_data)
                    else:
                        raise LookupError(f"Variable '{var_name}' not found in script.")

                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            else:
                self.circuit_manager.add_custom_circuit(source_design)

            # Success Path
            aug_qb = self.circuit_manager._collection[self.circuit_manager.primary_key]
            self._data_store["augmented_qb_circuit"] = aug_qb
            self.qb_circuit_ready.emit(str(aug_qb.qasm), str(aug_qb.draw()))

            if switch_to_transform:
                await self.handle_qb_to_zx_transform(aug_qb)
            else:
                self.status_changed.emit("Circuit loaded.")

        except Exception as e:
            self.status_changed.emit(f"ERROR: {e}")
        finally:
            self._set_processing(False, "Ready")

    async def handle_qb_to_zx_transform(self, aug_qb_circuit: AugmentedQBCircuit):
        """Transpile qBraid circuit to PyZX and update global drawer."""
        self._set_processing(True, "Transpiling to ZX...")
        try:
            # Heavy transpilation in background thread
            aug_zx_in = await asyncio.to_thread(
                self.zx_manager.add_graph_from_qasm, qasm_str=aug_qb_circuit.qasm, use_primary=True
            )
            self._data_store["augmented_zx_graph_in"] = aug_zx_in

            self.section_changed.emit("DESIGN")
            await asyncio.sleep(0.1)  # Buffer for UI tab switch

            # Emit the NetworkX graph for the ZXCanvas
            self.zx_input_ready.emit(aug_zx_in)

        except Exception as e:
            self.status_changed.emit(f"Transpilation Error: {e}")
        finally:
            self._set_processing(False, "Ready")

    async def handle_lattice_surgery(self, use_reduced: bool = False):
        """Execute 3D Lattice Surgery and generate verification graph."""
        if self.is_processing:
            return
        self._set_processing(True, "Performing Lattice Surgery...")

        try:
            aug_zx_in = self._data_store.get("augmented_zx_graph_in")
            if not aug_zx_in:
                raise ValueError("No input ZX graph found.")

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
            self.zx_output_ready.emit(aug_zx_out.nx_graph)

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
