"""UX manager.

The central nervous system for Topologiq UX handling data (controller) and orchestration (manager).

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

from topologiq.core.graph_manager.graph_manager import BlockGraphManager
from topologiq.input.circuit_manager import CircuitManager
from topologiq.input.zx_manager import AugmentedZXGraph, ZXGraphManager


class UXManager(QObject):
    """Data flow manager and top-level UX actions orchestrator."""

    # UI state signals
    status_changed = Signal(str)
    processing_state_changed = Signal(bool)
    section_changed = Signal(str)

    # Global data signals
    qb_circuit_ready = Signal(str, str)  # Raw QASM, ASCII Diagram
    zx_input_ready = Signal(object)  # Carries AugmentedZXGraph

    # Compilation result signals
    zx_staged_ready = Signal(str, object)
    blockgraph_ready = Signal(object)
    zx_output_ready = Signal(str)  # Carries graph key

    # Verification signals
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
        """Initialise data store."""
        return {
            "circuit_raw": "",
            "augmented_zx_graph_in": {},  # {key: AugZX}
            "lattice_surgery": {},  # {key: blockgraph_manager}
            "compiled_figures": {},  # {key: matplotlib.figure.Figure} -> NEW FIGURE LEDGER
            "augmented_zx_graph_out": {},  # {key: AugZX}
            "graphs_match": {},  # {key: bool}
        }

    @property
    def is_processing(self) -> bool:
        """Communicate if there are processes running."""
        return self._process_count > 0

    def _set_processing(self, active: bool, message: str):
        """Register process to process count."""
        if active:
            self._process_count += 1
        else:
            self._process_count = max(0, self._process_count - 1)
        self.processing_state_changed.emit(self.is_processing)
        self.status_changed.emit(message)

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

    def emergency_stop(self):
        """Abort processes."""
        if self._active_proc and self._active_proc.poll() is None:
            self._active_proc.kill()
            self._active_proc = None
            self.status_changed.emit("PROCESS TERMINATED")

    async def handle_load_source_circuit(
        self,
        source_circuit: str,
        mode: str,
        var_name: str = "circuit",
        draw_only: bool = True,
    ):
        """Ingest raw circuit string from a file path or user-facing script editor."""
        # Prevent run if anything is processing
        if self.is_processing and not draw_only:
            return

        # Clear store and session
        self.clear_session()
        self._set_processing(True, f"Executing {mode.upper()} source...")

        # Placeholder attributes
        aug_zx_to_emit = None
        is_native_pyzx = False

        try:
            # Enforce strict alphanumeric sequencing for initial imports
            base_key = f"{var_name}_0" if not var_name.endswith("_0") else var_name

            # Cache raw script reference
            self._data_store["circuit_raw"] = source_circuit

            # Process using strategy corresponding to IDE contents
            if mode == "python":
                # Execute user-submitted script using a restricted dictionary context namespace
                context = {"__name__": "__main__", "zx": zx, "qbraid": qbraid}

                def _execute():
                    exec(source_circuit, context)  # noqa: S102
                    return context.get(var_name)

                # Offload dynamic evaluation to thread pool to prevent GUI freeze
                target = await asyncio.to_thread(_execute)
                if target is None:
                    raise LookupError(f"Variable '{var_name}' not found in the script.")

                # Branch handling if target is an instantiated native PyZX graph
                if isinstance(target, zx.graph.base.BaseGraph):
                    is_native_pyzx = True
                    self.status_changed.emit("Integrating live PyZX graph...")

                    # Register within the internal incoming sub-manager registry
                    aug_zx_to_emit = self.zx_manager_in.add_graph_from_pyzx(
                        target, graph_key=base_key
                    )

                    # Quick rendering bypass short-circuit if user only requested an interactive layout draw pass
                    if draw_only:
                        self.qb_circuit_ready.emit(
                            zx.to_qasm(target) if hasattr(zx, "to_qasm") else "// PyZX Source",
                            "Graph is a native PyZX object. Ready to stage directly to interactive canvas.",
                        )
                        self.status_changed.emit(f"Parsed PyZX Graph '{base_key}'. Ready to stage.")
                        return

                    try:
                        self.qb_circuit_ready.emit(zx.to_qasm(target), "[Native PyZX Graph]")
                    except Exception:
                        self.qb_circuit_ready.emit(
                            "// Topology-only graph", "[Native PyZX Graph Layout]"
                        )

                else:
                    # Ingest path for non-PyZX objects (e.g. Qiskit, Cirq, Amazon Braket, OpenQASM)
                    # via qBraid
                    qasm_str = qbraid.transpiler.transpile(target, "qasm2")
                    self.circuit_manager.add_custom_circuit(qasm_str)

            else:
                # Direct string injection for native OpenQASM
                self.circuit_manager.add_custom_circuit(source_circuit)

            # Abstract syntax tree extraction block for qBraid representations
            if not is_native_pyzx:
                aug_qb = self.circuit_manager._collection[self.circuit_manager.primary_key]
                self._data_store["augmented_qb_circuit"] = aug_qb
                self.qb_circuit_ready.emit(str(aug_qb.qasm), str(aug_qb.draw()))

                if draw_only:
                    self.status_changed.emit(
                        f"Generated text visualisation for '{base_key}'. Ready to stage."
                    )
                    return

                # Convert OpenQASM -> ZX graph off-thread
                aug_zx_to_emit = await asyncio.to_thread(
                    self.zx_manager_in.add_graph_from_qasm,
                    qasm_str=aug_qb.qasm,
                    graph_key=base_key,
                )

            # Cascade and auto-stage valid generated data payloads
            if aug_zx_to_emit:
                self.handle_stage_to_compile(base_key, aug_zx_to_emit)
                self.zx_input_ready.emit(aug_zx_to_emit)
                self.status_changed.emit(
                    f"Staged '{base_key}' natively into compilation pipelines."
                )
            else:
                self.status_changed.emit(
                    "Staging aborted: Code executed, but no downstream ZX representation was generated."
                )

        except Exception as e:
            self.status_changed.emit(f"Execution Error: {e!s}")
        finally:
            self._set_processing(False, "Ready")

    async def handle_load_json_graph(self, json_str: str, graph_key: str):
        """Ingest and instantiate a native PyZX graph blueprint directly from JSON text."""
        # Prevent run if anything is processing
        if self.is_processing:
            return

        # Clear store and session
        self.clear_session()

        # Ensure naming consistency with the underscore suffix rule
        base_key = f"{graph_key}_0" if not graph_key.endswith("_0") else graph_key
        self._set_processing(True, f"Parsing JSON Topology for '{base_key}'...")

        try:

            def _parse():
                g = zx.Graph.from_json(json_str)
                return AugmentedZXGraph(zx_graph=g)

            aug_zx_to_emit = await asyncio.to_thread(_parse)

            # Force populate the main in-memory sub-manager data repository maps
            self.zx_manager_in.add_graph(aug_zx_to_emit, graph_key=base_key)

            self.qb_circuit_ready.emit(
                "// PyZX JSON Specification Document Data Ingested",
                "[Topology Ingestion Success]\nGraph bypasses qBraid and matches PyZX specifications natively.",
            )

            # Stage layout variables to active data dict trees
            self.handle_stage_to_compile(base_key, aug_zx_to_emit)

            # Inform active view panels of BOTH the graph object and target key reference
            # so ZXCanvas knows exactly what tab entry text to build and focus
            if hasattr(self, "zx_staged_ready"):
                self.zx_staged_ready.emit(base_key, aug_zx_to_emit)

            self.zx_input_ready.emit(aug_zx_to_emit)
            self.status_changed.emit(
                f"JSON Graph '{base_key}' auto-staged onto compilation profiles."
            )

        except Exception as e:
            self.status_changed.emit(f"JSON Compilation Failure: {e!s}")
        finally:
            self._set_processing(False, "Ready")

    async def handle_snapshot_zx_graph(self, pyzx_graph: zx.Graph, target_key: str):
        """Register a modified PyZX graph instance into collection while matching original vertex IDs."""
        # Prevent run if anything is processing
        if self.is_processing:
            return

        self._set_processing(True, f"Snapshotting active layout variant as {target_key}...")

        try:
            # Instantiate a clean target container
            sanitized_graph = zx.Graph()

            # High-water mark allocation: pre-generate vertices up to the maximum ID found.
            # This allows us to use the exact same integer indices natively.
            if pyzx_graph.num_vertices() > 0:
                max_id = max(pyzx_graph.vertices())
                # add_vertices(n) creates vertices with IDs 0 to n-1
                sanitized_graph.add_vertices(max_id + 1)

            # Keep a track of valid vertices to drop any unused pre-allocated indices later
            active_vertices = set(pyzx_graph.vertices())

            # Configure properties for the active vertices using their original IDs
            for v in active_vertices:
                r_val = pyzx_graph.row(v)
                q_val = pyzx_graph.qubit(v)
                rounded_row = int(r_val + 0.5) if r_val >= 0 else int(r_val - 0.5)
                rounded_qubit = int(q_val + 0.5) if q_val >= 0 else int(q_val - 0.5)

                # Overwrite the pre-allocated vertex configuration data
                sanitized_graph.set_type(v, pyzx_graph.type(v))
                sanitized_graph.set_qubit(v, rounded_qubit)
                sanitized_graph.set_row(v, rounded_row)
                sanitized_graph.set_phase(v, pyzx_graph.phase(v))

                switch_to_cube_val = pyzx_graph.vdata(v, "switch_to_cube", default=None)
                if switch_to_cube_val is not None:
                    sanitized_graph.set_vdata(v, "switch_to_cube", switch_to_cube_val)

            # Purge the unused intermediate gap vertices from our pre-allocation pass
            total_allocated = max_id + 1 if pyzx_graph.num_vertices() > 0 else 0
            all_allocated_ids = list(range(total_allocated))
            vertices_to_remove = [v for v in all_allocated_ids if v not in active_vertices]
            if vertices_to_remove:
                sanitized_graph.remove_vertices(vertices_to_remove)

            # Bind connectivity maps using the preserved original IDs
            for edge in pyzx_graph.edges():
                u, v = edge[0], edge[1]
                e_type = pyzx_graph.edge_type(edge)
                sanitized_graph.add_edge((u, v), edgetype=e_type)

            # Map inputs and outputs using the preserved original IDs
            if hasattr(pyzx_graph, "inputs"):
                sanitized_graph.set_inputs([i for i in pyzx_graph.inputs() if i in active_vertices])
            if hasattr(pyzx_graph, "outputs"):
                sanitized_graph.set_outputs(
                    [o for o in pyzx_graph.outputs() if o in active_vertices]
                )

            # Encapsulate into an internal wrapper
            aug_zx_to_emit = AugmentedZXGraph(zx_graph=sanitized_graph)
            self.zx_manager_in.add_graph(aug_zx_to_emit, graph_key=target_key)

            self.qb_circuit_ready.emit(
                zx.to_qasm(sanitized_graph)
                if hasattr(zx, "to_qasm")
                else "// Snapshot Branch State",
                f"[Snapshot Success]\nCreated normalised, compilable variant checkpoint: {target_key}.",
            )

            # Auto-stage snapshot variations directly to compile targets
            self.handle_stage_to_compile(target_key, aug_zx_to_emit)
            self.zx_input_ready.emit(aug_zx_to_emit)
            self.status_changed.emit(f"Snapshot variant '{target_key}' auto-staged.")

        except Exception as e:
            self.status_changed.emit(f"Snapshot Branching Failure: {e!s}")
        finally:
            self._set_processing(False, "Ready")

    def handle_stage_to_compile(self, graph_key: str, aug_zx_in: AugmentedZXGraph):
        """Stage a target graph layout and initialise pre-surgery."""
        # Ensure state trees exist
        if "augmented_zx_graph_in" not in self._data_store:
            self._data_store["augmented_zx_graph_in"] = {}
        self._data_store["augmented_zx_graph_in"][graph_key] = aug_zx_in

        if "lattice_surgery" not in self._data_store:
            self._data_store["lattice_surgery"] = {}

        # Bind isolated block graph management context shell to track surgery
        if hasattr(aug_zx_in, "zx_graph"):
            bgraph_manager = BlockGraphManager(aug_zx_in)
            self._data_store["lattice_surgery"][graph_key] = bgraph_manager

        self.zx_staged_ready.emit(graph_key, aug_zx_in)

    async def handle_compile(
        self, graph_key: str, options: dict | None = None, write_bgraph: bool = False
    ):
        """Reset to baseline and execute compilation."""
        # Fetch target blueprint definition metrics
        aug_zx_in = self._data_store["augmented_zx_graph_in"].get(graph_key)
        if not aug_zx_in or not hasattr(aug_zx_in, "zx_graph"):
            self.status_changed.emit(
                f"Compilation Aborted: Baseline layout data missing for '{graph_key}'."
            )
            return

        # Isolate target context within a thread-safe instance
        local_bgraph_manager = BlockGraphManager(aug_zx_in)
        override_kwargs = options if options is not None else {}
        local_session = self._session_id

        self._set_processing(True, f"Compiling layout for {graph_key}...")
        try:
            # Execute compilation in thread to prevent GUI lockups
            await asyncio.to_thread(local_bgraph_manager.build, override_kwargs=override_kwargs)

            # Drop out if session updates occurred during execution
            if local_session != self._session_id:
                return

            # Transaction success so it is safe to commit to the global ledger
            self._data_store["lattice_surgery"][graph_key] = local_bgraph_manager

            # Explicitly generate and store static MPL view
            visualiser = local_bgraph_manager.draw_blockgraph(is_final_vis=True, embedded=True)
            if visualiser and hasattr(visualiser, "view_3d") and hasattr(visualiser.view_3d, "fig"):
                self._data_store["compiled_figures"][graph_key] = visualiser.view_3d.fig

            # Instant UX refresh
            self.blockgraph_ready.emit(self._data_store["lattice_surgery"])
            self.zx_output_ready.emit(graph_key)
            self.status_changed.emit(
                f"Compilation complete for '{graph_key}'. Rendering visualisation..."
            )

            # Post-process file writes with session guard tracking
            if write_bgraph:
                if local_session != self._session_id:
                    return
                self.status_changed.emit(
                    f"Writing BGRAPH structural topology files for '{graph_key}'..."
                )
                await asyncio.to_thread(local_bgraph_manager.write_bgraph, circuit_name=graph_key)

            # Process animation matrices with session guard tracking
            if override_kwargs.get("animate"):
                if local_session != self._session_id:
                    return
                self.status_changed.emit(
                    f"Compiling animation frames for '{graph_key}' in background..."
                )
                await asyncio.to_thread(local_bgraph_manager.animate, filename_prefix=graph_key)

                if local_session == self._session_id:
                    self.status_changed.emit(f"Animation rendering complete for '{graph_key}'.")

        except Exception as e:
            self.status_changed.emit(f"Surgery Engine Error [{graph_key}]: {e}")
        finally:
            # Secure processing toggles only if session thread scopes match
            if local_session == self._session_id:
                self._set_processing(False, "Ready")

    def get_data(self, key: str) -> Any:
        """Retrieve from data store."""
        return self._data_store.get(key)
