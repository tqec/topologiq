"""PyZX graph and PyZX graph manager classes.

This module provides a unified interface for ingesting PyZX circuits from
QASM as well as managing and producing PyZX graphs from them.

"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import networkx as nx
import pyzx as zx
from pyzx.circuit import Circuit
from pyzx.pauliweb import compute_pauli_webs

from topologiq.utils.zx import ZXColors, ZXEdgeTypes, ZXTypes


######################
# PyZX GRAPH MANAGER #
######################
class ZXGraphManager:
    """Registry class to keep AugmentedZXGraph(s) organised."""

    def __init__(self, primary_key: str = "primary", debug: int = 0):
        """Initialise class with incoming or default primary key and empty collection."""
        self._collection: dict[str, AugmentedZXGraph] = {}
        self.primary_key = primary_key
        self.debug = debug

    def get_graph(
        self,
        use_primary: bool = False,
        graph_key: str = "",
    ) -> AugmentedZXGraph:
        """Retrieve an AugmentedZXGraph from the collection.

        Args:
            use_primary: Flag to set key to primary key.
            graph_key: Open key string to save intermediate/modified ZX graphs.

        """
        key = self.primary_key if use_primary else graph_key
        if not key or key not in self._collection:
            raise ValueError(f"ERROR. Key {key} not in AugmentedZXGraph collection.")
        return self._collection[key]

    def add_graph(
        self,
        aug_zx_graph: AugmentedZXGraph,
        graph_key: str,
    ):
        """Add an AugmentedZXGraph to the collection.

        Args:
            aug_zx_graph: The AugmentedZXGraph to preserve.
            use_primary: Flag to set key to primary key.
            graph_key: String to use as collection key.
            debug: Debug mode to use.

        """
        if not graph_key:
            raise ValueError("ERROR. A key is needed to add an AugmentedZXGraph to collection.")
        self._collection[graph_key] = aug_zx_graph

    def add_graph_from_pyzx(
        self,
        zx_graph: zx.Graph,
        use_primary: bool = False,
        graph_key: str = "",
    ) -> AugmentedZXGraph:
        """Add an AugmentedZXGraph to the collection starting with a standard PyZX graph.

        Args:
            zx_graph: The PyZX graph.
            use_primary: Flag to set key to primary key.
            graph_key: Open key string to save intermediate/modified ZX graphs.
            debug: Debug mode to use.

        """
        key = self.primary_key if use_primary else graph_key
        self.add_graph(AugmentedZXGraph(zx_graph, debug=self.debug), graph_key=key)
        return self._collection[key]

    def add_graph_from_qasm(
        self,
        qasm_str: str | None = None,
        path_to_qasm_file: Path | None = None,
        use_primary: bool = False,
        graph_key: str = "",
    ) -> AugmentedZXGraph:
        """Add an AugmentedZXGraph to the collection from a QASM string or file.

        Args:
            qasm_str: A quantum circuit encoded as a QASM string.
            path_to_qasm_file: A path to a QASM file.
            use_primary: Flag to set key to primary key.
            graph_key: Open key string to save intermediate/modified ZX graphs.
            debug: Debug mode to use.

        """
        key = self.primary_key if use_primary else graph_key
        aug_zx_graph = AugmentedZXGraph.from_qasm(
            qasm_str=qasm_str, path_to_qasm_file=path_to_qasm_file, debug=self.debug
        )
        self.add_graph(aug_zx_graph, graph_key=key)
        return self._collection[key]

    def set_primary(self, graph_key: str):
        """Switch the key designating the primary AugmentedZXGraph."""
        if graph_key not in self._collection:
            raise ValueError(f"ERROR. Key {graph_key} not found in AugmentedZXGraph collection.")
        self.primary_key = graph_key


########################
# AUGMENTED PyZX GRAPH #
########################
class AugmentedZXGraph:
    """Topologiq's dual-graph PyZX Graph implementation."""

    def __init__(
        self,
        zx_graph: zx.Graph,
        debug: int = 0,
    ):
        """Initialise class with incoming ZX graph or empty one.

        Args:
            zx_graph: The PyZX graph to use as basis for initialisation.
            debug: Debug mode to use.

        """

        print("\n=> Ingesting PyZX graph.")
        self.debug = debug

        # Store the primary graph
        if self.debug > 0:
            print("* Consuming raw input graph.")
        self.zx_graph = zx_graph if zx_graph else zx.Graph()

        # Store a reduced version of the graph
        if self.debug > 0:
            print("* Storing a reduced version of graph.")
        self.zx_graph_reduced = self.zx_graph.copy()
        zx.full_reduce(self.zx_graph_reduced)

        # Get Pauli webs and order for full and reduced graphs
        self.add_pauli_packs()

    @classmethod
    def from_qasm(
        cls,
        qasm_str: str | None = None,
        path_to_qasm_file: Path | None = None,
        remove_lonely_resets: bool = True,
        debug: int = 0,
    ) -> AugmentedZXGraph:
        """Create ZX graph from a QASM string or QASM file.

        Args:
            qasm_str: A quantum circuit encoded as a QASM string.
            path_to_qasm_file: A path to a QASM file.
            remove_lonely_resets: Flag to trigger removal of any resets tied exclusively to an input.
                * Lonely resets happen when QASM uses reset for initialisation and create isolated island graphs.
                * Lonely resets are always followed by a gap and, immediately after, an initialisation spider.
                * Lonely resets are therefore irrelevant for computation.
            debug: Debug mode to use.

        """

        # Health checks
        if not qasm_str and not path_to_qasm_file:
            raise ValueError("ERROR. A QASM string or path to a QASM file is needed.")

        # Load QASM from file, else from QASM string
        if path_to_qasm_file:
            zx_circuit = Circuit.load(str(path_to_qasm_file))
        else:
            zx_circuit = Circuit.from_qasm(qasm_str)

        # Convert to graph
        zx_graph = zx_circuit.to_graph()

        # Remove lonely spiders
        if remove_lonely_resets:
            zx_graph = cls._rm_lonely_resets(zx_graph)
            zx_graph = cls._rm_post_measurement_spiders(zx_graph)

        for ph in zx_graph.phases().values():
            if ph == Fraction(1, 4):
                zx.simplify.gadgetize(zx_graph, graphlike=False)
                zx.id_simp(zx_graph)
                break

        return cls(zx_graph)

    def add_pauli_packs(self):
        """Calculate and store Pauli webs as attributes."""
        if self.debug > 0:
            print("* Calculating Pauli webs using PyZX.")
        try:
            self.pauli_pack = compute_pauli_webs(self.zx_graph)
        except Exception as _:
            print(
                " - WARNING. PyZX Pauli webs unavailable for full graph, which can hinder lattice surgery."
            )
        try:
            self.pauli_pack_reduced = compute_pauli_webs(self.zx_graph_reduced)
        except Exception as _:
            print(
                " - WARNING. PyZX Pauli webs unavailable for reduced graph, which can hinder lattice surgery."
            )

    def check_equality(self, zx_graph: zx.Graph) -> bool:
        """Check if two PyZX graphs are equivalent."""

        print("=> Verifying input/output equality.")

        # Use reduced version of AugZXGraph
        zx_graph_in = self.zx_graph_reduced.copy()

        try:
            zx_graph_out = zx_graph.copy()
            zx.full_reduce(self.zx_graph_reduced)
        except Exception as _:
            print("Equality verification not possible. Cannot reduce output ZX graph.")

        if zx_graph_out:
            try:
                # Add dummy inputs and outputs if not present
                for g in [zx_graph_in, zx_graph_out]:
                    if not zx_graph.inputs():
                        dummy = g.add_vertex(ty=0)
                        dummy_z = g.add_vertex(ty=1)
                        g.add_edge((dummy, dummy_z))
                        g.set_inputs(tuple([dummy]))
                    if not g.outputs():
                        dummy = g.add_vertex(ty=0)
                        dummy_z = g.add_vertex(ty=1)
                        g.add_edge((dummy, dummy_z))
                        g.set_outputs(tuple([dummy]))

                # Convert to tensor
                g1 = zx_graph_in.to_tensor(preserve_scalar=False)
                g2 = zx_graph_out.to_tensor(preserve_scalar=False)

                # Compare tensors
                verification = zx.compare_tensors(g1, g2)
                print(f"  - Result: {'EQUIVALENT' if verification else 'NOT equivalent'}.")
                return verification
            except Exception as e:
                print(f"  - Compare tensors failed during verification: {e}")

        print("  - Verification inconclusive. Verification method returns `False` by default.")
        return False

    def get_native_visualisation(self, use_reduced: bool = False) -> Any:
        """Convert PyZX graph into a positioned NX graph that allows 3D visualisation."""
        fig_data = zx.draw_matplotlib(
            self.zx_graph_reduced if use_reduced else self.zx_graph, labels=True
        )
        return fig_data

    def get_visual_data(self, use_reduced: bool = False) -> nx.Graph:
        """Convert PyZX graph into a positioned NX graph that allows 3D visualisation."""

        # Work on copy ZX graph
        zx_graph = self.zx_graph_reduced.copy() if use_reduced else self.zx_graph.copy()

        # Create base NX graph
        zx_graph_as_nx = nx.Graph()

        # Loop vertices -> nodes
        for v_id in zx_graph.vertices():
            # Core info
            t = zx_graph.type(v_id)
            phase = zx_graph.phase(v_id)
            phase_float = float(phase) if isinstance(phase, (Fraction, int, float)) else 0.0
            qubit = zx_graph.qubit(v_id)
            row = zx_graph.row(v_id)

            # Derivative info
            t_name = ZXTypes(t).name
            color = ZXColors.lookup(t_name)

            # Create rich/verbose NX node
            zx_graph_as_nx.add_node(
                v_id,
                type=t_name,
                qubit=qubit,
                row=row,
                color=color,
                phase=phase,
                phase_float=phase_float,
            )

        # Loop ZX edges -> NX edges
        for e_id in zx_graph.edges():
            # Core info
            src_id, tgt_id = zx_graph.edge_st(e_id)
            t = zx_graph.edge_type(e_id)

            # Derivative info
            t_name = ZXEdgeTypes(t).name
            color = ZXColors.lookup(t_name)

            # Create rich/verbose NX edge
            zx_graph_as_nx.add_edge(
                src_id, tgt_id, etype=t_name, color=color, hdm=True if t == 2 else False
            )

        # Define positions using NX layouts
        if zx_graph_as_nx.number_of_nodes() > 1:
            pos_dict = nx.spectral_layout(zx_graph_as_nx, dim=3)
            for v_id, coords in pos_dict.items():
                zx_graph_as_nx.nodes[v_id]["pos"] = tuple((coords * 10).tolist())
        elif zx_graph_as_nx.number_of_nodes() == 1:
            v_id = list(zx_graph_as_nx.nodes)[0]
            zx_graph_as_nx.nodes[v_id]["pos"] = (0, 0, 0)

        return zx_graph_as_nx

    @staticmethod
    def _rm_lonely_resets(zx_graph: zx.Graph) -> zx.Graph:
        """Remove reset-initialisation spiders from a ZX graph.

        Args:
            zx_graph: A PyZX graph with lonely spiders in the first "row"
                (happens when loading QASM files using reset for initialisation).

        Returns:
            zx_graph: The updated ZX graph.

        """

        lonely_spider_ids = []
        inputs = zx_graph.inputs()
        for in_id in inputs:
            neigh_ids = list(zx_graph.neighbors(in_id))
            if len(neigh_ids) == 1:
                neigh_neigh_ids = list(zx_graph.neighbors(neigh_ids[0]))
                if len(neigh_neigh_ids) == 1:
                    if neigh_neigh_ids[0] == in_id:
                        lonely_spider_ids.extend([in_id, neigh_ids[0]])
        zx_graph.remove_vertices(lonely_spider_ids)
        return zx_graph

    @staticmethod
    def _rm_post_measurement_spiders(zx_graph: zx.Graph) -> zx.Graph:
        """Remove spiders that come immediately after a measured spider.

        Args:
            zx_graph: A PyZX graph with spiders after measured spiders.
                (happens when loading QASM files).

        Returns:
            zx_graph: The updated ZX graph.

        """
        measure_spiders_id = [
            spider_id
            for spider_id in zx_graph.vertices()
            if isinstance(zx_graph.phase(spider_id), zx.symbolic.Poly)
        ]

        post_measurement_spiders = []
        for spider_id in measure_spiders_id:
            post_measurement_spiders.extend(
                [
                    neigh_id
                    for neigh_id in zx_graph.neighbors(spider_id)
                    if zx_graph.row(neigh_id) > zx_graph.row(spider_id)
                ]
            )

        zx_graph.remove_vertices(post_measurement_spiders)

        return zx_graph
