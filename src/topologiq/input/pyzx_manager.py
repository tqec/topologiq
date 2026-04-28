"""PyZX graph and PyZX graph manager classes.

This module provides a unified interface for ingesting PyZX circuits from
QASM as well as managing and producing PyZX graphs from them.

"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import networkx as nx
import pyzx as zx
from pyzx.circuit import Circuit

from topologiq.core.graph_manager.graph_manager import runner
from topologiq.input.utils import ZXColors, ZXEdgeTypes, ZXTypes
from topologiq.utils.classes import SimpleDictGraph, StandardBlock
from topologiq.utils.misc import kind_to_zx_type


######################
# PyZX GRAPH MANAGER #
######################
class ZXGraphManager:
    """Registry class to keep augmented ZX graphs organised."""

    def __init__(self, primary_key: str = "primary"):
        """Initialise class with incoming or default primary key and empty collection."""
        self.primary_key: str = primary_key
        self._collection: dict[str, AugmentedZXGraph] = {}

    def get_graph(
        self,
        use_primary: bool = False,
        graph_key: str = "",
    ) -> AugmentedZXGraph:
        """Retrieve an augmented ZX graph from the collection.

        Args:
            use_primary: Flag to set key to primary key.
            graph_key: Open key string to save intermediate/modified ZX graphs.

        """
        key = self.primary_key if use_primary else graph_key
        if not key or key not in self._collection:
            raise ValueError(f"ERROR. Key {key} not in augmented ZX graph collection.")
        return self._collection[key]

    def add_graph(
        self,
        aug_zx_graph: AugmentedZXGraph,
        graph_key: str,
    ):
        """Add an augmented ZX graph to the collection.

        Args:
            aug_zx_graph: The augmented ZX graph to preserve.
            graph_key: String to use as collection key.

        """
        if not graph_key:
            raise ValueError("ERROR. A key is needed to add an augmented ZX graph to collection.")
        self._collection[graph_key] = aug_zx_graph

    def add_graph_from_pyzx(
        self,
        zx_graph: zx.Graph,
        use_primary: bool = False,
        graph_key: str = "",
    ) -> AugmentedZXGraph:
        """Add an augmented ZX graph to the collection starting with a standard PyZX graph.

        Args:
            zx_graph: The PyZX graph.
            use_primary: Flag to set key to primary key.
            graph_key: Open key string to save intermediate/modified ZX graphs.

        """
        key = self.primary_key if use_primary else graph_key
        self.add_graph(AugmentedZXGraph(zx_graph), graph_key=key)
        return self._collection[key]

    def add_graph_from_qasm(
        self,
        qasm_str: str | None = None,
        path_to_qasm_file: Path | None = None,
        use_primary: bool = False,
        graph_key: str = "",
    ) -> AugmentedZXGraph:
        """Add an augmented ZX graph to the collection from a QASM string or file.

        Args:
            qasm_str: A quantum circuit encoded as a QASM string.
            path_to_qasm_file: A path to a QASM file.
            use_primary: Flag to set key to primary key.
            graph_key: Open key string to save intermediate/modified ZX graphs.

        """
        key = self.primary_key if use_primary else graph_key
        aug_zx_graph = AugmentedZXGraph.from_qasm(
            qasm_str=qasm_str, path_to_qasm_file=path_to_qasm_file
        )
        self.add_graph(aug_zx_graph, graph_key=key)
        return self._collection[key]

    def add_graph_from_blockgraph(
        self,
        blockgraph_cubes: dict[int, StandardBlock],
        blockgraph_pipes: dict[tuple[int, int], list[str | tuple[int, int]]],
        use_primary: bool = False,
        graph_key: str = "",
        other: AugmentedZXGraph | None = None,
    ) -> AugmentedZXGraph:
        """Add an augmented ZX graph to the collection from a blockgraph.

        Args:
            use_primary: Flag to set key to primary key.
            graph_key: Open key string to save intermediate/modified ZX graphs.
            blockgraph_cubes: The cubes of the blockgraph.
            blockgraph_pipes: The pipes of the blockgraph.
            other: A separate ZX graph against which to compare.

        """
        key = self.primary_key if use_primary else graph_key
        aug_zx_graph = AugmentedZXGraph.from_blockgraph(
            blockgraph_cubes=blockgraph_cubes, blockgraph_pipes=blockgraph_pipes, other=other
        )
        self.add_graph(aug_zx_graph, graph_key=key)
        return self._collection[key]

    def set_primary(self, graph_key: str):
        """Switch the key designating the primary augmented ZX graph."""
        if graph_key not in self._collection:
            raise ValueError(f"ERROR. Key {graph_key} not found in augmented ZX graph collection.")
        self.primary_key = graph_key


########################
# AUGMENTED PyZX GRAPH #
########################
class AugmentedZXGraph:
    """Topologiq's dual-graph PyZX Graph implementation."""

    def __init__(self, zx_graph: zx.Graph):
        """Initialise class with incoming ZX graph or empty one."""
        self.zx_graph = zx_graph if zx_graph else zx.Graph()
        self.zx_graph_reduced = self.zx_graph.copy()
        zx.full_reduce(self.zx_graph_reduced)

    @classmethod
    def from_qasm(
        cls,
        qasm_str: str | None = None,
        path_to_qasm_file: Path | None = None,
        remove_lonely_resets: bool = True,
    ) -> AugmentedZXGraph:
        """Create ZX graph from a QASM string or QASM file.

        Args:
            qasm_str: A quantum circuit encoded as a QASM string.
            path_to_qasm_file: A path to a QASM file.
            remove_lonely_resets: Flag to trigger removal of any resets tied exclusively to an input.
                * Lonely resets happen when QASM uses reset for initialisation and create isolated island graphs.
                * Lonely resets are always followed by a gap and, immediately after, an initialisation spider.
                * Lonely resets are therefore irrelevant for computation.

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

        return cls(zx_graph)

    @classmethod
    def from_blockgraph(
        cls,
        blockgraph_cubes: dict[int, StandardBlock],
        blockgraph_pipes: dict[tuple[int, int], list[str | tuple[int, int]]],
        other: AugmentedZXGraph | None = None,
    ) -> AugmentedZXGraph:
        """Create ZX graph from an blockgraph.

        Args:
            path_to_input_file: The path to the input `.bgraph` file.
            blockgraph_cubes: The cubes of the blockgraph.
            blockgraph_pipes: The pipes of the blockgraph.
            other: A separate ZX graph against which to compare.

        """
        zx_graph = zx.Graph()
        id_swaps = {}

        for cube_id, (coords, kind) in blockgraph_cubes.items():
            zx_type = ZXTypes.from_str(kind_to_zx_type(kind))

            if other and cube_id in other.zx_graph.vertex_set():
                qubit = other.zx_graph.qubit(cube_id)
                row = other.zx_graph.row(cube_id)
            else:
                qubit = -1
                row = -1

            vertex = zx_graph.add_vertex(ty=zx_type, qubit=qubit, row=row, index=cube_id)

            zx_graph.set_vdata(vertex, "coords", coords)
            id_swaps[cube_id] = vertex

        for (src_id, tgt_id), (kind, _) in blockgraph_pipes.items():
            zx_type = ZXEdgeTypes.from_str(kind_to_zx_type(kind))
            zx_graph.add_edge((id_swaps[src_id], id_swaps[tgt_id]), edgetype=zx_type)

        if other.zx_graph.inputs():
            zx_graph.set_inputs(other.zx_graph.inputs())

        if other.zx_graph.outputs():
            zx_graph.set_outputs(other.zx_graph.outputs())

        # Set graph in class
        return cls(zx_graph)

    def get_blockgraph(
        self: AugmentedZXGraph,
        circuit_name: str = "primary",
        use_reduced: bool = False,
        final_vis=False,
        **kwargs,
    ) -> tuple[
        dict[int, StandardBlock] | None,
        dict[tuple[int, int], list[str | tuple[int, int]]] | None,
    ]:
        """Perform algorithmic lattice surgery on ZX graph.

        Args:
            circuit_name: The name of the circuit.
            use_reduced: Whether to use the full or reduced version of the ZX graph.
            final_vis: Trigger simple visualisation of output blockgraph.
            **kwargs: See `./kwargs.py` for a comprehensive breakdown.
                NB! If an arbitrary kwarg is not given explicitly, it is created against defaults on `./src/topologiq/kwargs.py`.
                NB! By extension, it only makes sense to give the specific kwargs where user wants to deviate from defaults.

        """

        # Choose if to show visualisation of outcome or not
        if not kwargs:
            kwargs = {"vis_options": ("final" if final_vis else None, None)}

        # Choose full or reduced ZX graph
        zx_graph = self.zx_graph_reduced if use_reduced else self.zx_graph

        # Move into Topologiq's native format
        simple_graph = pyzx_g_to_simple_g(zx_graph)

        # Perform lattice surgery
        _, _, blockgraph_cubes, blockgraph_pipes = runner(
            simple_graph,  # The simple_graph to be processed by Topologiq
            circuit_name,  # Name of the circuit
            **kwargs,
        )

        return blockgraph_cubes, blockgraph_pipes

    def check_equality(self, other: AugmentedZXGraph) -> bool:
        """Check if two PyZX graphs are equivalent."""

        # NB! None of this is currently working because blockgraph
        # does not yet come with explicit input/output labels.
        # Needs to be revisited when I/O labels are added to
        # blockgraph

        try:
            zx_graph_in = self.zx_graph_reduced.copy()
            zx_graph_out = other.zx_graph_reduced.copy()

            for zx_graph in [zx_graph_in, zx_graph_out]:
                if not zx_graph.inputs():
                    dummy = zx_graph.add_vertex(ty=0)
                    dummy_z = zx_graph.add_vertex(ty=1)
                    zx_graph.add_edge((dummy, dummy_z))
                    zx_graph.set_inputs(tuple([dummy]))
                if not zx_graph.outputs():
                    dummy = zx_graph.add_vertex(ty=0)
                    dummy_z = zx_graph.add_vertex(ty=1)
                    zx_graph.add_edge((dummy, dummy_z))
                    zx_graph.set_outputs(tuple([dummy]))
            print(
                "\nVerifying equality. Input ZX (reduced) v. Output BGRAPH->ZX (reduced).",
                f"\nIn: {zx_graph_in}, i: {zx_graph_in.inputs()}, o: {zx_graph_in.outputs()}",
                f"\nOut: {zx_graph_in}, i: {zx_graph_out.inputs()}, o: {zx_graph_out.outputs()}",
            )

            g1 = zx_graph_in.to_tensor(preserve_scalar=False)
            g2 = zx_graph_out.to_tensor(preserve_scalar=False)
            zx.draw(zx_graph_out, labels=True)
            return zx.compare_tensors(g1, g2)
        except Exception as e:
            print(f"Compare tensors failed during verification: {e}")

        print("\nVerification inconclusive. Method returns False as default.")
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


#########################
# PyZX METHODS WRAPPERS #
#########################
# SOON TO BE LEGACY #
#####################
def qasm_to_pyzx(qasm_str: str) -> zx.BaseGraph:
    """Import a circuit from QASM and convert it to a PyZX graph.

    Args:
        qasm_str: A quantum circuit encoded as a QASM string.

    """
    # QASM --> PyZX circuit --> PyZX graph
    zx_circuit = zx.Circuit.from_qasm(qasm_str)
    pyzx_graph = zx_circuit.to_graph()

    return zx_circuit, pyzx_graph


def get_dict_from_pyzx(g: zx.BaseGraph | zx.GraphS):
    """Extract circuit information from a PyZX graph and dumps it into a dictionary.

    Args:
        g: a PyZX graph.

    Returns:
        g_dict: a dictionary with graph info.

    """

    # EMPTY DICT FOR RESULTS
    g_dict: dict[str, dict] = {"meta": {}, "nodes": {}, "edges": {}}

    # GET AND TRANSFER DATA FROM PyZX
    try:
        # Dump graph into dict
        dict_graph = g.to_dict(include_scalar=True)

        # Add meta-information
        g_dict["meta"]["scalar"] = dict_graph["scalar"]

        # Add nodes
        for v in g.vertices():
            g_dict["nodes"][v] = {
                "coords": (0, 0, 0),
                "rot": (0, 0, 0),
                "scale": (0, 0, 0),
                "type": g.type(v).name,
                "phase": str(g.phase(v)),
                "degree": g.vertex_degree(v),
                "connections": list(g.neighbors(v)),
            }

        # Add edges
        c = 0
        for e in g.edges():
            typed_type: zx.EdgeType = cast(zx.EdgeType, g.edge_type(e))
            g_dict["edges"][f"e{c}"] = {
                "type": typed_type.name,
                "src": e[0],
                "tgt": e[1],
            }
            c += 1

    except Exception as e:
        print(f"Error extracting info from graph: {e}")

    return g_dict


def pyzx_g_to_simple_g(g: zx.BaseGraph | zx.GraphS) -> SimpleDictGraph:
    """Extract circuit information from a PyZX graph and dumps it into a simple graph.

    Args:
        g: a PyZX graph.

    Returns:
        g_simple: a dictionary with graph info.

    """

    # GET FULL GRAPH INTO DICTIONARY
    g_full = get_dict_from_pyzx(g)

    # TRANSFER INTO A SIMPLE GRAPH
    g_simple: SimpleDictGraph = {"nodes": [], "edges": []}
    for n in g_full["nodes"]:
        n_type = "O" if g_full["nodes"][n]["type"] == "BOUNDARY" else g_full["nodes"][n]["type"]
        g_simple["nodes"].append((n, n_type))

    for e in g_full["edges"]:
        src = g_full["edges"][e]["src"]
        tgt = g_full["edges"][e]["tgt"]
        e_type = g_full["edges"][e]["type"]
        g_simple["edges"].append(((src, tgt), e_type))

    return g_simple
