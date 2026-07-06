"""Primary graph management class and related operations, incl. the outer graph manager BFS.

This file contains classes and functions that altogether create Topologiq's core blockgraph,
determine the first spider to process, the order in which subsequent spiders get processed,
and calls the inner pathfinder algorithm as appropriate per edge.

Usage:
    Create an BlockGraphManager from a separate script and use its methods as appropriate.

"""

import os
import random
from fractions import Fraction
from itertools import chain
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pyzx as zx
from pyzx.pauliweb import compute_pauli_webs

from topologiq.core.blocks import CandidatePath, PositionedZXBlock, ZXBlockRegistry
from topologiq.core.graph_manager.beams import (
    check_beam_clashes,
    check_beam_clashes_for_twins,
    prune_beams,
)
from topologiq.core.graph_manager.edge_handlers import add_path_to_bgraph, queue_precalc
from topologiq.core.graph_manager.first_cube import get_first_cube_data
from topologiq.core.graph_manager.kwargs import check_assemble_kwargs
from topologiq.core.graph_manager.spatial import (
    gen_tent_tgt_coords,
    max_four_edges_random,
    max_four_edges_single_spider_graph,
)
from topologiq.core.pathfinder.pathfinder import pathfinder
from topologiq.core.pathfinder.symbolic import check_exits_add_beams
from topologiq.utils.classes import CubeBeams, GraphBounds, StandardCoord
from topologiq.utils.file import rm_temp_files
from topologiq.utils.manhattan import get_manhattan
from topologiq.utils.time import datetime_manager
from topologiq.utils.zx import ZXEdgeTypes, ZXTypes
from topologiq.vis.animate import create_animation
from topologiq.vis.draw import BlockGraphVisualiser, VisualiserState, draw_as_zx

#########
# PATHS #
#########
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent.parent
BGRAPH_DIR = REPO_ROOT / "output/bgraph"
MEDIA_DIR = REPO_ROOT / "output/media"


#######################
# OUTER GRAPH MANAGER #
#######################
class BlockGraphManager:
    """Topologiq's primary graph management subroutine."""

    def __init__(
        self,
        input_zx_graph: zx.Graph,
        graph_name: str = "computation",
        **kwargs,
    ):
        """Initialise with empty blockgraph."""

        # Delete any temporary files that may have survived previous builds
        rm_temp_files(MEDIA_DIR / "temp")

        # Direct assignations
        self.graph_name: str = graph_name
        self.input_zx_graph: zx.Graph = input_zx_graph

        # Pad (if some given) or build (if none given) kwargs
        self._kwargs = check_assemble_kwargs(**kwargs)

        # Set seed if KWARGs has one (used for testing)
        if self._kwargs["seed"]:
            random.seed(self._kwargs["seed"])

        # Internalise key input ZX graph metrics
        self.init_zx_trackers()

        # Initialise blockgraph
        self.init_blockgraph()

    def init_zx_trackers(self):
        """Internalise input ZX graph."""

        # Extract key data for future reference
        self.in_ids: set[int] = self.input_zx_graph.vertex_set()
        self.in_qubits: dict[int, int] = self.input_zx_graph.qubits()
        self.in_rows: dict[int, int] = self.input_zx_graph.rows()
        self.in_inputs: list[int] = self.input_zx_graph.inputs()
        self.in_outputs: list[int] = self.input_zx_graph.outputs()
        self.in_phases: dict[int, int | Fraction] = self.input_zx_graph.phases()
        self.in_edge_types: dict[tuple[int, int], str] = {
            edge_id: ZXEdgeTypes(self.input_zx_graph.edge_type(edge_id)).name
            for edge_id in self.input_zx_graph.edges()
        }
        self.in_degrees: dict[int, int] = {
            k: len(self.input_zx_graph.neighbors(k)) for k in self.input_zx_graph.vertex_set()
        }

        self.in_types: dict[int, str] = {}
        for k, v in self.input_zx_graph.types().items():
            switch_to_cube = self.input_zx_graph.vdata(k, "switch_to_cube", default=None)
            if not switch_to_cube:
                self.in_types[k] = "O" if v == 0 else ZXTypes(v).name
            else:
                self.in_types[k] = switch_to_cube

    def init_blockgraph(self):
        """Build an preliminary non-positioned blockgraph from the input ZX graph."""

        # Space management objects
        self.taken: set[StandardCoord] = set()
        self.beams: dict[int, CubeBeams] = {}
        self.beams_short: dict[int, CubeBeams] = {}
        self.ids_to_twin: list[int] = set()
        self.completed_in_zx_edges: dict[tuple[int, int], list[int]] = {}

        # Replicate ZX trackers to have editable copies
        self.ids: set[int] = self.in_ids.copy()
        self.types: dict[int, str] = self.in_types.copy()
        self.edge_types: dict[tuple[int, int], str] = self.in_edge_types.copy()
        self.degrees: dict[int, int] = self.in_degrees.copy()
        self.qubits: dict[int, int] = self.in_qubits.copy()
        self.rows: dict[int, int] = self.in_rows.copy()
        self.inputs: list[int] = self.in_inputs
        self.outputs: list[int] = self.in_outputs
        self.phases: dict[int, int | Fraction] = self.in_phases.copy()

        # Introduce empty trackers for completion and time-order dependencies
        self.completed_base_ids: list[int] = []
        self.completed_base_edges: dict[tuple[int, int], list[int]] = {}
        self.ante: dict[int, set[int]] = {}
        self.post: dict[int, set[int]] = {}

        if self._kwargs["graph_traverse_mode"] == "cnot-cycles":
            spiders_by_row: dict[int, set[int]] = {}
            for spider_id, row_number in self.rows.items():
                if row_number not in spiders_by_row:
                    spiders_by_row[row_number] = set([spider_id])
                else:
                    spiders_by_row[row_number].add(spider_id)

            prev_row = None
            for curr_row, spider_ids_in_row in spiders_by_row.items():
                if prev_row:
                    for spider_id in spider_ids_in_row:
                        pass
                        # self.ante[spider_id] = spiders_by_row[prev_row]
                prev_row = curr_row

        # Other trackers
        self.run_success: bool = False

        # Initialise empty NX graph
        self.bgraph = nx.Graph()

        # Cubes
        [
            self.bgraph.add_node(
                n_id,
                zx_block=ZXBlockRegistry.get_create(zx_type=self.types[n_id]),
                coords=None,
                completions={
                    "degree": self.degrees[n_id],
                    "pending": self.degrees[n_id],
                },
            )
            for n_id in self.input_zx_graph.vertices()
        ]

        # Edges
        [
            self.bgraph.add_edge(
                u, v, edge_type=zx_e_type, start_coords=None, end_coords=None, kind=None
            )
            for (u, v), zx_e_type in self.edge_types.items()
        ]

        # Reactive re-writes
        # Re-writes needed to ensure the ZX graph does not have
        # any spider with more than four edges are reflected in the
        # ZX trackers created in `init_zx_trackers`
        self.enforce_max_four_legs_per_spider()  # Enforce max. allowed neighbour number

        # Proactive re-writes
        # Re-writes needed to convert from spiders using standard PyZX conventions to
        # LS-friendly cubes and patterns are NOT reflected in the ZX trackers
        self.handle_y_t_cubes()

        # Calculate boundaries of theoretical chip surface
        self.get_bounds()

        # Place the first cube at centre of available space
        self.place_first_cube()

        # Pre-calculate edge_queue based on graph traverse strategy
        self.get_queue()

        # Re-initialise trackers to keep trace of any changes introduced by graph rewrites
        self.ids = set(self.bgraph.nodes())
        self.types = {k: zx_block.zx_type for k, zx_block in self.bgraph.nodes(data="zx_block")}
        self.degrees = {k: self.bgraph.degree(k) for k in self.bgraph.nodes()}
        self.edge_types = {
            (u, v): attrs["edge_type"] for u, v, attrs in self.bgraph.edges(data=True)
        }
        self.twin_trace: dict[int, list[int]] = {k: [k] for k in self.ids}
        self.twin_trace_inverse: dict[int, int] = {}

        # Rewrite pending information to reflect any changes introduced by graph rewrites
        for spider_id in self.bgraph.nodes():
            self.bgraph.nodes[spider_id]["completions"]["degree"] = self.bgraph.degree(spider_id)
            self.bgraph.nodes[spider_id]["completions"]["pending"] = self.bgraph.degree(spider_id)

        # Visualise base_graph
        if self._kwargs["debug"] >= 4:
            self.draw_zx(draw_style="zx")

    def handle_y_t_cubes(self):
        """Convert spiders with phases to Y- and T-cube patterns."""

        # Trackers for special cubes introduced in this method
        self.y_cubes: dict[int, str] = {}
        self.s_gates: dict[int, list[tuple[int, int] | None]] = {}
        self.msc_cubes: dict[int, str] = {}
        self.t_gates: dict[int, list[tuple[int, int] | None]] = {}
        self.t_zx_tracker: dict[int, int] = {}

        # Look over nodes with phases
        for spider_id, phase in self.phases.items():
            if phase == Fraction(1, 2):
                # Calculate next ID
                y_id = max(self.bgraph.nodes()) + 1

                # Remove phase of original node as it will get a pattern instead
                self.phases[spider_id] = 0

                # Initialisation Y-cube
                if self.bgraph.degree(spider_id) == 1:
                    # Log and change ZX block in node attributes
                    self.y_cubes[spider_id] = "Yi"
                    self.bgraph.nodes[spider_id]["zx_block"] = ZXBlockRegistry.get_create(
                        zx_type="Y"
                    )

                # Mid-circuit S-gate
                else:
                    # Log Y-cube as measurement
                    self.s_gates[spider_id] = [(spider_id, y_id)]
                    self.y_cubes[y_id] = "Ym"

                    # Add Y-cube to BGRAPH
                    self.bgraph.add_node(
                        y_id,
                        zx_block=ZXBlockRegistry.get_create(zx_type="Y"),
                        coords=None,
                        completions={
                            "degree": None,
                            "pending": None,
                        },
                    )

                    # Add corresponding entry to qubit and row trackers
                    self.qubits[y_id] = self.qubits[spider_id] - 1
                    self.rows[y_id] = self.rows[spider_id] - 1

                    # Add corresponding edge
                    self.bgraph.add_edge(
                        spider_id,
                        y_id,
                        edge_type="SIMPLE",
                        start_coords=None,
                        end_coords=None,
                        kind=None,
                    )

            if phase == Fraction(1, 4):
                # Determine current max ID in graph
                max_id = max(self.bgraph.nodes())

                # Initialisation T-gate (MSC)
                if spider_id in self.inputs:
                    # Remove phase of original node as it will get a pattern instead
                    self.phases[spider_id] = 0

                    # Add to MSC block to tracker
                    self.msc_cubes[spider_id] = "Mi"

                    # Update graph node to a T
                    self.bgraph.nodes[spider_id]["zx_block"] = ZXBlockRegistry.get_create(
                        zx_type="T"
                    )

                else:
                    # IDs for all spiders in sequence
                    msc_id = max_id + 1
                    x_bridge_id, y_id, xz_id = (max_id + 3, max_id + 4, max_id + 5)

                    # Add to MSC block to tracker
                    # Add to MSC block to tracker
                    self.msc_cubes[spider_id] = "Mm"
                    self.t_gates[spider_id] = [
                        (spider_id, msc_id),
                        (spider_id, x_bridge_id),
                        (x_bridge_id, y_id),
                        (x_bridge_id, xz_id),
                    ]
                    self.t_zx_tracker[spider_id] = xz_id

                    # Time orders for all spiders in sequence
                    self.ante[xz_id] = set([spider_id, msc_id, x_bridge_id, y_id])

                    # Reset phase on original spider since it is getting a pattern instead
                    self.phases[spider_id] = 0

                    # Attach T
                    self.bgraph.add_node(
                        msc_id,
                        zx_block=ZXBlockRegistry.get_create(zx_type="T"),
                        coords=None,
                        completions={
                            "degree": None,
                            "pending": None,
                        },
                    )
                    self.qubits[msc_id] = self.qubits[spider_id]
                    self.rows[msc_id] = self.rows[spider_id] - 1

                    self.bgraph.add_edge(
                        spider_id,
                        msc_id,
                        edge_type="SIMPLE",
                        start_coords=None,
                        end_coords=None,
                        kind=None,
                    )

                    # Attach Y-XZ combo pattern (full)
                    prev_id = spider_id
                    types_in_seq = {0: "X", 1: "Y", 2: "XZ"}
                    for i, s_id in enumerate([x_bridge_id, y_id, xz_id]):
                        zx_type = types_in_seq[i]
                        self.bgraph.add_node(
                            s_id,
                            zx_block=ZXBlockRegistry.get_create(zx_type=zx_type),
                            coords=None,
                            completions={
                                "degree": None,
                                "pending": None,
                            },
                        )
                        self.qubits[s_id] = self.qubits[spider_id] - (0 if zx_type != "Y" else -1)
                        self.rows[s_id] = self.rows[prev_id] + (1 if zx_type != "Y" else 0)
                        self.bgraph.add_edge(
                            prev_id if zx_type != "XZ" else x_bridge_id,
                            s_id,
                            edge_type="SIMPLE",
                            start_coords=None,
                            end_coords=None,
                            kind=None,
                        )
                        prev_id = s_id

        # Draw ZX as NX if applicable
        if self.t_gates:
            try:
                self.build_deps_from_pauli_webs()
            except Exception as e:
                print(f"Error calculating time constraints for T-gates: {e}.")

            if self._kwargs["debug"] >= 4:
                self.draw_zx(draw_style="zx")

    def build_deps_from_pauli_webs(self):
        """Build a set of cube IDs that must be placed BEFORE their any arbitrary T-gate."""

        # Compute Pauli webs
        order, zwebs, xwebs = compute_pauli_webs(self.input_zx_graph)

        # Proceed only if it was possible to compute Pauli webs
        # Note. Block adds ID of conditional of applicable T-gate pattern
        if order and zwebs and xwebs:
            for t_gate_id in order:
                if t_gate_id in (*xwebs, *zwebs):
                    # Add any IDs in path of X webs
                    if t_gate_id in xwebs:
                        self.ante[self.t_zx_tracker[t_gate_id]].update(
                            chain.from_iterable(xwebs[t_gate_id].half_edges())
                        )

                    # Add any IDs in path of Z webs
                    if t_gate_id in zwebs:
                        self.ante[self.t_zx_tracker[t_gate_id]].update(
                            chain.from_iterable(zwebs[t_gate_id].half_edges())
                        )

            # Use the BEFORE deps just contructed to build inverse AFTER deps
            self.build_rebuild_post_deps()

    def build_rebuild_post_deps(self):
        """Use BEFORE time constraints to build a set of IDs of cubes that must be placed AFTER their respective key."""

        # Proceed only if there are BEFORE deps to work with
        if self.ante:
            for k, predecessors in self.ante.items():
                for predecessor in predecessors:
                    # Add each predecessor T-gate as successor of corresponding ID
                    if predecessor in self.post:
                        self.post[predecessor].update([k])
                    else:
                        self.post[predecessor] = set([k])

    def get_bounds(self):
        """Define the max/min bounds for the chip surface."""

        self.bounds = GraphBounds()
        if "size_of_chip" in self._kwargs and "k" in self._kwargs:
            self.bounds.x = int(self._kwargs["size_of_chip"][0] / (self._kwargs["k"] + 2))
            self.bounds.y = int(self._kwargs["size_of_chip"][1] / (self._kwargs["k"] + 2))

    def get_nodes_verbose(self):
        """Retrieve all nodes in blockgraph (verbose)."""
        return [{"id": n, **attrs} for n, attrs in self.bgraph.nodes(data=True)]

    def get_pipes_verbose(self):
        """Retrieve all nodes in blockgraph (verbose)."""
        return [{"id": n, **attrs} for n, attrs in self.bgraph.edges(data=True)]

    def get_volume(self):
        """Get the space-time volume of the main blockgraph."""
        return len(
            [n for n, attrs in self.bgraph.nodes(data=True) if attrs["zx_block"].zx_type != "O"]
        )

    def enforce_max_four_legs_per_spider(self):
        """Ensure all spiders in an incoming blockgraph have at most four legs/edges."""

        # Proceed if there are
        # spiders with more than 4 neighbours
        # S-gates with more than 3 neighbours
        more_than_four_ids = [v for _, v in self.bgraph.degree if v > 4]
        s_gates_more_than_three = [
            k for k, v in self.phases.items() if (v == Fraction(1, 2) and self.bgraph.degree(k) > 3)
        ]
        if more_than_four_ids or s_gates_more_than_three:
            # Determine if graph has only one colour spider
            non_boundary_spiders = [
                n
                for n, attr in self.bgraph.nodes(data=True)
                if (attr.get("zx_block").zx_type != "O")
            ]

            # Use special method for single spider graphs
            if len(non_boundary_spiders) == 1:
                self.bgraph, new_ids = max_four_edges_single_spider_graph(self.bgraph)
            # Use generic method for all other graphs
            else:
                self.bgraph, new_ids = max_four_edges_random(
                    self.bgraph, s_gates_more_than_three=s_gates_more_than_three
                )

            # Update ID trackers to match changes in input graph
            self.ids.update(new_ids.keys())
            for new_id, ref_id in new_ids.items():
                self.rows[new_id] = self.rows[ref_id]
                self.qubits[new_id] = self.qubits[ref_id]
                self.types[new_id] = self.types[ref_id]
                self.phases[new_id] = 0
            self.edge_types = {
                (u, v): attrs["edge_type"] for u, v, attrs in self.bgraph.edges(data=True)
            }
            self.degrees = {k: self.bgraph.degree(k) for k in self.bgraph.nodes()}
            self.twin_trace: dict[int, list[int]] = {k: [k] for k in self.bgraph.nodes()}

    def place_first_cube(self):
        """Define and place the very first cube of the blockgraph."""

        # Reset trackers innaplicable to single cube placement
        self.cross_edge = False
        self.is_hadamard = False

        # Pick and ID and kind for first cube
        self.first_id, self.other_first_ids, self.first_zx_block, src_beams, src_beams_short = (
            get_first_cube_data(
                self.bgraph,
                self._kwargs["first_id_strategy"],
                self.qubits,
                self.inputs,
                self._kwargs["beams_len_short"],
                override_first_cube=self._kwargs["first_cube"],
                random_seed=self._kwargs["seed"],
            )
        )

        if self._kwargs["debug"] >= 4:
            self.draw_zx(draw_style="nx-atlas")

        # Update corresponding node
        first_coord = (
            (0, 0, 0)
            if not self.bounds.x or not self.bounds.y
            else (int(self.bounds.x / 2), int(self.bounds.y / 2), 0)
        )
        self.bgraph.nodes[self.first_id]["zx_block"] = self.first_zx_block
        self.bgraph.nodes[self.first_id]["coords"] = first_coord
        self.beams[self.first_id] = src_beams
        self.beams_short[self.first_id] = src_beams_short

        # Update taken
        self.taken.add(first_coord)

        # Update user if applicable
        if self._kwargs["debug"] > 0:
            print(
                f"First cube placed. ID: {self.first_id}. Kind: {self.first_zx_block.kind}. Coords: {first_coord}."
            )

    def get_queue(self):
        """Build queue using strategy defined in kwargs."""
        self.edge_queue = queue_precalc(
            self.bgraph,
            self.first_id,
            self.rows,
            self.qubits,
            self.inputs,
            self._kwargs["graph_traverse_mode"],
            other_first_ids=self.other_first_ids,
            t_gates=self.t_gates,
        )

    def build(self):
        """Build the blockgraph using pre-defined edge queue."""

        # Start timer
        self.t1, _ = datetime_manager()

        # Loop through edge queue
        for raw_u, raw_v in self.edge_queue:
            # Start iteration timer
            self.t1_iter, _ = datetime_manager()

            # Get (u, v) from the ID trace
            # If a given node has not twins, the last twin is itself
            # If a given node has twins, last twin is active twin
            u = self.twin_trace[raw_u][-1]
            v = self.twin_trace[raw_v][-1]

            # Ensure current source was placed in a prior iteration.
            if (
                self.bgraph.nodes[u]["zx_block"].kind is None
                or self.bgraph.nodes[u]["coords"] is None
            ):
                raise ValueError(f"BFS failed. Malformed source block: {u} --> {v}")

            # Internalise key edge characteristics (ease of access)
            self.curr_src_id, self.curr_tgt_id = (u, v)
            self.cross_edge = True if self.bgraph.nodes[v]["coords"] else False
            self.is_hadamard = self.bgraph.edges[(u, v)]["edge_type"] == "HADAMARD"
            self.curr_src_coords = self.bgraph.nodes[self.curr_src_id]["coords"]
            self.curr_tgt_coords = self.bgraph.nodes[self.curr_tgt_id]["coords"]
            self.curr_tgt_zx_type = self.bgraph.nodes[self.curr_tgt_id]["zx_block"].zx_type

            # Announce edge
            if self._kwargs["debug"] > 0:
                print(
                    f"\n=> Edge: {u} ({self.bgraph.nodes[u]['zx_block'].zx_type}) --> {v} ({self.curr_tgt_zx_type}). {'CROSS' if self.cross_edge else 'STANDARD'}"
                )

            # Create a copy of taken that does not include source or target coordinates
            self.pruned_taken = self.taken.copy()
            self.pruned_taken.discard(self.curr_src_coords)
            self.pruned_taken.discard(self.curr_tgt_coords)

            # Prune beams so edge fulfillment starts with clean slate
            self.prune_beams()

            # Edge fulfillment: try finding paths using increasing max search distance
            self.call_pathfinder()

            if self.winner_path:
                # Add path to blockgraph
                self.add_path()
                if self._kwargs["debug"] >= 3 or self._kwargs["animate"]:
                    self.draw_blockgraph(is_final_vis=False)

            else:
                if self._kwargs["debug"] >= 1 or self._kwargs["animate"]:
                    self.draw_blockgraph(is_final_vis=False, iter_fail=True)
                raise ValueError(
                    f"ERROR. No path found for {'CROSS' if self.cross_edge else 'STANDARD'} edge: {u} --> {v}"
                )

            # Prune beams on exit for good health
            self.prune_beams()

            _, duration_iter = datetime_manager(t_1=self.t1_iter)
            # Update user if applicable
            if self._kwargs["debug"] > 0:
                print(
                    "Edge completed.",
                    f"Vol +: {len(self.winner_path.full_path) - (2 if (self.cross_edge or self.curr_tgt_zx_type == 'O') else 1)}.",
                    f"Duration: {duration_iter:.2f}s",
                )

            # Check if placement created a need for twins
            self.just_checked_twins = False
            if self._kwargs["twins"]:
                self.check_need_twins()
                if self.just_checked_twins:
                    self.check_need_twins()

        # Stop timer
        _, duration_total = datetime_manager(t_1=self.t1)

        # Final user updates if applicable
        print(
            f"\nSUCCESS! Habemus BlockGraph. Volume: {self.get_volume()}. Duration: {duration_total:.2f}s\n"
        )

    def call_pathfinder(self, twin_mode: bool = False):
        """Call the pathfinder algorithm for an arbitrary edge.

        Args:
            step: The maximum distance allowed for the specific search.
            twin_mode (optional): True if the current edge is part of a twin creation cycle.

        """

        # Clear iteration-specific trackers
        self.tent_coords: list[StandardCoord] = None
        self.valid_paths: dict[PositionedZXBlock, list[PositionedZXBlock]] = None
        self.winner_path: CandidatePath = None
        self.pathfinder_vis_data: tuple[Any] = None
        self.faux_edge: bool = self.curr_src_id in self.inputs and self.curr_tgt_id in self.inputs

        # Prepare meta-attributes for iteration
        step = 1
        self.z_bounds: dict[str, int | None] = {"min": None, "max": None}

        # Overload tentative coords generation if applicable
        overload = True if self.curr_tgt_zx_type in ["Y"] or self._kwargs["z_stretch"] else False
        overload = False if self._kwargs["graph_traverse_mode"] == "bfs-rows" else overload
        if (
            self._kwargs["graph_traverse_mode"] in ["bfs-cnots", "bfs-cnot-cycles", "tfs-cnots"]
            and self.curr_src_id in self.qubits
            and self.curr_tgt_id in self.qubits
            and (
                self.qubits[self.first_id]
                == self.qubits[self.curr_src_id]
                == self.qubits[self.curr_tgt_id]
            )
        ):
            overload = False
            step = self._kwargs["z_stretch"]

        # Set time constraints if applicable
        if self.curr_tgt_id in self.ante:
            floor_coords = [
                self.bgraph.nodes[cube_id]["coords"]
                for cube_id in self.ante[self.curr_tgt_id]
                if self.bgraph.nodes[cube_id]["coords"]
            ]
            self.z_bounds["min"] = max([c[2] for c in floor_coords]) if floor_coords else None
            step = max(
                step,
                get_manhattan(
                    self.curr_src_coords,
                    (self.curr_src_coords[0], self.curr_src_coords[1], self.z_bounds["min"]),
                ),
            )
        if self.curr_tgt_id in self.post:
            roof_coords = [
                self.bgraph.nodes[cube_id]["coords"]
                for cube_id in self.post[self.curr_tgt_id]
                if self.bgraph.nodes[cube_id]["coords"]
            ]
            self.z_bounds["max"] = max([c[2] for c in roof_coords]) if roof_coords else None

        # Loop until path is found
        if not step or step == 0:
            step = 1
        max_step = step + 10
        while step < max_step:
            # Get many tentative coordinates or set a specific target coordinate
            self.tent_coords = (
                [self.curr_tgt_coords]
                if self.cross_edge
                else gen_tent_tgt_coords(
                    self.curr_src_coords,
                    step,
                    self.taken,
                    overload=overload if step < 3 else False,
                    z_bounds=self.z_bounds,
                )
            )

            # Try finding paths to each tentative coordinate
            if self.tent_coords:
                # Get a number of valid paths (topologically correct, not necessarily optimal)
                for iter_graph_bounds in [self.bounds, None]:
                    if (
                        not (self.cross_edge or self.curr_tgt_zx_type in ["Y", "T", "O", "XZ"])
                        and not iter_graph_bounds
                    ):
                        continue
                    self.valid_paths, self.pathfinder_vis_data = pathfinder(
                        self.bgraph,
                        self.beams,
                        self.beams_short,
                        self.curr_src_id,
                        self.curr_tgt_id,
                        self.tent_coords,
                        self.cross_edge,
                        self.taken,
                        self.pruned_taken,
                        self.is_hadamard,
                        self.z_bounds,
                        graph_bounds=iter_graph_bounds,
                        **self._kwargs,
                    )
                    if self.valid_paths:
                        break

            # Kill loop if cross edge fails
            if self.cross_edge and not self.valid_paths:
                raise ValueError("Failed to find cross edge.")

            # Handle cross edge
            elif self.cross_edge and len(self.valid_paths) == 1:
                self.winner_path = CandidatePath(
                    **{
                        "full_path": list(self.valid_paths.values())[0],
                        "tgt_beams": self.beams[self.curr_tgt_id],
                        "tgt_beams_short": self.beams_short[self.curr_tgt_id],
                        "beams_broken_by_path": 0,  # Not calculated (pathfinder handles beams tolerances internally for cross-edges)
                        "tgt_unobstr_exit_n": self.bgraph.nodes[self.curr_tgt_id]["completions"][
                            "pending"
                        ]
                        - 1,  # Not calculated (pathfinder broken exits internally for cross-edges)
                    }
                )

            # Pick between valid paths
            elif self.valid_paths:
                for valid_path in self.valid_paths.values():
                    # Extract key path information
                    tgt_coords, tgt_zx_block = valid_path[-1]
                    coords_in_path = [c for c, _ in valid_path][1:]

                    # Re-assign last block in sequence if target is a boundary
                    if self.curr_tgt_zx_type == "O":
                        tgt_zx_block = ZXBlockRegistry.get_create(kind="OOO")
                        valid_path[-1] = (tgt_coords, tgt_zx_block)

                    # Check if exits are unobstructed
                    tgt_unobstr_exit_n, self.tgt_beams, self.tgt_beams_short = (
                        check_exits_add_beams(
                            tgt_zx_block,
                            tgt_coords,
                            self.taken,
                            coords_in_path,
                            self._kwargs["beams_len_short"],
                        )
                    )

                    # Continue if minimum required number of exits available for target
                    # Note. Open boundaries typically are part of a computation, so leave one exit open
                    min_tgt_unobstr_exit_n = (
                        1
                        if self.faux_edge
                        else self.bgraph.nodes[self.curr_tgt_id]["completions"]["pending"] - 1
                    )
                    if tgt_unobstr_exit_n >= min_tgt_unobstr_exit_n:
                        # Check if path breaks more beams than tolerable
                        extra_allowance = 0
                        if (self.curr_src_id, self.curr_tgt_id) in self.edge_queue:
                            if not self.edge_queue.index(
                                (self.curr_src_id, self.curr_tgt_id)
                            ) + 1 == len(self.edge_queue):
                                nxt_edge = self.edge_queue[
                                    self.edge_queue.index((self.curr_src_id, self.curr_tgt_id)) + 1
                                ]
                                nxt_id = nxt_edge[1]
                                nxt_coords = self.bgraph.nodes[self.twin_trace[nxt_id][-1]][
                                    "coords"
                                ]
                                if nxt_coords and (self.curr_tgt_id, nxt_id) in self.bgraph.edges:
                                    md = get_manhattan(tgt_coords, nxt_coords)
                                    if md == 1:
                                        move = tuple(np.array(nxt_coords) - np.array(tgt_coords))
                                        nxt_zx_block = self.bgraph.nodes[nxt_id]["zx_block"]
                                        if tgt_zx_block.kind and nxt_zx_block.kind:
                                            exits_match = tgt_zx_block.cube_open_faces_match(
                                                move, tgt_zx_block=nxt_zx_block
                                            )
                                            faces_match = tgt_zx_block.face_match(
                                                move, nxt_zx_block
                                            )
                                            if exits_match and faces_match:
                                                extra_allowance = 1

                        beam_clashes, beams_broken_by_path = self.check_beams(
                            coords_in_path, twin_mode=twin_mode, extra_allowance=extra_allowance
                        )

                        # Append path to viable paths if path clears all checks
                        if not beam_clashes or self.faux_edge:
                            # Consolidate path data
                            candidate_path = CandidatePath(
                                **{
                                    "full_path": valid_path,
                                    "tgt_beams": self.tgt_beams,
                                    "tgt_beams_short": self.tgt_beams_short,
                                    "beams_broken_by_path": beams_broken_by_path,
                                    "tgt_unobstr_exit_n": tgt_unobstr_exit_n,
                                }
                            )

                            # Append to viable paths
                            self.winner_path = (
                                candidate_path
                                if (
                                    not self.winner_path
                                    or self.value_function(candidate_path)
                                    > self.value_function(self.winner_path)
                                )
                                else self.winner_path
                            )
            # Break if valid paths generated at step
            if self.winner_path:
                break

            # Increase distance if no valid paths found at current step
            step += 1

    def add_path(self):
        """Add a winner path to the main blockgraph."""

        # Add path
        self.bgraph, self.taken, self.intermediate_ids = add_path_to_bgraph(
            self.bgraph,
            self.taken,
            self.winner_path.full_path.copy(),
            self.curr_src_id,
            self.curr_tgt_id,
            self.is_hadamard,
            self.cross_edge,
        )

        # Add to completed
        in_zx_edge = (
            self.twin_trace_inverse[self.curr_src_id]
            if self.curr_src_id in self.twin_trace_inverse
            else self.curr_src_id,
            self.twin_trace_inverse[self.curr_tgt_id]
            if self.curr_tgt_id in self.twin_trace_inverse
            else self.curr_tgt_id,
        )
        self.completed_in_zx_edges[in_zx_edge] = [
            self.curr_src_id,
            *list(self.intermediate_ids),
            self.curr_tgt_id,
        ]
        self.completed_base_edges[in_zx_edge] = [
            self.curr_src_id,
            *list(self.intermediate_ids),
            self.curr_tgt_id,
        ]

        # Add target beams
        self.beams[self.curr_tgt_id] = self.winner_path.tgt_beams
        self.beams_short[self.curr_tgt_id] = self.winner_path.tgt_beams_short

        # Add to time dependencies if path touches a cube with a related constraint
        if self.intermediate_ids:
            for k, v in self.ante.items():
                if self.curr_src_id in v and self.curr_tgt_id in v:
                    self.ante[k].update(self.intermediate_ids)

            self.build_rebuild_post_deps()

    def check_beams(
        self,
        coords_in_path: list[StandardCoord],
        strict: bool = False,
        twin_mode: bool = False,
        extra_allowance: bool = False,
    ) -> tuple[bool, int]:
        """Determine if target placement triggers critical multi-beam clashes.

        Args:
            coords_in_path: All coords in the current path.
            strict (optional): Whether to perform a strict or loose check.
            twin_mode (optional): True if the current edge is part of a twin creation cycle.
            extra_allowance (optional): Number of extra beams that can be ignored beyond threshold.

        Returns:
            clash: False if no critical beam clashed found, else True.
            beams_broken_by_path: Accumulated total number of beams for which path creates some kind of problem.

        """

        # Full check for all other
        clash, beams_broken_by_path = check_beam_clashes(
            self.bgraph,
            self.tgt_beams,
            self.tgt_beams_short,
            self.beams,
            self.beams_short,
            self.curr_src_id,
            self.curr_tgt_id,
            coords_in_path,
            strict=strict,
            twin_mode=twin_mode,
            extra_allowance=extra_allowance,
        )

        return clash, beams_broken_by_path

    def check_boundary_beams(self):
        """Ensure boundaries have beams that match their input/output status."""
        if self.curr_tgt_zx_type == "O":
            if self.curr_tgt_id in self.inputs:
                return any([True for beam in self.tgt_beams if beam.z.direction == -1])
            if self.curr_tgt_id in self.outputs:
                return any([True for beam in self.tgt_beams if beam.z.direction == 1])
        return True

    def check_need_twins(self, strict: bool = True):
        """Determine if there is a need to create twins for any given target.

        Args:
            strict (optional): Whether to perform a strict or loose check.

        """

        self.ids_to_twin = check_beam_clashes_for_twins(
            self.bgraph,
            self.beams,
            self.beams_short,
            self.curr_src_id,
            self.curr_tgt_id,
            self.taken,
            ids_to_twin=self.ids_to_twin,
            strict=True,
        )

        while self.ids_to_twin:
            self.add_twins()
            self.just_checked_twins = not self.just_checked_twins

    def add_twins(self):
        """Create a twin spider for any given number of priority spiders."""

        # Start timer
        self.t1_iter, _ = datetime_manager()

        # Announce need for twins
        if self._kwargs["debug"] > 0:
            print(f"\n=> Twins needed for IDs: {self.ids_to_twin}")

        for original_id in self.ids_to_twin:
            # Define new ID
            twin_id = max(self.bgraph.nodes) + 1
            if original_id in self.twin_trace_inverse:
                self.twin_trace_inverse[twin_id] = self.twin_trace_inverse[original_id]
            else:
                self.twin_trace_inverse[twin_id] = original_id
            if original_id in self.twin_trace:
                self.twin_trace[original_id].append(twin_id)
            else:
                first_original_id = [
                    k for k, twin_ids in self.twin_trace.items() if original_id in twin_ids
                ][0]
                self.twin_trace[first_original_id].append(twin_id)

            # Get original node info
            parent_zx_block = self.bgraph.nodes[original_id]["zx_block"]

            # Add the bare twin
            self.bgraph.add_node(
                twin_id,
                zx_block=ZXBlockRegistry.get_create(zx_type=parent_zx_block.zx_type),
                coords=None,
                completions={
                    "degree": None,
                    "pending": None,
                },
            )

            # Add twin to list of BEFORE/AFTER time dependencies if applicable
            temp_time_deps: dict[int, set[int]] = {}
            for k, v in self.ante.items():
                if original_id in v:
                    if k not in temp_time_deps:
                        temp_time_deps[k] = [twin_id]
                    else:
                        temp_time_deps[k].append(twin_id)

            for k, v in temp_time_deps.items():
                if k in self.ante:
                    self.ante[k].update(v)

            self.build_rebuild_post_deps()

            # Get neighbours pending for original and transfer to twin
            original_pending_neighs = [
                n
                for n in self.bgraph.neighbors(original_id)
                if self.bgraph.get_edge_data(n, original_id)["kind"] is None
            ]

            # Connect original and twin
            self.bgraph.add_edge(
                original_id,
                twin_id,
                edge_type="SIMPLE",
                start_coords=None,
                end_coords=None,
                kind=None,
            )

            # Remove pending neighbours from original node and transfer to new twin
            for pending_id in original_pending_neighs:
                edge_type = self.bgraph.get_edge_data(original_id, pending_id)["edge_type"]
                self.bgraph.add_edge(
                    pending_id,
                    twin_id,
                    edge_type=edge_type,
                    start_coords=None,
                    end_coords=None,
                    kind=None,
                )
                self.bgraph.remove_edge(original_id, pending_id)

            # Update pending information for original and twin
            self.bgraph.nodes[original_id]["completions"] = {
                "degree": self.bgraph.degree(original_id),
                "pending": 1,
            }
            self.bgraph.nodes[twin_id]["completions"] = {
                "degree": self.bgraph.degree(twin_id),
                "pending": self.bgraph.degree(twin_id),
            }

            # Place twin
            # Internalise key edge characteristics (ease of access)
            self.curr_src_id, self.curr_tgt_id = (original_id, twin_id)
            self.cross_edge = False
            self.is_hadamard = False
            self.curr_src_coords = self.bgraph.nodes[original_id]["coords"]
            self.curr_tgt_coords = self.bgraph.nodes[twin_id]["coords"]
            self.curr_tgt_zx_type = self.bgraph.nodes[twin_id]["zx_block"].zx_type

            # Create a copy of taken that does not include source or target coordinates
            self.pruned_taken = self.taken.copy()
            self.pruned_taken.discard(self.curr_src_coords)
            self.pruned_taken.discard(self.curr_tgt_coords)

            # Prune beams so edge fulfillment starts with clean slate
            self.prune_beams()

            # Edge fulfillment: try finding paths using increasing max search distance
            self.call_pathfinder(twin_mode=True)

            if self.winner_path:
                # Add path to blockgraph
                self.add_path()
                if self._kwargs["debug"] >= 3 or self._kwargs["animate"]:
                    self.draw_blockgraph(is_final_vis=False)
            else:
                if self._kwargs["debug"] >= 3 or self._kwargs["animate"]:
                    self.draw_blockgraph(is_final_vis=False, iter_fail=True)
                raise ValueError(
                    f"ERROR. No path found for {'CROSS' if self.cross_edge else 'STANDARD'} edge: {self.curr_src_id} --> {self.curr_tgt_id}"
                )

            # Prune beams on exit for good health
            self.prune_beams()

            _, duration_iter = datetime_manager(t_1=self.t1_iter)

            # Update user if applicable
            if self._kwargs["debug"] > 0:
                print(
                    f"Twin added succesfully: {self.curr_src_id} --> {self.curr_tgt_id}.",
                    f"Vol +: {len(self.winner_path.full_path) - 1}.",
                    f"Duration: {duration_iter:.2f}s",
                )

        # Reset IDs to twin
        self.ids_to_twin = set()

        # Update user if applicable
        if self._kwargs["debug"] > 0:
            print("=> TWINS round complete.\n")

    def prune_beams(self):
        """Prune beams eliminating broken beams and beams of completed nodes."""
        self.beams, self.beams_short = prune_beams(
            self.bgraph, self.beams, self.beams_short, self.taken
        )

    def value_function(self, candidate_path: CandidatePath):
        """Weigh value of a candidate path against other BlockGraph characteristics.

        Args:
            candidate_path: A path under consideration for adding to the blockgraph.

        Returns:
            path_value: The weighed value of a path.

        """

        # Extract or define key parameters
        ## Weights
        path_len_w, beams_broken_w = self._kwargs["weights"]

        ## Last candidate path
        last_coords, last_zx_block = candidate_path.full_path[-1]

        # Penalise length
        len_contrib = len(candidate_path.full_path) * path_len_w

        # Penalise number of beams broken by path
        broken_beams_contrib = candidate_path.beams_broken_by_path * beams_broken_w

        # Push on Z-axis and graph bounds
        out_of_bounds, z_push = (0, 0)
        if self._kwargs["z_stretch"] or last_zx_block.zx_type in ["Y", "T"] or self.faux_edge:
            # Define weight
            stretch_multiplier = self._kwargs["z_stretch"] if self._kwargs["z_stretch"] else 1

            # Push down for Y-cubes and cultivation/distillation
            if last_zx_block.zx_type in ["Y", "T"]:
                z_push = -1 * last_coords[2]
            # Favour row difference for all other cubes
            else:
                row_diff = 0
                if self.curr_tgt_id in self.rows and self.curr_src_id in self.rows:
                    row_diff = self.rows[self.curr_tgt_id] - self.rows[self.curr_src_id]
                z_push = (
                    (row_diff * stretch_multiplier * last_coords[2]) if not self.faux_edge else 0
                )

            # Apply bounds if given
            if not self.faux_edge and self.bounds.x and self.bounds.y:
                coords_in_path = np.array([c for c, _ in candidate_path.full_path])
                x_coords = coords_in_path[:, 0]
                y_coords = coords_in_path[:, 1]
                x_out = [x < 0 or x > self.bounds.x for x in x_coords]
                y_out = [y < 0 or y > self.bounds.y for y in y_coords]
                out_of_bounds = (sum(x_out) + sum(y_out)) * -stretch_multiplier

        # Gravity around a specific point in existing blockgraph if applicable
        pull_to_centre = 0
        if self._kwargs["gravity"]:
            # Override push on Z for faux edges
            if self.faux_edge:
                z_push = -100 * abs(last_coords[2])

            # Find centre
            # Aim for centremost point of graph
            centre_coords = np.sum([np.array(coords) for coords in self.taken], axis=0) / len(
                self.taken
            )

            # !!! TO BE IMPLEMENTED
            # IN THEORY, THE CENTRE POINT CAN CHANGE AS GRAPH EVOLVES

            # Push towards neighbour if next edge is a crosss edge
            nxt_is_cross = False
            curr_edge = (self.curr_src_id, self.curr_tgt_id)
            if curr_edge in self.edge_queue and not self.edge_queue.index(curr_edge) + 1 == len(
                self.edge_queue
            ):
                nxt_id = self.edge_queue[self.edge_queue.index(curr_edge) + 1][1]
                nxt_coords = self.bgraph.nodes[self.twin_trace[nxt_id][-1]]["coords"]
                if nxt_coords and (self.curr_tgt_id, nxt_id) in self.bgraph.edges:
                    centre_coords = nxt_coords
                    nxt_is_cross = True

            if (
                nxt_is_cross
                or self.faux_edge
                or (
                    self._kwargs["graph_traverse_mode"] == "bfs-cycles"
                    and self.curr_tgt_zx_type != "O"
                )
            ):
                d_to_centre = np.linalg.norm(np.array(last_coords) - np.array(centre_coords))
                pull_to_centre = d_to_centre * -10 * self._kwargs["gravity"]
            elif self.curr_tgt_zx_type not in ["O", "Y", "XZ", "T"]:
                centre_x, centre_y, _ = centre_coords
                x, y, _ = last_coords
                pull_to_centre = -self._kwargs["gravity"] * (abs(x - centre_x) + abs(y - centre_y))

        # Return cumulative value
        path_value = len_contrib + broken_beams_contrib + out_of_bounds + z_push + pull_to_centre
        return path_value

    def get_stats(self) -> tuple[int, int]:
        """Calculate key build stats."""

        # Incoming ZX graph
        in_zx_spiders: int = len(self.in_ids)
        in_zx_edges: int = len(self.in_edge_types)
        in_zx_density: float = in_zx_edges / in_zx_spiders

        # BGRAPH 101
        real_coords = np.array(
            [
                attrs["coords"]
                for _, attrs in self.bgraph.nodes(data=True)
                if attrs["coords"] and attrs["zx_block"].zx_type != "O"
            ]
        )
        bgraph_volume: int = len(real_coords)
        bgraph_overhead: float = bgraph_volume / in_zx_spiders

        # Hardware footprint
        bgraph_surface_footprint: dict[int, tuple[int, int]] = {}
        if real_coords.size > 0:
            min_x, max_x = real_coords[:, 0].min(), real_coords[:, 0].max()
            min_y, max_y = real_coords[:, 1].min(), real_coords[:, 1].max()
            x_span = abs(max_x - min_x) + 1
            y_span = abs(max_y - min_y) + 1
            bgraph_surface_footprint = {
                d: (x_span * d, y_span * d) for d in range(3, 17) if d % 2 != 0
            }

        self.stats: dict[str, int | float | dict[int, tuple[int, int]]] = {
            "in_zx_spiders": in_zx_spiders,
            "in_zx_edges": in_zx_edges,
            "in_zx_density": in_zx_density,
            "bgraph_volume": bgraph_volume,
            "bgraph_overhead": bgraph_overhead,
            "bgraph_surface_footprint": bgraph_surface_footprint,
        }

    def draw_zx(self, draw_style: str = "zx"):
        """Draw the NX blockgraph using ZX or NX styling.

        draw_style: The style of drawing:
            zx: Positions nodes in a PyZX-like manner
            nx: Positions nodes using NX algorithms (defaults to spectral layout).


        """
        draw_as_zx(
            self.bgraph,
            self.qubits,
            self.rows,
            draw_style=draw_style,
            first_spider=self.first_id if draw_style == "nx-bfs" else None,
        )

    def draw_blockgraph(self, is_final_vis: bool = True, iter_fail: bool = False):
        """Draw the NX blockgraph using ZX or NX styling.

        Args:
            is_final_vis: Boolean to flag if current visualisation is the final blockgraph.
            iter_fail: Boolean to flag if current visualisation comes right after an iteration failure.

        """

        # Pack the input ZX as NX into a single object
        in_zx = {
            "ids": self.in_ids,
            "qubits": self.in_qubits,
            "rows": self.in_rows,
            "inputs": self.in_inputs,
            "outputs": self.in_outputs,
            "phases": self.in_phases,
            "edge_types": self.in_edge_types,
            "degrees": self.in_degrees,
            "types": self.in_types,
            "completed_edges": self.completed_in_zx_edges,
        }

        # Pack the base NX after transformations into a single object
        base_graph = {
            "ids": self.ids,
            "qubits": self.qubits,
            "rows": self.rows,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "phases": self.phases,
            "edge_types": self.edge_types,
            "degrees": self.degrees,
            "types": self.types,
            "completed_edges": self.completed_base_edges,
        }

        base_graph_draw_styles = {
            "bfs": "zx",
            "bfs-cross": "zx",
            "bfs-cross-boundaries-last": "nx-fruchterman_reingold",
            "bfs-cycles": "nx-fruchterman_reingold",
            "bfs-rows": "zx",
            "bfs-cnots": "zx",
            "bfs-cnot-cycles": "zx",
            "tfs-cnots": "zx",
        }

        self.get_stats()
        pop_vis = self._kwargs["debug"] > 1 or is_final_vis
        vis_state = VisualiserState(
            self.bgraph,
            self.beams,
            self.beams_short,
            self.tent_coords,
            [] if is_final_vis else [self.curr_src_id, *self.intermediate_ids, self.curr_tgt_id],
            self.cross_edge,
            in_zx,
            base_graph,
            self.twin_trace,
            is_final_vis=is_final_vis,
            iter_fail=iter_fail,
            block_style="pipe",
            base_graph_draw_style=base_graph_draw_styles[self._kwargs["graph_traverse_mode"]],
            stats=self.stats,
            vis_mode=(pop_vis, self._kwargs["animate"]),
        )

        visualiser = BlockGraphVisualiser(vis_state)
        visualiser.build_layout()
        visualiser.show()

    def write_bgraph(
        self,
        output_dir: Path | str = BGRAPH_DIR,
        circuit_name: str = "qc",
    ):
        """Write BlockGraph to a BGRAPH file.

        Args:
            output_dir (optional): The path to the directory where BGRAPH file should be saved.
            circuit_name (optional): The name of the circuit.

        """
        # Create output directory if it doesn't exist.
        if not isinstance(output_dir, Path):
            try:
                output_dir = Path(str(output_dir))
            except Exception as e:
                raise NotADirectoryError(
                    f"Unable to create output directory: '{output_dir}'"
                ) from e
        os.makedirs(output_dir, exist_ok=True)

        # Write to BGRAPH file
        path_to_output_file = output_dir / f"{circuit_name}.bgraph"

        with open(path_to_output_file, "w") as f:
            f.write("BLOCKGRAPH 0.1.0;\n")

            f.write("\nMETADATA: attr_name; value;\n")
            f.write("source; topologiq;\n")
            f.write(f"circuit_name; {circuit_name};\n")

            f.write("\nCUBES: index;x;y;z;kind;label;\n")

            for n_id, attrs in self.bgraph.nodes(data=True):
                # Get coords and kind
                if attrs["coords"] and attrs["zx_block"]:
                    x, y, z = attrs["coords"]
                    kind = attrs["zx_block"].kind
                else:
                    x, y, z = (None, None, None)
                    kind = ""

                # Re-write kind into BGRAPH standard if applicable
                if kind in ["YYO", "TTO"]:
                    neighs = list(self.bgraph.neighbors(n_id))
                    if len(neighs) > 1:
                        raise ValueError("Error writing BGRAPH. Malformed Y cube.")
                    _, _, neigh_z = self.bgraph.nodes(data=True)[neighs[0]]["coords"]
                    if kind == "YYOO":
                        kind = "Yi" if neigh_z < z else "Ym"
                    if kind == "TTO":
                        if neigh_z <= z:
                            raise ValueError(
                                "Error writing BGRAPH. Malformed cultivation or distillation cube."
                            )
                        neigh_kind = self.bgraph.nodes(data=True)[neighs[0]]["zx_block"].kind
                        kind = neigh_kind[:2] + "t"

                # Assemble label
                label = ""
                if n_id in self.inputs:
                    label = f"in_{self.qubits[n_id]}" if n_id in self.qubits else "in"
                if n_id in self.outputs:
                    label = f"out_{self.qubits[n_id]}" if n_id in self.qubits else "out"

                # Write
                f.write(f"{n_id};{x!s};{y!s};{z!s};{kind};{label};\n")

            f.write("\nPIPES: src;tgt;kind;\n")
            f.writelines(
                [f"{u!s};{v!s};{kind};\n" for u, v, kind in self.bgraph.edges(data="kind")]
            )

    def animate(self, filename_prefix: str = "computation"):
        """Call animation sequence."""
        if self._kwargs["animate"]:
            create_animation(
                MEDIA_DIR / "temp",
                MEDIA_DIR / "animations",
                filename_prefix=filename_prefix,
                format=self._kwargs["animate"],
            )

        self.cleanup()

    def cleanup(self):
        """Carry out cleanup operations after build."""
        # Delete temporary files
        rm_temp_files(MEDIA_DIR / "temp")
