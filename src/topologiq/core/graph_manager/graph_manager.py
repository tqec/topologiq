"""Primary graph management class and related operations, incl. the outer graph manager BFS.

This file contains classes and functions that altogether create Topologiq's core blockgraph,
determine the first spider to process, the order in which subsequent spiders get processed,
and calls the inner pathfinder algorithm as appropriate per edge.

Usage:
    Create an BlockGraphManager from a separate script and use its methods as appropriate.

"""

import itertools
import os
import random
from fractions import Fraction
from itertools import chain
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pyzx as zx
from numpy.typing import NDArray
from pyzx.pauliweb import PauliWeb

from topologiq.core.beams import CubeBeams
from topologiq.core.blocks import CandidatePath, PositionedZXBlock, ZXBlock, ZXBlockRegistry
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
from topologiq.core.pathfinder.pathfinder import PathfinderInitState, PathFinderManager
from topologiq.core.pathfinder.symbolic import check_exits_add_beams
from topologiq.input.zx_manager import AugmentedZXGraph
from topologiq.utils.classes import GraphBounds, StandardCoord
from topologiq.utils.file import rm_temp_files
from topologiq.utils.manhattan import get_manhattan
from topologiq.utils.time import datetime_manager
from topologiq.utils.zx import ZXEdgeTypes, ZXTypes
from topologiq.vis.animate import create_animation
from topologiq.vis.draw import BlockGraphVisualiser, VisualiserState

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
    """Manage the primary BlockGraph and the process of building it."""

    def __init__(
        self,
        aug_zx: AugmentedZXGraph,
        graph_name: str = "computation",
        **kwargs,
    ):
        """Initialise with empty blockgraph."""

        # Delete any temporary files that may have survived previous builds
        rm_temp_files(MEDIA_DIR / "temp")

        # Direct assignations
        self.graph_name: str = graph_name
        self.aug_zx: AugmentedZXGraph = aug_zx

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
        self.in_ids: set[int] = self.aug_zx.zx_graph.vertex_set()
        self.in_qubits: dict[int, int] = self.aug_zx.zx_graph.qubits()
        self.in_rows: dict[int, int] = self.aug_zx.zx_graph.rows()
        self.in_inputs: list[int] = self.aug_zx.zx_graph.inputs()
        self.in_outputs: list[int] = self.aug_zx.zx_graph.outputs()
        self.in_phases: dict[int, int | Fraction] = self.aug_zx.zx_graph.phases()
        self.in_edge_types: dict[tuple[int, int], str] = {
            edge_id: ZXEdgeTypes(self.aug_zx.zx_graph.edge_type(edge_id)).name
            for edge_id in self.aug_zx.zx_graph.edges()
        }
        self.in_degrees: dict[int, int] = {
            k: len(self.aug_zx.zx_graph.neighbors(k)) for k in self.aug_zx.zx_graph.vertex_set()
        }

        self.in_types: dict[int, str] = {}
        for k, v in self.aug_zx.zx_graph.types().items():
            switch_to_cube = self.aug_zx.zx_graph.vdata(k, "switch_to_cube", default=None)
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
        self.intermediate_ids: list[int] = []

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
        self.cross_edge: bool = False
        self.tent_coords: list[StandardCoord] = None
        self.completed_base_ids: list[int] = []
        self.completed_base_edges: dict[tuple[int, int], list[int]] = {}
        self.ante: dict[int, set[int]] = {}
        self.post: dict[int, set[int]] = {}
        self.twin_trace: dict[int, list[int]] = {}

        # Other trackers
        self.run_success: bool = False

        # Initialise empty NX graph
        self.bgraph: nx.Graph = nx.Graph()

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
            for n_id in self.aug_zx.zx_graph.vertices()
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
        self.handle_s_t_spiders()

        # Calculate boundaries of theoretical chip surface
        self.get_bounds()

    def handle_s_t_spiders(self):
        """Convert spiders with phases to Y- and T-cube patterns."""

        # Trackers for special cubes introduced in this method
        self.y_cubes: dict[int, str] = {}
        self.s_gates: dict[int, list[tuple[int, int] | None]] = {}
        self.msc_cubes: dict[int, str] = {}
        self.msc_stretch: dict[int, int] = {}
        self.t_gates: dict[int, list[tuple[int, int] | None]] = {}
        self.t_zx_tracker: dict[int, int] = {}
        self.msc_factory: dict[int, StandardCoord] = {}

        # Look over nodes with phases
        for spider_id, phase in self.phases.items():
            if phase == Fraction(1, 2):
                _handle_s_spider(
                    spider_id,
                    self.bgraph,
                    self.phases,
                    self.y_cubes,
                    self.s_gates,
                    self.qubits,
                    self.rows,
                )

            if phase == Fraction(1, 4):
                _handle_t_spider(
                    spider_id,
                    self.bgraph,
                    self.phases,
                    self.msc_cubes,
                    self.t_gates,
                    self.t_zx_tracker,
                    self.ante,
                    self.qubits,
                    self.rows,
                    self.inputs,
                )

        # Add time dependencies
        if self.msc_cubes:
            self.msc_stretch = {k: 3 for k in self.msc_cubes}
        if self.t_gates:
            try:
                self.build_deps_from_pauli_webs()
            except Exception as e:
                print(f"Error calculating time constraints for T-gates: {e}.")

        # Update any trackers that might have changed from transformations
        self.update_base_trackers()

    def build_deps_from_pauli_webs(self):
        """Build a dictionary of cubes with ANTECESSORs."""
        # Proceed only if Pauli Webs exist

        # Extract Pauli webs from input AugZXGraph
        order, zwebs, xwebs = self.aug_zx.pauli_pack

        # Proceed if Pauli webs are available
        if order and zwebs and xwebs:
            # Build ANTE dependencies
            _prep_ante(self.ante, self.t_zx_tracker, order, zwebs, xwebs)
            # Build inverse AFTER dependencies

            self.build_rebuild_post_deps()

    def build_rebuild_post_deps(self):
        """Build a dictionary containing any cubes with DESCENDANTs."""
        # Proceed only if ANTE dependencies exist
        if self.ante:
            _prep_post(self.post, self.ante)

    def get_bounds(self):
        """Define the max/min bounds for the chip surface."""

        self.bounds = GraphBounds()
        if "size_of_chip" in self._kwargs and "k" in self._kwargs:
            self.bounds.x = int(self._kwargs["size_of_chip"][0] / (self._kwargs["k"])) - 1
            self.bounds.y = int(self._kwargs["size_of_chip"][1] / (self._kwargs["k"])) - 1

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

            # Update any trackers that might have changed from transformations
            self.update_base_trackers(new_ids=new_ids)

    def update_base_trackers(self, new_ids: dict[int, int] = {}):
        """Update ID trackers to match changes in input graph."""

        if new_ids:
            self.ids.update(new_ids.keys())
            for new_id, ref_id in new_ids.items():
                self.rows[new_id] = self.rows[ref_id]
                self.qubits[new_id] = self.qubits[ref_id]
                self.types[new_id] = self.types[ref_id]
                self.phases[new_id] = 0
        self.ids = set(self.bgraph.nodes())
        self.types = {k: zx_block.zx_type for k, zx_block in self.bgraph.nodes(data="zx_block")}
        self.degrees = {k: self.bgraph.degree(k) for k in self.bgraph.nodes()}
        self.edge_types = {
            (u, v): attrs["edge_type"] for u, v, attrs in self.bgraph.edges(data=True)
        }
        self.twin_trace: dict[int, list[int]] = {k: [k] for k in self.ids}
        self.twin_trace_inverse: dict[int, int] = {}

    def place_first_cube(self):
        """Define and place the very first cube of the blockgraph."""

        # Reset trackers innaplicable to single cube placement
        if self._kwargs["debug"] > 0:
            print("* First cube.")
        self.cross_edge = False
        self.is_hadamard = False

        # Pick and ID and kind for first cube
        self.first_coords = (0, 0, 0)
        self.first_id, self.other_first_ids, self.first_zx_block, src_beams, src_beams_short = (
            get_first_cube_data(
                self.bgraph,
                self._kwargs["first_id_strategy"],
                self.qubits,
                self.inputs,
                self._kwargs["beams_len_short"],
                first_coords=self.first_coords,
                override_first_cube=self._kwargs["first_cube"],
                s_gates=self.s_gates,
                t_gates=self.t_gates,
                random_seed=self._kwargs["seed"],
            )
        )

        # Update corresponding node
        self.bgraph.nodes[self.first_id]["coords"] = self.first_coords
        self.bgraph.nodes[self.first_id]["zx_block"] = self.first_zx_block
        self.beams[self.first_id] = src_beams
        self.beams_short[self.first_id] = src_beams_short

        # Update taken
        self.taken.add(self.first_coords)

        # Update user if applicable
        if self._kwargs["debug"] > 0:
            print(
                f"  - Completed. ID: {self.first_id}. Kind: {self.first_zx_block.kind}, Coords: {self.first_coords}."
            )

    def get_queue(self):
        """Build queue using strategy defined in kwargs."""
        self.edge_queue = queue_precalc(
            self.bgraph,
            self.first_id,
            self.rows,
            self.qubits,
            self.inputs,
            self.outputs,
            self._kwargs["graph_traverse_mode"],
            other_first_ids=self.other_first_ids,
            t_gates=self.t_gates,
        )

    def build(self, override_kwargs: dict = {}):
        """Build the blockgraph using pre-defined edge queue."""

        # Announce build
        print("\n=> Building BlockGraph.")

        # Start timer
        self.t1, _ = datetime_manager()

        # Override KWARGs if needed
        if override_kwargs:
            self._kwargs = check_assemble_kwargs(**override_kwargs)
            self._kwargs["debug"] = 1

        # Place the first cube at centre of available space
        self.place_first_cube()

        # Pre-calculate edge_queue based on graph traverse strategy
        self.get_queue()

        # Re-initialise trackers to keep trace of any changes ahead of build
        self.update_base_trackers()

        # Rewrite pending information to reflect any changes introduced by graph rewrites
        for spider_id in self.bgraph.nodes():
            self.bgraph.nodes[spider_id]["completions"]["degree"] = self.bgraph.degree(spider_id)
            self.bgraph.nodes[spider_id]["completions"]["pending"] = self.bgraph.degree(spider_id)

        # Loop through edge queue
        for raw_u, raw_v in self.edge_queue:
            # Start iteration timer
            self.t1_iter, _ = datetime_manager()

            # Get (u, v) from ID trace (if ID has no twins, the last twin is itself)
            u = self.twin_trace[raw_u][-1]
            v = self.twin_trace[raw_v][-1]

            # Skip if edge has already been resolved
            if (raw_u, raw_v) in self.completed_base_edges or (
                raw_v,
                raw_u,
            ) in self.completed_base_edges:
                continue

            # Ensure current source was placed in a prior iteration.
            if (
                self.bgraph.nodes[u]["zx_block"].kind is None
                or self.bgraph.nodes[u]["coords"] is None
            ):
                raise ValueError(f"Failed to launch: {u} --> {v}. Source not yet placed.")

            # Internalise key edge characteristics & clear iteration-specific parameters
            self.curr_src_id, self.curr_tgt_id = (u, v)
            self.clear_iter(u, v)

            # Announce edge
            if self._kwargs["debug"] > 0:
                print(
                    f"* Edge: {u} ({self.bgraph.nodes[u]['zx_block'].zx_type}) --> {v} ({self.curr_tgt_zx_type}). {'CROSS' if self.cross_edge else 'STANDARD'}"
                )

            # Edge fulfillment
            if not self.cross_edge:
                call_pathfinder_bfs_std(self)
            else:
                call_pathfinder_bfs_cross(self)

            # Add path to blockgraph or end process as failure
            if self.winner_path:
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
                    "  - Completed.",
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
            "\n=> SUCCESS! Habemus BlockGraph.",
            f"\n  - Volume: {self.get_volume()}.",
            f"\n  - Duration: {duration_total:.2f}s.\n",
        )

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

    def check_need_twins(self):
        """Determine if there is a need to create twins for any given target."""
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
            print(f"* Entering TWIN mode. IDs to twin: {self.ids_to_twin}")

        for orig_id in self.ids_to_twin:
            # Define new ID
            twin_id = max(self.bgraph.nodes) + 1
            if orig_id in self.twin_trace_inverse:
                self.twin_trace_inverse[twin_id] = self.twin_trace_inverse[orig_id]
            else:
                self.twin_trace_inverse[twin_id] = orig_id
            if orig_id in self.twin_trace:
                self.twin_trace[orig_id].append(twin_id)
            else:
                first_orig_id = [
                    k for k, twin_ids in self.twin_trace.items() if orig_id in twin_ids
                ][0]
                self.twin_trace[first_orig_id].append(twin_id)

            # Get original node info
            parent_zx_block = self.bgraph.nodes[orig_id]["zx_block"]

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

            # Add twin to row & qubit tracker
            self.rows[twin_id] = self.rows[orig_id]
            self.qubits[twin_id] = self.qubits[orig_id]

            # Add twin to list of BEFORE/AFTER time dependencies if applicable
            temp_time_deps: dict[int, set[int]] = {}
            for k, v in self.ante.items():
                if orig_id in v:
                    if k not in temp_time_deps:
                        temp_time_deps[k] = [twin_id]
                    else:
                        temp_time_deps[k].append(twin_id)

            for k, v in temp_time_deps.items():
                if k in self.ante:
                    self.ante[k].update(v)

            self.build_rebuild_post_deps()

            # Get neighbours pending for original and transfer to twin
            orig_pending_neighs = [
                n
                for n in self.bgraph.neighbors(orig_id)
                if self.bgraph.get_edge_data(n, orig_id)["kind"] is None
            ]

            # Connect original and twin
            self.bgraph.add_edge(
                orig_id,
                twin_id,
                edge_type="SIMPLE",
                start_coords=None,
                end_coords=None,
                kind=None,
            )

            # Remove pending neighbours from original node and transfer to new twin
            for pending_id in orig_pending_neighs:
                edge_type = self.bgraph.get_edge_data(orig_id, pending_id)["edge_type"]
                self.bgraph.add_edge(
                    pending_id,
                    twin_id,
                    edge_type=edge_type,
                    start_coords=None,
                    end_coords=None,
                    kind=None,
                )
                self.bgraph.remove_edge(orig_id, pending_id)

            # Update pending information for original and twin
            self.bgraph.nodes[orig_id]["completions"] = {
                "degree": self.bgraph.degree(orig_id),
                "pending": 1,
            }
            self.bgraph.nodes[twin_id]["completions"] = {
                "degree": self.bgraph.degree(twin_id),
                "pending": self.bgraph.degree(twin_id),
            }

            # Place twin > Internalise edge characteristics & clear iteration-specific parameters
            self.curr_src_id, self.curr_tgt_id = (orig_id, twin_id)
            self.clear_iter(orig_id, twin_id, twin_mode=True)

            # Edge fulfillment
            call_pathfinder_bfs_std(self, twin_mode=True)

            # Add path
            if self.winner_path:
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

            # Update duration
            _, duration_iter = datetime_manager(t_1=self.t1_iter)

            # Update user if applicable
            if self._kwargs["debug"] > 0:
                print(
                    f"  - Completed: {self.curr_src_id} --> {self.curr_tgt_id}.",
                    f"Vol +: {len(self.winner_path.full_path) - 1}.",
                    f"Duration: {duration_iter:.2f}s",
                )

        # Reset IDs to twin
        self.ids_to_twin = set()

        # Update user if applicable
        if self._kwargs["debug"] > 0:
            print("  - Exiting TWIN mode.")

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

        # Extract fixed weights and key path characteristics
        path_len_w, beams_broken_w = self._kwargs["weights"]
        last_coords, last_zx_block = candidate_path.full_path[-1]
        coords_in_path = np.array([c for c, _ in candidate_path.full_path])

        # Penalise length
        len_contrib = len(candidate_path.full_path) * path_len_w

        # Penalise number of beams broken by path
        broken_beams_contrib = candidate_path.beams_broken_by_path * beams_broken_w

        # Push on Z-axis
        z_push, out_of_bounds = _calculate_z_nudges(
            self.curr_src_id,
            self.curr_tgt_id,
            last_coords,
            last_zx_block,
            coords_in_path,
            self._kwargs["z_stretch"],
            self._kwargs["gravity"],
            self.faux_edge,
            self.rows,
            self.bounds,
            inputs=self.inputs,
            outputs=self.outputs,
            s_gates=self.s_gates,
            t_gates=self.t_gates,
        )

        # Gravity around a specific point in existing blockgraph if applicable
        gravity_pull = _calculate_gravity_nudges(
            self.bgraph,
            self.edge_queue,
            self.curr_src_id,
            self.curr_tgt_id,
            self.twin_trace,
            self.taken,
            last_coords,
            coords_in_path,
            self._kwargs["gravity"],
            self.faux_edge,
            self._kwargs["graph_traverse_mode"],
            self.curr_tgt_zx_type,
        )

        # Return cumulative value
        path_value = len_contrib + broken_beams_contrib + out_of_bounds + z_push + gravity_pull
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

    def draw_blockgraph(
        self, is_final_vis: bool = True, iter_fail: bool = False, embedded: bool = False
    ) -> BlockGraphVisualiser | None:
        """Draw the NX blockgraph using ZX or NX styling.

        Args:
            is_final_vis: Boolean to flag if current visualisation is the final blockgraph.
            iter_fail: Boolean to flag if current visualisation comes right after an iteration failure.
            embedded: Method is being called from the UX and visualisation will therefore be embedded.

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
            "t_gates": self.t_gates,
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
            "t_gates": self.t_gates,
            "completed_edges": self.completed_base_edges,
        }

        # Get stats and clarify parameters that sometimes need adjustment
        self.get_stats()
        pop_vis = self._kwargs["debug"] > 1 or is_final_vis
        draw_style = (
            "nx-fruchterman_reingold"
            if self._kwargs["graph_traverse_mode"] == "bfs-cycles"
            else "zx"
        )
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
            base_graph_draw_style=draw_style,
            msc_stretch=self.msc_stretch,
            stats=self.stats,
            vis_mode=(pop_vis, self._kwargs["animate"]),
        )

        visualiser = BlockGraphVisualiser(vis_state)
        visualiser.build_layout()
        if not embedded:
            visualiser.show()

        return visualiser

    def write_bgraph(self, circuit_name: str = "qc"):
        """Write BlockGraph to a BGRAPH file.

        Args:
            circuit_name (optional): The name of the circuit.

        """
        # Create output directory if it doesn't exist.
        output_dir = BGRAPH_DIR
        if not isinstance(output_dir, Path):
            try:
                output_dir = Path(str(output_dir))
            except Exception as e:
                raise NotADirectoryError(
                    f"Unable to create output directory: '{output_dir}'"
                ) from e
        os.makedirs(output_dir, exist_ok=True)

        # Prepare BGRAPH content
        bgraph_lines = _prep_bgraph_lines(
            circuit_name,
            self.bgraph,
            self.inputs,
            self.outputs,
            self.qubits,
            msc_stretch=self.msc_stretch,
        )

        # Write to BGRAPH file
        if bgraph_lines:
            path_to_output_file = output_dir / f"{circuit_name}.bgraph"
            with open(path_to_output_file, "w") as f:
                f.writelines(bgraph_lines)

    def to_zx_graph(self):
        """Distill into a PyZX graph."""

        # Create empty ZX graph
        zx_graph = zx.Graph()

        # Extract cubes into spiders
        for cube_id, attrs in self.bgraph.nodes(data=True):
            # Each cube gets a spider
            zx_block = attrs.get("zx_block")
            zx_type = ZXTypes.from_str(zx_block.zx_type)
            coords = attrs.get("coords")

            # Qubit and row number possible if ID in input ZX
            if cube_id in self.in_ids:
                qubit = self.in_qubits[cube_id]
                row = self.in_rows[cube_id]
            else:
                qubit = -1
                row = -1

            # Add spider
            vertex = zx_graph.add_vertex(ty=zx_type, qubit=qubit, row=row, index=cube_id)
            zx_graph.set_vdata(vertex, "coords", coords)

        # Add edges
        for u, v, attrs in self.bgraph.edges(data=True):
            zx_type = ZXEdgeTypes.from_str(attrs.get("edge_type"))
            zx_graph.add_edge((u, v), edgetype=zx_type)

        # Write inputs and outputs explicitly
        if self.inputs:
            zx_graph.set_inputs(self.inputs)
        if self.outputs:
            zx_graph.set_outputs(self.outputs)

        # Return PyZX graph
        return zx_graph

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

    def clear_iter(self, u, v, twin_mode=False, slice_mode=False):
        """Clear iteration specific trackers and other variables."""

        # Clear iteration trackers
        self.faux_edge: bool = self.curr_src_id in self.inputs and self.curr_tgt_id in self.inputs
        self.z_bounds: dict[str, int | None] = {"min": None, "max": None}
        self.tent_coords: list[StandardCoord] = None
        self.valid_paths: dict[PositionedZXBlock, list[PositionedZXBlock]] = None
        self.winner_path: CandidatePath = None

        # Edge characteristics
        self.cross_edge = True if (self.bgraph.nodes[v]["coords"] and not twin_mode) else False
        self.is_hadamard = (
            False
            if (twin_mode or slice_mode)
            else self.bgraph.edges[(u, v)]["edge_type"] == "HADAMARD"
        )
        self.curr_src_coords = self.bgraph.nodes[self.curr_src_id if not twin_mode else u]["coords"]
        self.curr_tgt_coords = self.bgraph.nodes[self.curr_tgt_id if not twin_mode else v]["coords"]
        self.curr_tgt_zx_type = self.bgraph.nodes[self.curr_tgt_id if not twin_mode else v][
            "zx_block"
        ].zx_type

        # Copy of taken without source and target coordinates
        self.pruned_taken = self.taken.copy()
        self.pruned_taken.discard(self.curr_src_coords)
        self.pruned_taken.discard(self.curr_tgt_coords)

        # Prune beams so edge fulfillment starts with clean slate
        self.prune_beams()

    def cleanup(self):
        """Carry out cleanup operations after build."""
        # Delete temporary files
        rm_temp_files(MEDIA_DIR / "temp")

    def stretch_msc_cubes(self, max_stretch=10):
        """Expand MSC cubes up to maximum Z-factor available.

        Args:
            max_stretch: The maximum possible stretch for a MSC cube.

        """

        # Get lowest point of blockgraph
        lowest_in_bgraph = min([z for _, _, z in self.taken])

        # Check if space below MSC cubes is taken
        for cube_id in self.msc_cubes:
            # Get MSC coords
            x, y, z = self.bgraph.nodes[cube_id]["coords"]

            # Reset max stretch
            new_stretch = 3
            for i in range(self.msc_stretch[cube_id] + 1, max_stretch):
                if (x, y, z - i) in self.taken or z - i < lowest_in_bgraph:
                    break
                new_stretch = i
            self.msc_stretch[cube_id] = new_stretch
            self.taken.update([(x, y, z - i) for i in range(new_stretch)])

    def distributed_msc_factory(self):
        """Expand MSC cubes up to maximum Z-factor available.

        Args:
            max_stretch: The maximum possible stretch for a MSC cube.

        """

        def _sums(n, k):
            for combo in itertools.combinations(range(n + k - 1), k - 1):
                s = [combo[0]]
                s.extend([(combo[i] - combo[i - 1] - 1) for i in range(1, k - 1)])
                s.append(n + k - 2 - combo[k - 2])
                yield s

        def _manhattan_sphere(point, step):
            k = len(point)
            for differences in _sums(step, k):
                signed_differences = [[-d, d] if d != 0 else [d] for d in differences]
                for p in itertools.product(*signed_differences):
                    yield [x + d for x, d in zip(point, p)]

        # Get lowest point of blockgraph
        all_x = [x for x, _, _ in self.taken]
        all_y = [y for _, y, _ in self.taken]
        all_z = [z for _, _, z in self.taken]

        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        z_min = min(all_z)
        x_span = x_max - x_min
        y_span = y_max - y_min
        max_span = max(x_span, y_span)

        # Check if space below MSC cubes is taken
        for cube_id in self.msc_cubes:
            # Flag addition for cube
            msc_added = False

            # Get MSC coords
            x, y, z = self.bgraph.nodes[cube_id]["coords"]

            # Try in increasing distance from existing MSC cube
            for d in range(max_span + 2):
                # Break if an MSC has been added
                if msc_added:
                    break

                # Generate a Manhattan sphere of potential coords for placement
                sphere_coords = [
                    (int(c[0]), int(c[1]), int(c[2])) for c in list(_manhattan_sphere((x, y, z), d))
                ]

                # Eliminate coords outside of limits of existing blockgraph
                tent_base_coords = [
                    c
                    for c in sphere_coords
                    if (
                        c not in self.taken
                        and x_min < c[0] < x_max + 1
                        and y_min < c[1] < y_max + 1
                        and c[2] < z
                        and c[2] > z_min + 3
                    )
                ]

                # Check remaining tentative coords for space availability
                if tent_base_coords:
                    for x, y, z in tent_base_coords:
                        # Break if an MSC has been added
                        if msc_added:
                            break

                        # Connection is impossible if space immediately above is taken
                        if (x, y, z + 1) in self.taken:
                            continue

                        # Placement is impossible if there is not enough space for MSC cube
                        for i in range(3):
                            if ((x, y, z - i) in self.taken) or (z - i < z_min):
                                break

                            # Place if enough space is found
                            if i == 2:
                                new_id = max(self.bgraph.nodes) + 1
                                self.bgraph.add_node(
                                    new_id,
                                    zx_block=ZXBlockRegistry.get_create(kind="TTO"),
                                    coords=(x, y, z),
                                    completions={
                                        "degree": 0,
                                        "pending": 0,
                                    },
                                )
                                self.msc_stretch[new_id] = 3
                                self.msc_factory[new_id] = (x, y, z)
                                msc_added = True

    def msc_discard(self, discard_id: int):
        """Discard an failed factory MSC cube.

        Args:
            discard_id: ID to remove from the computation.

        Note. This operation is meant to be run only on factory MSCs to
        clear the space of unused and failed attempts to produce a magic
        state. To discard a scheduled MSC you must exchange it for a
        factory MSC using `msc_exchange`.

        """

        # Fail any attempt to discard a scheduled MSC
        if discard_id not in self.msc_factory:
            print("Skipping discard operation because requested discard ID is not a factory MSC.")
            return

        # Get coords of MSC that is leaving the party
        msc_coords = self.bgraph.nodes[discard_id]["coords"]
        remove_coords = [
            (msc_coords[0], msc_coords[1], msc_coords[2] - i)
            for i in range(self.msc_stretch[discard_id])
        ]

        # Remove from taken
        [self.taken.discard(c) for c in remove_coords]

        # Remove from BGRAPH
        self.bgraph.remove_node(discard_id)

        # Remove from applicable trackers
        del self.msc_factory[discard_id]
        del self.msc_stretch[discard_id]

    def msc_exchange(self, include_id: int, discard_id: int):
        """Exchange an MSC for another.

        Args:
            include_id: ID for factory MSC to plug into the computation.
            discard_id: ID for scheduled MSC to remove from the computation.

        NB!
            This operation **can** and *will sometimes* fail, especially in
            very dense/crowded blockgraphs. The safeties needed to guarantee
            success do not seem warranted because they would put rather than
            release pressure on the main build. Additionally, this particular
            operation is not a last recourse. The last recourse alternative
            that cannot fail is to stretch the computation to give time for
            discard_id to succeed.

        """

        # Check that both include_id and discard_id are MSC blocks
        include_zx_type = self.bgraph.nodes[include_id]["zx_block"].zx_type
        remove_zx_type = self.bgraph.nodes[discard_id]["zx_block"].zx_type
        print(include_zx_type, remove_zx_type)
        if include_zx_type != "T" or remove_zx_type != "T":
            print("Skipping exchange because one of the requested IDs is not an MSC.")
            return

        # Get coords of MSC that is leaving the party and remove from taken
        remove_coords = self.bgraph.nodes[discard_id]["coords"]
        self.taken.discard(remove_coords)

        # Find neighbour of the MSC that is leaving
        neigh_id = list(self.bgraph.neighbors(discard_id))[0]

        # Connect factory MSC cube to neigh of MSC to remove
        # Create empty edge
        u, v = include_id, neigh_id
        self.bgraph.add_edge(
            u,
            v,
            edge_type="SIMPLE",
            start_coords=None,
            end_coords=None,
            kind=None,
        )

        # Set attributes needed for pathfinder to run
        self.curr_src_id, self.curr_tgt_id = (u, v)
        self.clear_iter(u, v)

        # Call pathfinder
        call_pathfinder_bfs_cross(self)

        # Prune beams (beams --> `None`)
        self.prune_beams()

        # Add path to blockgraph
        if self.winner_path:
            # Remove old MSC since connection was possible
            self.bgraph.remove_node(discard_id)
            del self.msc_cubes[discard_id]
            del self.msc_stretch[discard_id]

            # Add new path
            self.add_path()

            # Prune beams again (`None` beams removed entirely)
            self.prune_beams()

        else:
            # Reset to initial status since connection was not possible
            self.taken.add(remove_coords)
            self.bgraph.remove_edge(u, v)

            # Prune beams again (`None` beams removed entirely)
            self.prune_beams()

    def slice_stretch(self, slice_at_z: int, shift_z: int = 5):
        """Pause the computation at an arbitrary Z-index (post-processing)."""

        # Announce slicing
        print(f"\n=> Slice & Stretch BlockGraph at Z={slice_at_z}.")

        # Start timer
        self.t1_iter, _ = datetime_manager()

        # Loop over cubes detecting which to shift
        broken_edges = []
        for cube_id, attrs in self.bgraph.nodes(data=True):
            # Get cubes original position
            orig_x, orig_y, orig_z = attrs["coords"]

            # Shift all cubes at or above slice
            if orig_z >= slice_at_z:
                attrs["coords"] = (orig_x, orig_y, orig_z + shift_z)

            # Handle predecessors
            if orig_z == slice_at_z:
                # ID predecessors
                predecessors = [
                    n
                    for n in self.bgraph.neighbors(cube_id)
                    if self.bgraph.nodes[n]["coords"][2] < slice_at_z
                ]

                if predecessors:
                    prev_id = predecessors[0]
                    prev_zx_type = self.bgraph.nodes[prev_id]["zx_block"].zx_type
                    if prev_zx_type == "T":
                        self.msc_stretch[prev_id] = self.msc_stretch[prev_id] + shift_z
                    else:
                        self.bgraph.edges[prev_id, cube_id]["start_coords"] = None
                        self.bgraph.edges[prev_id, cube_id]["end_coords"] = None
                        self.bgraph.edges[prev_id, cube_id]["kind"] = None
                    broken_edges.append((prev_id, cube_id))

        # Rebuild taken
        coords_dict = nx.get_node_attributes(self.bgraph, "coords")
        self.taken = set([coord for coord in coords_dict.values()])
        for k, v in self.msc_stretch.items():
            x, y, z = self.bgraph.nodes[k]["coords"]
            for i in range(v):
                self.taken.add((x, y, z - i))
        self.pruned_taken = self.taken.copy()

        # Reposition all edges in graph above slice
        for u, v, attrs in self.bgraph.edges(data=True):
            if attrs["start_coords"] and attrs["end_coords"]:
                start_x, start_y, start_z = attrs["start_coords"]
                end_x, end_y, end_z = attrs["end_coords"]
                if start_z >= slice_at_z:
                    self.bgraph.edges[u, v]["start_coords"] = (start_x, start_y, start_z + shift_z)
                    self.bgraph.edges[u, v]["end_coords"] = (end_x, end_y, end_z + shift_z)

        # Draw split blockgraph if applicable
        if self._kwargs["debug"] > 1:
            self.draw_blockgraph()

        # Reconnect broken edges
        for u, v in broken_edges:
            # Internalise key edge characteristics & clear iteration-specific parameters
            if self._kwargs["debug"] > 1:
                print(f"* Reconnecting: {u} --> {v}")

            self.curr_src_id, self.curr_tgt_id = (u, v)
            self.clear_iter(u, v)
            added_volume = 0

            if self.bgraph.nodes[self.curr_src_id]["zx_block"].zx_type == "T":
                # Get original coords & remove from taken
                x, y, z = self.bgraph.nodes[self.curr_src_id]["coords"]
                [
                    self.taken.remove((x, y, z - i))
                    for i in range(self.msc_stretch[self.curr_src_id])
                ]

                # Add new coords and update taken
                self.bgraph.nodes[self.curr_src_id]["coords"] = (x, y, z + shift_z)
                x, y, z = self.bgraph.nodes[self.curr_src_id]["coords"]
                [self.taken.add((x, y, z - i)) for i in range(self.msc_stretch[self.curr_src_id])]

                # Update other applicable trackers
                if self.curr_src_id in self.msc_factory:
                    self.msc_factory[self.curr_src_id] = self.bgraph.nodes[self.curr_src_id][
                        "coords"
                    ]

            # Exchange conditional for standard cube if target is conditional
            v_block = self.bgraph.nodes[v]["zx_block"]
            if "*" in v_block.kind:
                u_block = self.bgraph.nodes[u]["zx_block"]
                temp_block = ZXBlockRegistry.get_create(kind=u_block.kind)
                self.bgraph.nodes[v]["zx_block"] = temp_block

            # Call pathfinder
            call_pathfinder_bfs_cross(self, stretch=True)

            # Prune beams (existing beams go to `None`)
            self.prune_beams()

            # Add path to blockgraph
            if self.winner_path:
                # Exchange final block of winner path if target is boundary
                if v_block.zx_type == "O":
                    self.winner_path.full_path[-1] = (self.curr_tgt_coords, v_block)

                # Add path
                self.add_path()
                added_volume = len(self.winner_path.full_path) - 2

                # Exchange conditional back if target is conditional
                if "*" in v_block.kind:
                    self.bgraph.nodes[v]["zx_block"] = v_block

            else:
                if self._kwargs["debug"] > 2 or self._kwargs["animate"]:
                    self.draw_blockgraph(is_final_vis=False, iter_fail=True)
                raise ValueError(f"ERROR. Unable to reconnect: {u} --> {v}")

            # Prune beams again (IDs with `None` beams are removed)
            # Removal is needed for visualisations to work
            self.prune_beams()

            if self._kwargs["debug"] > 0:
                print(
                    f" - Reconnected: {u} --> {v}.",
                    f"Vol +: {added_volume}.",
                )
                if self._kwargs["debug"] > 2:
                    self.draw_blockgraph()

        # Update duration
        _, duration_iter = datetime_manager(t_1=self.t1_iter)

        print(
            f" - SUCCESS! Slice & Stretched BlockGraph at Z={slice_at_z}.",
            f"Duration: {duration_iter:.2f}s",
        )

    def switch_conditional(self, switch_id: int, to_y: bool):
        """Switches the conditional to and X or Y cube.

        Args:
            switch_id: The ID of the conditional to switch.
            to_y: Whether to place a Y-cube or not.

        """
        if to_y:
            self.bgraph.nodes[switch_id]["zx_block"] = ZXBlockRegistry.get_create(kind="YYO")
        else:
            conditional_kind = self.bgraph.nodes[switch_id]["zx_block"].kind
            self.bgraph.nodes[switch_id]["zx_block"] = ZXBlockRegistry.get_create(
                kind=conditional_kind[:2] + "X"
            )


###########
# CALLERS #
###########
def call_pathfinder_bfs_std(bgraph_manager: BlockGraphManager, twin_mode: bool = False):
    """Call the BFS pathfinder for a standard edge.

    Args:
        bgraph_manager: The BlockGraph Manager currently driving the build.
        twin_mode (optional): True if the current edge is part of a twin creation cycle.

    """

    # Calculate optimal step and overload
    step, overload = _calculate_overload(
        bgraph_manager._kwargs["graph_traverse_mode"],
        bgraph_manager.first_id,
        bgraph_manager.curr_src_id,
        bgraph_manager.curr_tgt_id,
        bgraph_manager.curr_tgt_zx_type,
        bgraph_manager.qubits,
        bgraph_manager.rows,
        twin_mode,
        bgraph_manager._kwargs["z_stretch"],
    )

    # Set time constraints if applicable
    if bgraph_manager.curr_tgt_id in bgraph_manager.ante:
        floor_coords = [
            bgraph_manager.bgraph.nodes[cube_id]["coords"]
            for cube_id in bgraph_manager.ante[bgraph_manager.curr_tgt_id]
            if bgraph_manager.bgraph.nodes[cube_id]["coords"]
        ]
        bgraph_manager.z_bounds["min"] = max([c[2] for c in floor_coords]) if floor_coords else None
        step = max(
            step,
            get_manhattan(
                bgraph_manager.curr_src_coords,
                (
                    bgraph_manager.curr_src_coords[0],
                    bgraph_manager.curr_src_coords[1],
                    bgraph_manager.z_bounds["min"],
                ),
            ),
        )
    if bgraph_manager.curr_tgt_id in bgraph_manager.post:
        roof_coords = [
            bgraph_manager.bgraph.nodes[cube_id]["coords"]
            for cube_id in bgraph_manager.post[bgraph_manager.curr_tgt_id]
            if bgraph_manager.bgraph.nodes[cube_id]["coords"]
        ]
        bgraph_manager.z_bounds["max"] = max([c[2] for c in roof_coords]) if roof_coords else None

    # Loop until path is found
    max_step = step + 100
    while step < max_step:
        # Get many tentative coordinates or set a specific target coordinate
        bgraph_manager.tent_coords = gen_tent_tgt_coords(
            bgraph_manager.curr_src_coords,
            step,
            bgraph_manager.taken,
            overload=overload if step < 3 else 0,
            z_bounds=bgraph_manager.z_bounds,
            twin_mode=twin_mode,
            graph_bounds=bgraph_manager.bounds,
        )

        # THIS SHOULDN'T BE NEEDED BUT I'M GOING TO LEAVE IT
        # HERE FOR A BIT JUST IN CASE
        if twin_mode and not bgraph_manager.tent_coords:
            bgraph_manager.tent_coords = gen_tent_tgt_coords(
                bgraph_manager.curr_src_coords,
                step,
                bgraph_manager.taken,
                overload=overload if step < 3 else 0,
                z_bounds=bgraph_manager.z_bounds,
                twin_mode=False,
                graph_bounds=bgraph_manager.bounds,
            )

        # Try finding paths to each tentative coordinate
        if bgraph_manager.tent_coords:
            # Get a number of valid paths (topologically correct, not necessarily optimal)
            for iter_graph_bounds in [bgraph_manager.bounds, None]:
                pathfinder_init_state = _get_bgraph_snapshot(bgraph_manager, iter_graph_bounds)
                pathfinder = PathFinderManager(pathfinder_init_state)
                if step > 4:
                    bgraph_manager.valid_paths = pathfinder.pathfinder_a_star_multi_target()
                else:
                    bgraph_manager.valid_paths = pathfinder.pathfinder_bfs()
                if bgraph_manager.valid_paths:
                    break

        # Pick between valid paths
        if bgraph_manager.valid_paths:
            for valid_path in bgraph_manager.valid_paths.values():
                # Extract key path information
                tgt_coords, tgt_zx_block = valid_path[-1]
                coords_in_path = [c for c, _ in valid_path][1:]

                # Re-assign last block in sequence if target is a boundary
                if bgraph_manager.curr_tgt_zx_type == "O":
                    tgt_zx_block = ZXBlockRegistry.get_create(kind="OOO")
                    valid_path[-1] = (tgt_coords, tgt_zx_block)

                # Check if exits are unobstructed
                tgt_unobstr_exit_n, bgraph_manager.tgt_beams, bgraph_manager.tgt_beams_short = (
                    check_exits_add_beams(
                        tgt_zx_block,
                        tgt_coords,
                        bgraph_manager.taken,
                        coords_in_path,
                        bgraph_manager._kwargs["beams_len_short"],
                    )
                )

                # Continue if minimum required number of exits available for target
                # Note. Open boundaries typically are part of a computation, so leave one exit open
                min_tgt_unobstr_exit_n = (
                    1
                    if bgraph_manager.faux_edge
                    else bgraph_manager.bgraph.nodes[bgraph_manager.curr_tgt_id]["completions"][
                        "pending"
                    ]
                    - 1
                )
                if tgt_unobstr_exit_n >= min_tgt_unobstr_exit_n:
                    # Check if path breaks more beams than tolerable
                    extra_allowance = 0
                    if (
                        bgraph_manager.curr_src_id,
                        bgraph_manager.curr_tgt_id,
                    ) in bgraph_manager.edge_queue:
                        if not bgraph_manager.edge_queue.index(
                            (bgraph_manager.curr_src_id, bgraph_manager.curr_tgt_id)
                        ) + 1 == len(bgraph_manager.edge_queue):
                            nxt_edge = bgraph_manager.edge_queue[
                                bgraph_manager.edge_queue.index(
                                    (bgraph_manager.curr_src_id, bgraph_manager.curr_tgt_id)
                                )
                                + 1
                            ]
                            nxt_id = nxt_edge[1]
                            nxt_coords = bgraph_manager.bgraph.nodes[
                                bgraph_manager.twin_trace[nxt_id][-1]
                            ]["coords"]
                            if (
                                nxt_coords
                                and (bgraph_manager.curr_tgt_id, nxt_id)
                                in bgraph_manager.bgraph.edges
                            ):
                                md = get_manhattan(tgt_coords, nxt_coords)
                                if md == 1:
                                    move = tuple(np.array(nxt_coords) - np.array(tgt_coords))
                                    nxt_zx_block = bgraph_manager.bgraph.nodes[nxt_id]["zx_block"]
                                    if tgt_zx_block.kind and nxt_zx_block.kind:
                                        exits_match = tgt_zx_block.cube_open_faces_match(
                                            move, tgt_zx_block=nxt_zx_block
                                        )
                                        faces_match = tgt_zx_block.face_match(move, nxt_zx_block)
                                        if exits_match and faces_match:
                                            extra_allowance = 1

                    beam_clashes, beams_broken_by_path = bgraph_manager.check_beams(
                        coords_in_path, twin_mode=twin_mode, extra_allowance=extra_allowance
                    )

                    # Append path to viable paths if path clears all checks
                    if not beam_clashes or bgraph_manager.faux_edge:
                        # Consolidate path data
                        candidate_path = CandidatePath(
                            **{
                                "full_path": valid_path,
                                "tgt_beams": bgraph_manager.tgt_beams,
                                "tgt_beams_short": bgraph_manager.tgt_beams_short,
                                "beams_broken_by_path": beams_broken_by_path,
                                "tgt_unobstr_exit_n": tgt_unobstr_exit_n,
                            }
                        )

                        # Append to viable paths
                        bgraph_manager.winner_path = (
                            candidate_path
                            if (
                                not bgraph_manager.winner_path
                                or bgraph_manager.value_function(candidate_path)
                                > bgraph_manager.value_function(bgraph_manager.winner_path)
                            )
                            else bgraph_manager.winner_path
                        )
        # Break if valid paths generated at step
        if bgraph_manager.winner_path:
            break

        # Increase distance if no valid paths found at current step
        step += 1


def call_pathfinder_bfs_cross(bgraph_manager: BlockGraphManager, stretch: bool = False):
    """Call the Djikstra pathfinder for a cross edge.

    Args:
        bgraph_manager: The BlockGraph Manager currently driving the build.
        stretch: Run is part of a stretch operation.

    """

    # Define tentative coordinates as tgt coords
    bgraph_manager.tent_coords = [bgraph_manager.curr_tgt_coords]

    # Try finding shortest path
    if bgraph_manager.tent_coords:
        # Get a number of valid paths (topologically correct, not necessarily optimal)

        for iter_graph_bounds in [bgraph_manager.bounds, None]:
            pathfinder_init_state = _get_bgraph_snapshot(bgraph_manager, iter_graph_bounds)
            pathfinder = PathFinderManager(pathfinder_init_state)
            bgraph_manager.valid_paths = pathfinder.pathfinder_a_star(stretch=stretch)
            if bgraph_manager.valid_paths:
                break
        if not bgraph_manager.valid_paths:
            bgraph_manager.valid_paths = pathfinder.pathfinder_bfs(stretch=stretch)

    # Handle cross edge
    if len(bgraph_manager.valid_paths) == 1:
        bgraph_manager.winner_path = CandidatePath(
            **{
                "full_path": list(bgraph_manager.valid_paths.values())[0],
                "tgt_beams": (
                    bgraph_manager.beams[bgraph_manager.curr_tgt_id]
                    if bgraph_manager.curr_tgt_id in bgraph_manager.beams
                    else None
                ),
                "tgt_beams_short": (
                    bgraph_manager.beams_short[bgraph_manager.curr_tgt_id]
                    if bgraph_manager.curr_tgt_id in bgraph_manager.beams
                    else None
                ),
                "beams_broken_by_path": 0,  # Not calculated (pathfinder handles internally)
                "tgt_unobstr_exit_n": (
                    bgraph_manager.bgraph.nodes[bgraph_manager.curr_tgt_id]["completions"][
                        "pending"
                    ]
                    - 1
                ),  # Not calculated (pathfinder handles internally)
            }
        )


######################
# AUX BGRAPH MANAGER #
######################
def _prep_bgraph_lines(
    circuit_name: str,
    bgraph: nx.Graph,
    inputs: list[int],
    outputs: list[int],
    qubits: dict[int, int],
    msc_stretch: dict[int, int] = {},
) -> list[str]:
    # Initialise lines array
    bgraph_lines = []

    # Append metadata
    bgraph_lines.append("BLOCKGRAPH 0.1.0;\n")
    bgraph_lines.append("\nMETADATA: attr_name; value;\n")
    bgraph_lines.append("source; topologiq;\n")
    bgraph_lines.append(f"circuit_name; {circuit_name};\n")

    # CUBES
    bgraph_lines.append("\nCUBES: index;x;y;z;kind;label;\n")
    for n_id, attrs in bgraph.nodes(data=True):
        # Get coords and kind
        if attrs["coords"] and attrs["zx_block"]:
            x, y, z = attrs["coords"]
            kind = attrs["zx_block"].kind
        else:
            x, y, z = (None, None, None)
            kind = ""

        # Re-write kind into BGRAPH standard if applicable
        if kind in ["YYO", "TTO"]:
            neighs = list(bgraph.neighbors(n_id))
            if len(neighs) > 1:
                raise ValueError("Error writing BGRAPH. Malformed Y cube.")
            _, _, neigh_z = bgraph.nodes(data=True)[neighs[0]]["coords"]
            if kind == "YYO":
                kind = "Yi" if neigh_z > z else "Ym"
            if kind == "TTO":
                if neigh_z <= z:
                    raise ValueError(
                        "Error writing BGRAPH. Malformed cultivation or distillation cube."
                    )
                neigh_kind = bgraph.nodes(data=True)[neighs[0]]["zx_block"].kind
                kind = neigh_kind[:2] + "t"
                z = z - msc_stretch[n_id]

        # Assemble label
        label = ""
        if n_id in inputs:
            label = f"in_{qubits[n_id]}" if n_id in qubits else "in"
        if n_id in outputs:
            label = f"out_{qubits[n_id]}" if n_id in qubits else "out"

        # Write
        bgraph_lines.append(f"{n_id};{x!s};{y!s};{z!s};{kind};{label};\n")

    # PIPES
    bgraph_lines.append("\nPIPES: src;tgt;kind;\n")
    for u, v, kind in bgraph.edges(data="kind"):
        bgraph_lines.append(f"{u!s};{v!s};{kind};\n")

    return bgraph_lines


def _prep_ante(
    ante: dict[int, set[int]],
    t_zx_tracker: dict[int, int],
    order: dict[Any, int],
    xwebs: dict[Any, PauliWeb],
    zwebs: dict[Any, PauliWeb],
):
    for t_gate_id in order:
        if t_gate_id in (*xwebs, *zwebs):
            # Add any IDs in path of X webs
            if t_gate_id in xwebs:
                ante[t_zx_tracker[t_gate_id]].update(
                    chain.from_iterable(xwebs[t_gate_id].half_edges())
                )

            # Add any IDs in path of Z webs
            if t_gate_id in zwebs:
                ante[t_zx_tracker[t_gate_id]].update(
                    chain.from_iterable(zwebs[t_gate_id].half_edges())
                )


def _prep_post(post: dict[int, set[int]], ante: dict[int, set[int]]):
    for k, predecessors in ante.items():
        for predecessor in predecessors:
            # Add each predecessor T-gate as successor of corresponding ID
            if predecessor in post:
                post[predecessor].update([k])
            else:
                post[predecessor] = set([k])


def _handle_s_spider(
    spider_id: int,
    bgraph: nx.Graph,
    phases: dict[int, int | Fraction],
    y_cubes: dict[int, str],
    s_gates: dict[int, list[tuple[int, int] | None]],
    qubits: dict[int, int],
    rows: dict[int, int],
):
    # Calculate next ID
    y_id = max(bgraph.nodes()) + 1

    # Remove phase of original node as it will get a pattern instead
    phases[spider_id] = 0

    # Initialisation Y-cube
    if bgraph.degree(spider_id) == 1:
        # Log and change ZX block in node attributes
        y_cubes[spider_id] = "Yi"
        bgraph.nodes[spider_id]["zx_block"] = ZXBlockRegistry.get_create(zx_type="Y")

    # Mid-circuit S-gate
    else:
        # Log Y-cube as measurement
        s_gates[spider_id] = [(spider_id, y_id)]
        y_cubes[y_id] = "Ym"

        # Add Y-cube to BGRAPH
        bgraph.add_node(
            y_id,
            zx_block=ZXBlockRegistry.get_create(zx_type="Y"),
            coords=None,
            completions={
                "degree": None,
                "pending": None,
            },
        )

        # Add corresponding entry to qubit and row trackers
        qubits[y_id] = qubits[spider_id] - 1
        rows[y_id] = rows[spider_id] - 1

        # Add corresponding edge
        bgraph.add_edge(
            spider_id,
            y_id,
            edge_type="SIMPLE",
            start_coords=None,
            end_coords=None,
            kind=None,
        )


def _handle_t_spider(
    spider_id: int,
    bgraph: nx.Graph,
    phases: dict[int, int | Fraction],
    msc_cubes: dict[int, str],
    t_gates: dict[int, list[tuple[int, int] | None]],
    t_zx_tracker: dict[int, int],
    ante: dict[int, set[int]],
    qubits: dict[int, int],
    rows: dict[int, int],
    inputs: list[int],
):
    # Determine current max ID in graph
    max_id = max(bgraph.nodes())

    # Initialisation T-gate (MSC)
    if spider_id in inputs:
        # Remove phase of original node as it will get a pattern instead
        phases[spider_id] = 0

        # Add to MSC block to tracker
        msc_cubes[spider_id] = "Mi"

        # Update graph node to a T
        bgraph.nodes[spider_id]["zx_block"] = ZXBlockRegistry.get_create(zx_type="T")

    else:
        # IDs for all spiders in sequence
        msc_id, xz_id = (max_id + 1, max_id + 2)

        # Update applicable trackers
        msc_cubes[msc_id] = "Mm"
        t_gates[spider_id] = [(spider_id, msc_id), (spider_id, xz_id)]
        t_zx_tracker[spider_id] = xz_id
        ante[xz_id] = set([spider_id, msc_id])

        # Reset phase on original spider (it is getting a pattern instead)
        phases[spider_id] = 0

        # Attach Magic State & Conditional Cubes
        id_to_types = {msc_id: "T", xz_id: "XZ"}
        for new_id in [msc_id, xz_id]:
            bgraph.add_node(
                new_id,
                zx_block=ZXBlockRegistry.get_create(zx_type=id_to_types[new_id]),
                coords=None,
                completions={
                    "degree": None,
                    "pending": None,
                },
            )
            qubits[new_id] = qubits[spider_id] + (0 if new_id == msc_id else -0.5)
            rows[new_id] = rows[spider_id] - (0.3 if new_id == msc_id else -0.3)

            bgraph.add_edge(
                spider_id,
                new_id,
                edge_type="SIMPLE",
                start_coords=None,
                end_coords=None,
                kind=None,
            )


def _calculate_z_nudges(
    curr_src_id: int,
    curr_tgt_id: int,
    last_coords: StandardCoord,
    last_zx_block: ZXBlock,
    coords_in_path: NDArray[Any],
    z_stretch: int,
    gravity: int,
    faux_edge: bool,
    rows: dict[int, int],
    bounds: GraphBounds,
    inputs: list[int] = [],
    outputs: list[int] = [],
    s_gates: dict[int, list[tuple[int, int] | None]] = {},
    t_gates: dict[int, list[tuple[int, int] | None]] = {},
):
    # Default values
    z_push, out_of_bounds = (0, 0)

    # Calculate nudge only if applicable
    if z_stretch or last_zx_block.zx_type in ["Y", "T"] or faux_edge:
        # Define weight
        stretch_multiplier = z_stretch if z_stretch else 1

        # Push down for Y-cubes and cultivation/distillation
        if last_zx_block.zx_type in ["Y", "T", "XZ"]:
            z_push = -1 * last_coords[2]
        # Favour row difference for all other cubes
        else:
            row_diff = 0
            if curr_tgt_id in rows and curr_src_id in rows:
                row_diff = rows[curr_tgt_id] - rows[curr_src_id]
            z_push = (row_diff * stretch_multiplier * last_coords[2]) if not faux_edge else 0

        # Apply bounds if given
        if not faux_edge and bounds.x and bounds.y:
            x_coords = coords_in_path[:, 0]
            y_coords = coords_in_path[:, 1]
            x_out = [x < 0 or x > bounds.x for x in x_coords]
            y_out = [y < 0 or y > bounds.y for y in y_coords]
            out_of_bounds = (sum(x_out) + sum(y_out)) * -stretch_multiplier

    # Adjust z-push for faux edges
    if gravity and faux_edge:
        z_push = -100 * abs(last_coords[2])

    # Discount move UP for Z-bases of T- and S-gate patterns
    if curr_tgt_id in t_gates:
        first_z_coords = coords_in_path[1][2]
        z_push = first_z_coords * -1 * 100

    return z_push, out_of_bounds


def _calculate_gravity_nudges(
    bgraph: nx.Graph,
    edge_queue: list[tuple[int, int]],
    curr_src_id: int,
    curr_tgt_id: int,
    twin_trace: dict[int, list[int]],
    taken: set[StandardCoord],
    last_coords: StandardCoord,
    coords_in_path: NDArray[Any],
    gravity: int,
    faux_edge: bool,
    graph_traverse_mode: str,
    curr_tgt_zx_type: str,
):
    # Default values
    gravity_pull = 0

    # Calculate nudge only if applicable
    if gravity:
        # Find centre
        # Aim for centremost point of graph
        centre_coords = np.sum([np.array(coords) for coords in taken], axis=0) / len(taken)

        # Push towards neighbour if next edge is a crosss edge
        nxt_is_cross = False
        curr_edge = (curr_src_id, curr_tgt_id)
        if curr_edge in edge_queue and not edge_queue.index(curr_edge) + 1 == len(edge_queue):
            nxt_id = edge_queue[edge_queue.index(curr_edge) + 1][1]
            nxt_coords = bgraph.nodes[twin_trace[nxt_id][-1]]["coords"]
            if nxt_coords and (curr_tgt_id, nxt_id) in bgraph.edges:
                centre_coords = nxt_coords
                nxt_is_cross = True

        if (
            nxt_is_cross
            or faux_edge
            or (graph_traverse_mode == "bfs-cycles" and curr_tgt_zx_type != "O")
        ):
            d_to_centre = np.linalg.norm(np.array(last_coords) - np.array(centre_coords))
            gravity_pull = d_to_centre * -10 * gravity
        elif curr_tgt_zx_type not in ["O", "Y", "T", "XZ"]:
            centre_x, centre_y, _ = centre_coords
            x, y, _ = last_coords
            gravity_pull = -gravity * (abs(x - centre_x) + abs(y - centre_y))
        elif curr_tgt_zx_type in ["Y", "T", "XZ"]:
            centre_x, centre_y, _ = centre_coords
            centre_z = bgraph.nodes[curr_src_id]["coords"][2] - (
                1 if curr_tgt_zx_type in ["Y", "T", "XZ"] else 0
            )
            mean_d_to_centre = np.sum(
                [
                    (abs(x - centre_x) + abs(y - centre_y) + abs(z - centre_z))
                    for x, y, z in coords_in_path
                ]
            ) / len(coords_in_path)
            gravity_pull = -gravity * mean_d_to_centre

    return gravity_pull


#########################
# AUX PATHFINDER CALLER #
#########################
def _calculate_overload(
    graph_traverse_mode: str,
    first_id: int,
    curr_src_id: int,
    curr_tgt_id: int,
    curr_tgt_zx_type: str,
    qubits: dict[int, int],
    rows: dict[int, int],
    twin_mode: bool,
    z_stretch: int | None,
) -> int:
    step = 1
    overload = 0
    if graph_traverse_mode in ["bfs-cnots", "bfs-cnot-cycles", "tfs-cnots"]:
        if (
            curr_src_id in qubits
            and curr_tgt_id in qubits
            and qubits[curr_src_id] == qubits[curr_tgt_id]
        ):
            if qubits[first_id] == qubits[curr_src_id]:
                overload = 0
                step = max(z_stretch, 1)
        else:
            overload = 1
    elif z_stretch and rows[curr_src_id] != rows[curr_tgt_id]:
        overload = z_stretch

    if twin_mode:
        overload = 1

    if curr_tgt_zx_type in ["Y", "XZ", "T"]:
        overload = 2

    return step, overload


def _get_bgraph_snapshot(
    bgraph_manager: BlockGraphManager, iter_graph_bounds: GraphBounds | None
) -> PathfinderInitState:
    return PathfinderInitState(
        bgraph_manager.bgraph,
        bgraph_manager.beams,
        bgraph_manager.beams_short,
        bgraph_manager.curr_src_id,
        bgraph_manager.curr_tgt_id,
        bgraph_manager.tent_coords,
        bgraph_manager.cross_edge,
        bgraph_manager.taken,
        bgraph_manager.pruned_taken,
        bgraph_manager.is_hadamard,
        bgraph_manager.z_bounds,
        iter_graph_bounds,
        bgraph_manager._kwargs,
    )
