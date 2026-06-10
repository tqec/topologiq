"""Core script for the inner pathfinder algorithm (BFS).

This file contains functions that altogether create topologically-correct 3D edge paths
between a given source cube with pre-determined position and kind and one or more target cubes.
The algorithm is flexible enough to accomodate different kinds of requests. If it gets more
than one tentative coordinates for the target cube, it assumes the target cube has not yet
been placed in the 3D space and creates tentative paths to a user-determined max. % of
tentative coordinates (the max can be 100% but this has found to be unnecessary).
If it gets only one tentative position (and information for the target cube in that coordinates),
it assumes the target cube has already been placed in the 3D space and goes into single-path mode,
where it returns the shortest path between source and target cubes.

Usage:
    Call `pathfinder()` programmatically from a separate script, with an appropriate combination of optional parameters.

Notes:
    For now, none of the functions in this file are to be called individually.
    In the future, some of the functions could be called by variant algorithms that
        do not necessarily need or want to implement all separate features.

"""

from collections import deque

import networkx as nx
import numpy as np

from topologiq.core.blocks import PositionedZXBlock, ZXBlock
from topologiq.core.pathfinder.spatial import (
    check_skip_move,
    gen_bounding_box,
    get_coords_for_current_move,
)
from topologiq.core.pathfinder.utils import gen_exit_conditions
from topologiq.utils.classes import CubeBeams, StandardBlock, StandardCoord
from topologiq.utils.misc import get_manhattan


####################
# INNER PATHFINDER #
####################
class PathFinder:
    """Topologiq's primary pathfinding subroutine."""

    def __init__(
        self,
        bgraph: nx.Graph,
        beams: CubeBeams,
        beams_short: CubeBeams,
        curr_src_id: int,
        curr_tgt_id: int,
        tent_coords: list[StandardCoord],
        cross_edge: bool,
        taken: set[StandardCoord],
        pruned_taken: set[StandardCoord],
        is_hadamard: bool,
        time_deps: tuple[int, int] | None = None,
        **kwargs,
    ):
        """Initialise pathfinder using flat attribute hierarchy.

        Args:
            bgraph_manager: The BlockGraphManager currently managing the build.
            bgraph: The BlockGraph currently being built.
            beams: The beams for all the cubes in blockgraph that need beams.
            beams_short: The short beams for all the cubes in blockgraph that need beams.
            curr_src_id: The ID of the current source cube.
            curr_tgt_id: The ID of the current target cube.
            tent_coords: The list of tentative coords for the placement of the current target.
            cross_edge: True if the current edge is a cross-edge (as opposed to a standard edge).
            taken: The set of taken coordinates.
            pruned_taken: A pruned version of taken not containing source and target coordinates.
            is_hadamard: True if the current edge is a Hadamard in the input ZX graph.
            time_deps (optional): A time-ordering dependency for the target spider (relative to another spider in the graph).
            **kwargs: See `./kwargs.py` for a comprehensive breakdown.
                NB! If an arbitrary kwarg is not given explicitly, it is created against defaults on `./src/topologiq/kwargs.py`.
                NB! By extension, it only makes sense to give the specific kwargs where user wants to deviate from defaults.

        Returns:
            bgraph: An updated blockgraph including the new path to be added to the calling BlockGraphManager.
            pathfinder_vis_data: A list containing data for visualisation of a given pathfinder run.

        """

        # Internalise state of graph manager prior to edge resolution
        self.bgraph = bgraph
        self.beams = beams
        self.beams_short = beams_short
        self.curr_src_id = curr_src_id
        self.curr_tgt_id = curr_tgt_id
        self.tent_coords = tent_coords
        self.cross_edge = cross_edge
        self.taken = taken
        self.pruned_taken = pruned_taken
        self.is_hadamard = is_hadamard
        self.time_deps = time_deps
        self._kwargs = kwargs

        # Extract key info for clarity and readability
        self.src_zx_block: ZXBlock = bgraph.nodes[curr_src_id]["zx_block"]
        self.src_coords: StandardCoord = bgraph.nodes[curr_src_id]["coords"]
        self.src_block_positioned = (self.src_coords, self.src_zx_block)

        self.tgt_zx_block: ZXBlock = bgraph.nodes[curr_tgt_id]["zx_block"]

        self.time_deps_id, self.time_deps_diff = time_deps if time_deps else (None, None)
        self.time_deps_coords = (
            bgraph.nodes[self.time_deps_id]["coords"] if self.time_deps_id else None
        )

        # Parametrise taken
        self.parametrised_taken: dict[int, list[tuple[int, int]]] = {}
        for x, y, z in pruned_taken:
            if z in self.parametrised_taken:
                self.parametrised_taken[z].add((x, y))
            else:
                self.parametrised_taken[z] = set([(x, y)])

        # Generate kinds that could in theory be assigned to the target cube
        self.tent_tgt_kinds = (
            self.tgt_zx_block.kind if cross_edge else self.tgt_zx_block.get_kind_family
        )

        # Top-level trackers for beam clash detection
        self.path_clashes, self.src_tgt_adjusts, self.out_pendings = ({}, {}, {})

        # Create bounding box to limit space search
        self.bounding_box, self.max_span = gen_bounding_box(taken, cross_edge=cross_edge)

    def get_path(self):
        """Create or find a path for the edge being requested."""

        # Initialise BFS
        self.init_bfs()

        # Run BFS search queue
        self.manage_bfs()

        # Pack visualisation data
        vis_data = (
            self.tent_tgt_kinds,
            self.visit_attempts,
            len(self.visited),
            self.all_search_paths,
        )

        return self.valid_paths, vis_data

    def init_bfs(self):
        """Initialise key BFS objects."""

        self.queue = deque([self.src_block_positioned])
        self.visited = {(self.src_block_positioned, (0, 0, 0)): 0}
        self.visit_attempts = 0
        self.path = {self.src_block_positioned: [self.src_block_positioned]}
        self.valid_paths = {}
        self.all_search_paths = {}

        # Define exit conditions in case something goes wrong
        self.tgts_to_fill, self.max_manhattan, self.src_tgt_manhattan = gen_exit_conditions(
            self.src_coords,
            self.tent_coords,
            self.pruned_taken,
            self.max_span,
            self.cross_edge,
            self._kwargs["min_succ_rate"],
            special_target_kind=self.tgt_zx_block.zx_type in ["Y", "T", "XZ"],
        )

    def manage_bfs(self):
        """Manage the pathfinder's main BFS queue."""

        # Create editable Hadamard indicator
        hdm = self.is_hadamard

        # Run BFS queue
        while self.queue:
            # Unpack current block (source for iteration)
            self.curr_block_positioned: PositionedZXBlock = self.queue.popleft()

            # Check skip/break tolerances
            curr_manhattan = get_manhattan(self.src_coords, self.curr_block_positioned[0])
            if curr_manhattan > self.src_tgt_manhattan * 3:
                continue

            # Check for success
            if self.curr_block_positioned[0] in self.tent_coords:
                if self.check_for_success():
                    break
                else:
                    if self.cross_edge or self.curr_block_positioned[1].zx_type in ["Y", "T", "XZ"]:
                        continue

            # Avoid multiple special gates on same path
            if self.tgt_zx_block.zx_type in ["Y", "T", "XZ"]:
                if any(
                    [
                        zx_b.zx_type == self.tgt_zx_block.zx_type
                        for _, zx_b in self.path[self.curr_block_positioned]
                    ]
                ):
                    continue

            # Check path for beam clashes before attempting move
            if self.check_beams_clashes():
                continue

            # Try moving in all directions
            for move in self.curr_block_positioned[1].get_move_vectors:
                # Calculate next position and update paths accordingly
                nxt_coords, curr_path_coords = get_coords_for_current_move(
                    self.curr_block_positioned, move, self.path
                )

                # Check if move can be skipped (for speed)
                if check_skip_move(
                    nxt_coords,
                    self.bounding_box,
                    curr_path_coords,
                    self.parametrised_taken,
                    self.cross_edge,
                    special_target_kind=self.tgt_zx_block.zx_type in ["Y", "T", "XZ"],
                ):
                    continue

                # Create a list of kinds that are valid for the next block
                possible_nxt_zx_blocks = self.curr_block_positioned[1].nxt_kinds(
                    move,
                    is_hadamard=hdm,
                    tgt_zx_type=self.tgt_zx_block.zx_type,
                )

                # Loop over all possible next types
                for possible_nxt_zx_block in possible_nxt_zx_blocks:
                    # Check if next kind needs to be rotated due to Hadamard
                    nxt_block: StandardBlock = (
                        nxt_coords,
                        possible_nxt_zx_block,
                    )

                    # Log to visited and update path lengths if all conditions met
                    self.to_visit_or_not_to_visit(nxt_block, move)

            # Hadamards are introduce on very first move so once loop clears first
            # set of possible moves, there can be no more Hadamards in a specific edge.
            hdm = False

    def check_for_success(self) -> tuple[dict[PositionedZXBlock, list[PositionedZXBlock]], bool]:
        """Check if iteration achieved success."""

        curr_coords, _ = self.curr_block_positioned

        time_deps_check = True
        if self.time_deps_coords and self.time_deps_diff:
            if self.time_deps_diff == -1:
                time_deps_check = self.time_deps_coords[2] > curr_coords[2]
            elif self.time_deps_diff == 1:
                time_deps_check = self.time_deps_coords[2] < curr_coords[2]

            if not time_deps_check:
                return False

        if self.tent_tgt_kinds[0] in ["YYO", "TTO", "XZ*", "ZX*"]:
            if curr_coords in list([c for c, b in self.valid_paths.keys()]):
                return False

            if self.tent_tgt_kinds == ["TTO"]:
                curr_path_coords = [coords for coords, _ in self.path[self.curr_block_positioned]]
                for i in range(1, 4):
                    check_coords = (curr_coords[0], curr_coords[1], curr_coords[2] - i)
                    if check_coords in self.pruned_taken or check_coords in curr_path_coords:
                        return False

            if self.tent_tgt_kinds[0] in ["TTO", "ZX*", "XZ*"]:
                prev_coords = self.path[self.curr_block_positioned][-2][0]
                last_z_diff = abs(curr_coords[2]) - abs(prev_coords[2])
                last_z_diff = 1 if last_z_diff > 0 else -1 if last_z_diff < 0 else 0
                valid_last_diffs = [-1] if self.tent_tgt_kinds[0] == "TTO" else [1]
                if last_z_diff not in valid_last_diffs:
                    return False

        if (
            self.tent_tgt_kinds == ["OOO"]
            or self.curr_block_positioned[1].kind in self.tent_tgt_kinds
        ):
            self.valid_paths[self.curr_block_positioned] = self.path[self.curr_block_positioned]
            tgts_filled = len(set([p[0] for p in self.valid_paths.keys()]))
            if tgts_filled >= self.tgts_to_fill:
                return True

        return False

    def check_beams_clashes(self):
        """Check if there are beam clashes."""

        # For the time being beams are only checked on cross edges
        if not self.cross_edge:
            return False

        # Placeholder to perform strict checks on full edges
        beams_to_check = self.beams.items() if self.cross_edge else self.beams_short.items()

        # Undertake checks
        if self.curr_block_positioned == self.src_block_positioned:
            self.path_clashes[self.curr_block_positioned] = {}
            for out_id, out_beams in beams_to_check:
                self.path_clashes[self.curr_block_positioned][out_id] = np.array(
                    [0 for _ in out_beams]
                )
        else:
            prev_block_positioned = self.path[self.curr_block_positioned][-2]
            self.path_clashes[self.curr_block_positioned] = self.path_clashes[
                prev_block_positioned
            ].copy()

            # Check each cube against all other cubes
            for out_id, out_beams in beams_to_check:
                # Track outer beams in a way that remembers which beam is which

                broken_beams = [
                    1 if out_beam.contains(self.curr_block_positioned[0]) else 0 for out_beam in out_beams
                ]

                self.path_clashes[self.curr_block_positioned][out_id] = np.add(self.path_clashes[
                    prev_block_positioned
                ][out_id], np.array(broken_beams))

                # Determine if out clashes are within tolerance
                if out_id not in self.src_tgt_adjusts:
                    self.src_tgt_adjusts[out_id] = (
                        1 if out_id in (self.curr_src_id, self.curr_tgt_id) else 0
                    )
                if out_id not in self.out_pendings:
                    self.out_pendings[out_id] = (
                        min(1, self.bgraph.nodes[out_id]["completions"]["pending"])
                        if self.src_tgt_adjusts[out_id] == 0
                        else self.bgraph.nodes[out_id]["completions"]["pending"]
                    )
                if (
                    len(out_beams)
                    + self.src_tgt_adjusts[out_id]
                    - self.path_clashes[self.curr_block_positioned][out_id].sum()
                    < self.out_pendings[out_id]
                ):
                    return True

        return False

    def to_visit_or_not_to_visit(self, nxt_block: PositionedZXBlock, move: tuple[int, int, int]):
        """Visit site if conditions are met."""

        # Update counters and add path to all_search_paths
        self.visit_attempts += 1
        self.all_search_paths[nxt_block] = [*self.path[self.curr_block_positioned], nxt_block]

        # Check next coords not in visited or path no longer than equiv. path
        if ((nxt_block, move)) not in self.visited or len(
            self.all_search_paths[nxt_block]
        ) < self.visited[(nxt_block, move)]:
            # Log to visited
            self.visited[(nxt_block, move)] = len(self.all_search_paths[nxt_block])

            # Append to queue
            self.queue.append(nxt_block)

            # Add path
            self.path[nxt_block] = self.all_search_paths[nxt_block]
