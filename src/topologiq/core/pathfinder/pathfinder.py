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

import heapq
from collections import deque
from dataclasses import dataclass, replace
from itertools import count
from typing import Any

import networkx as nx
import numpy as np

from topologiq.core.beams import CubeBeams
from topologiq.core.blocks import PositionedZXBlock, ZXBlock, ZXBlockRegistry
from topologiq.core.pathfinder.beams import check_beams_clashes_magic_state
from topologiq.core.pathfinder.spatial import (
    check_clashes_parametrised_taken,
    check_skip_move,
    gen_bounding_box,
    get_coords_current_move,
)
from topologiq.utils.classes import GraphBounds, StandardBlock, StandardCoord
from topologiq.utils.manhattan import get_manhattan, get_max_manhattan


#####################
# PATHFINDER STATE #
####################
@dataclass(frozen=True)
class PathfinderInitState:
    """Snapshot of a BlockGraphManager to use as the initial state of an arbitrary pathfinder iteration.

    Atttributes:
        bgraph: The primary NetworkX graph containing the blockgraph being built.
        beams: The beams for all the cubes in blockgraph that need beams.
        beams_short: The short beams for all the cubes in blockgraph that need beams.
        curr_src_id: The ID of the current source cube.
        curr_tgt_id: The ID of the current target cube.
        tent_coords: The list of tentative coords for the placement of the current target.
        cross_edge: True if the current edge is a cross-edge (as opposed to a standard edge).
        taken: The set of taken coordinates.
        pruned_taken: A pruned version of taken not containing source and target coordinates.
        is_hadamard: True if the current edge is a Hadamard in the input ZX graph.
        z_bounds: Min. and max. Z-coordinate possible for a given move, if either exists.
        graph_bounds (optional): A tuple of max_x and max_y coordinates to maintain build within bounds.
        _kwargs: See `./kwargs.py` for a comprehensive breakdown.
            NB! If an arbitrary kwarg is not given explicitly, it is created against defaults on `./src/topologiq/kwargs.py`.
            NB! By extension, it only makes sense to give the specific kwargs where user wants to deviate from defaults.

    """

    bgraph: nx.Graph
    beams: dict[int, CubeBeams]
    beams_short: dict[int, CubeBeams]
    curr_src_id: int
    curr_tgt_id: int
    tent_coords: list[StandardCoord]
    cross_edge: bool
    taken: set[StandardCoord]
    pruned_taken: set[StandardCoord]
    is_hadamard: bool
    z_bounds: dict[str, int | None]
    graph_bounds: GraphBounds | None
    _kwargs: dict[str, Any]


############################
# MAIN PATHFINDER WORKFLOW #
############################
class PathFinderManager:
    """Manage the process of find paths between blocks in the BlockGraph.

    Note. The A* implementation was coded using AI based on the BFS implementation.
    Feel free to improve A* using or not using AI, but please keep any work done
    to the BFS and BFS helper classes exclusively non-AI. Also, please do NOT merge
    BFS and A* helper methods for... secret reasons.

    """

    def __init__(self, pathfinder_init_state: PathfinderInitState):
        """Initialise with empty blockgraph."""

        # Internalise state for facilitated access
        self.s: PathfinderInitState = pathfinder_init_state

    def pathfinder_bfs(
        self, stretch: bool = False
    ) -> dict[PositionedZXBlock, list[PositionedZXBlock]] | None:
        """Find paths using a BFS algorithm.

        Returns:
            valid_paths: All valid paths found.
            stretch: Run is part of a stretch operation.

        """
        # Extract key info into easily accessible variables
        self.src_zx_block: ZXBlock = self.s.bgraph.nodes[self.s.curr_src_id]["zx_block"]
        self.src_coords: StandardCoord = self.s.bgraph.nodes[self.s.curr_src_id]["coords"]
        self.tgt_zx_block: ZXBlock = self.s.bgraph.nodes[self.s.curr_tgt_id]["zx_block"]

        # Parametrise taken (creates: self.parametrised_taken)
        self._parametrise_taken_init()

        # Generate kinds that could in theory be assigned to the target cube
        self.tent_tgt_kinds = (
            self.tgt_zx_block.kind if self.s.cross_edge else self.tgt_zx_block.get_fulfillment_kinds
        )

        # Create bounding box to limit search space (creates: self.bounding_box, self.max_span)
        self._gen_bounding_box()

        # Initialise BFS (creates: self.queue, self.visited, self.path, & trackers various)
        self._init_bfs()

        # Last recourse exit conditions (creates: self.tgts_to_fill, self.max_manhattan, self.src_tgt_manhattan)
        self._gen_exit_conditions()

        # Manage queue
        hdm, break_for_success = (self.s.is_hadamard, False)
        while self.queue:
            # Flag to exit prematurely due to success
            if break_for_success:
                break

            # Unpack current block (source for iteration)
            self.curr_block_positioned: PositionedZXBlock = self.queue.popleft()
            self.curr_coords: StandardCoord = self.curr_block_positioned[0]
            self.curr_zx_block: ZXBlock = self.curr_block_positioned[1]
            self.curr_path = self.path[self.curr_block_positioned]

            # Check distance tolerances
            if self._check_distance_breach():
                continue

            # Avoid overshooting path and appending multiple special gates to same path
            if self._check_special_cases():
                continue

            # Check path for beam clashes before attempting move
            if self._check_beams_clashes():
                continue

            # Try moving in all directions
            for move in self.curr_zx_block.get_move_vectors:
                if stretch and move != (0, 0, 1):
                    continue

                # Calculate next position and update paths accordingly
                nxt_coords, curr_path_coords = get_coords_current_move(
                    self.curr_block_positioned, move, self.path
                )

                # Check if move can be skipped altogether
                if not stretch:
                    if self._check_skip_move(nxt_coords, curr_path_coords):
                        continue

                # Create a list of kinds that are valid for the next block
                possible_nxt_zx_blocks = self.curr_zx_block.nxt_kinds(move, is_hadamard=hdm)

                # Loop over all possible next types
                for possible_nxt_zx_block in possible_nxt_zx_blocks:
                    # Check if next kind needs to be rotated due to Hadamard
                    nxt_block: StandardBlock = (nxt_coords, possible_nxt_zx_block)

                    # Check clashes for extended coords of MSC blocks
                    if self._check_special_t_clash(nxt_block, curr_path_coords):
                        continue

                    # Log to visited and update path lengths if all conditions met
                    self._to_visit_or_not_to_visit(nxt_block, move)

                    # Check for success
                    if nxt_coords in self.s.tent_coords:
                        if self._check_for_success(nxt_block):
                            break_for_success = True
                            break

                if break_for_success:
                    break

            # Hadamards must be placed in very first move only.
            hdm = False

        return self.valid_paths

    def _parametrise_taken_init(self):
        """Parametrise taken for easier retrieval and check."""
        self.parametrised_taken = {}
        for x, y, z in self.s.pruned_taken:
            if z in self.parametrised_taken:
                self.parametrised_taken[z].add((x, y))
            else:
                self.parametrised_taken[z] = set([(x, y)])

    def _gen_bounding_box(self):
        """Determine min/max coordinates for any second pass search."""
        self.bounding_box, self.max_span = gen_bounding_box(
            self.s.taken, self.s.graph_bounds, self.s.cross_edge
        )

    def _init_bfs(self):
        """Initialise BFS variables."""
        self.src_block_positioned = (self.src_coords, self.src_zx_block)
        self.queue = deque([self.src_block_positioned])
        self.visited = {(self.src_block_positioned, (0, 0, 0)): 0}
        self.visit_attempts = 0
        self.path = {self.src_block_positioned: [self.src_block_positioned]}
        self.valid_paths, self.all_search_paths = ({}, {})
        self.path_clashes, self.src_tgt_adjusts, self.out_pendings = ({}, {}, {})

    def _gen_exit_conditions(self):
        """Calculate conditions that need to be met to exit the pathfinder."""

        # Cross-edges
        if self.s.cross_edge:
            self.tgts_to_fill = 1
            self.src_tgt_manhattan = get_max_manhattan(self.src_coords, self.s.tent_coords)
            if self.s.cross_edge:
                self.max_manhattan = max(
                    get_max_manhattan(self.src_coords, self.s.taken) * 2,
                    self.max_span,
                )
            else:
                self.max_manhattan = self.src_tgt_manhattan + 6
        # Standard edges
        else:
            self.tgts_to_fill = int(len(self.s.tent_coords) * self.s._kwargs["min_succ_rate"] / 100)
            self.max_manhattan = get_max_manhattan(self.src_coords, self.s.tent_coords) * 2
            self.src_tgt_manhattan = self.max_manhattan

        # Final calculations
        self.src_tgt_manhattan = get_max_manhattan(self.src_coords, self.s.tent_coords)
        self.tgts_to_fill = min(10, self.tgts_to_fill)

    def _check_distance_breach(self) -> bool:
        """Check if search has not gone past a maximum tolerated distance."""
        curr_manhattan = get_manhattan(self.src_coords, self.curr_coords)
        if curr_manhattan > self.src_tgt_manhattan * 3:
            return True
        if curr_manhattan > self.max_manhattan:
            return True
        return False

    def _check_special_cases(self) -> bool:
        """Check if there is already a special gate in the current path."""
        if self.curr_coords in self.s.tent_coords:
            if self.s.cross_edge or self.curr_zx_block.zx_type in ["Y", "T", "O"]:
                return True
        if self.tgt_zx_block.zx_type in ["Y", "T"]:
            if any([zx_b.zx_type == self.tgt_zx_block.zx_type for _, zx_b in self.curr_path[1:]]):
                return True
        return False

    def _check_beams_clashes(self) -> bool:
        """Check if there are beam clashes."""

        # Do not check if it is a cross edge
        if not self.s.cross_edge:
            return False

        if len(self.path[self.curr_block_positioned]) > 1:
            self.path_clashes[self.curr_block_positioned] = self.path_clashes[
                self.path[self.curr_block_positioned][-2]
            ].copy()

            # Check each cube against all other cubes
            for out_id, out_beams in self.s.beams.items():
                # Track outer beams in a way that remembers which beam is which
                broken_beams = [
                    out_beam.contains(self.curr_block_positioned[0]) for out_beam in out_beams
                ]

                self.path_clashes[self.curr_block_positioned][out_id] = self.path_clashes[
                    self.path[self.curr_block_positioned][-2]
                ][out_id] + np.array(broken_beams)

                # Determine if out clashes are within tolerance
                if out_id not in self.src_tgt_adjusts:
                    self.src_tgt_adjusts[out_id] = (
                        1 if out_id in (self.s.curr_src_id, self.s.curr_tgt_id) else 0
                    )
                if out_id not in self.out_pendings:
                    self.out_pendings[out_id] = (
                        min(1, self.s.bgraph.nodes[out_id]["completions"]["pending"])
                        if self.src_tgt_adjusts[out_id] == 0
                        else self.s.bgraph.nodes[out_id]["completions"]["pending"]
                    )
                if (
                    len(out_beams)
                    + self.src_tgt_adjusts[out_id]
                    - self.path_clashes[self.curr_block_positioned][out_id].sum()
                    < self.out_pendings[out_id]
                ):
                    return True

        else:
            self.path_clashes[self.curr_block_positioned] = {}
            for out_id, out_beams in self.s.beams.items():
                self.path_clashes[self.curr_block_positioned][out_id] = np.array(
                    [False for _ in out_beams]
                )

        return False

    def _check_skip_move(
        self, nxt_coords: StandardCoord, curr_path_coords: list[StandardCoord]
    ) -> bool:
        """Check if a particular move can be skipped altogether."""
        return check_skip_move(
            nxt_coords, curr_path_coords, self.parametrised_taken, bounding_box=self.bounding_box
        )

    def _check_special_t_clash(
        self,
        nxt_block: PositionedZXBlock,
        curr_path_coords: list[StandardCoord],
    ) -> bool:
        nxt_coords, possible_nxt_zx_block = nxt_block
        if possible_nxt_zx_block.zx_type == "T":
            nxt_x, nxt_y, nxt_z = nxt_coords
            for i in [1, 2]:
                if check_clashes_parametrised_taken(
                    (nxt_x, nxt_y, nxt_z - i), self.parametrised_taken
                ):
                    return True
                if (nxt_x, nxt_y, nxt_z - i) in self.s.taken or (
                    nxt_x,
                    nxt_y,
                    nxt_z - i,
                ) in curr_path_coords:
                    return True
        return False

    def _to_visit_or_not_to_visit(self, nxt_block: PositionedZXBlock, move: tuple[int, int, int]):
        """Visit site if conditions are met.

        Args:
            nxt_block: The positioned ZX block for the next block.
            move: The spatial displacement (aka. move) currently under consideration.

        """

        # Update counters and add path to all_search_paths
        self.visit_attempts += 1
        self.all_search_paths[nxt_block] = [*self.path[self.curr_block_positioned], nxt_block]

        # Check next coords not in visited or path no longer than equiv. path
        if (nxt_block, move) not in self.visited:
            # Log to visited
            self.visited[(nxt_block, move)] = len(self.all_search_paths[nxt_block])

            # Append to queue
            self.queue.append(nxt_block)

            # Add path
            self.path[nxt_block] = self.all_search_paths[nxt_block]

    def _check_for_success(self, nxt_block_positioned: PositionedZXBlock) -> bool:
        """Check if iteration achieved success.

        Args:
            nxt_block_positioned: The positioned ZX block (coordinates, zx_block).

        Return:
            [bool]: True if success was achieved in this iteration, else False.

        """

        # Extract last block in path
        nxt_coords, _ = nxt_block_positioned

        # Separate checks into categories for readability
        fail_time_constraints = False
        fail_special_cube_constraints = False

        # Check time constraints
        if self.s.z_bounds.get("min"):
            fail_time_constraints = self.s.z_bounds["min"] >= nxt_coords[2]

        if not fail_time_constraints and self.s.z_bounds.get("max"):
            fail_time_constraints = self.s.z_bounds["max"] <= nxt_coords[2]

        if fail_time_constraints:
            return False

        # Check special cube conditions
        if self.tgt_zx_block.zx_type in ["Y", "T", "XZ", "O"]:
            fail_special_cube_constraints = nxt_coords in list(
                [c for c, _ in self.valid_paths.keys()]
            )
            if self.tgt_zx_block.zx_type == "T":
                curr_path_coords = [coords for coords, _ in self.path[nxt_block_positioned]]
                all_magic_coords = [
                    (nxt_coords[0], nxt_coords[1], nxt_coords[2] - i) for i in range(1, 4)
                ]
                fail_special_cube_constraints = any(
                    [
                        (check_coords in self.s.pruned_taken or check_coords in curr_path_coords)
                        for check_coords in all_magic_coords
                    ]
                )
                if (
                    not fail_special_cube_constraints
                    and self.s.bgraph
                    and self.s.beams_short
                    and self.s.curr_src_id
                    and self.s.curr_tgt_id
                ):
                    fail_special_cube_constraints = check_beams_clashes_magic_state(
                        self.s.bgraph,
                        all_magic_coords,
                        self.s.beams_short,
                        self.s.curr_src_id,
                        self.s.curr_tgt_id,
                    )

        if fail_special_cube_constraints:
            return False

        if self.tgt_zx_block.zx_type == "Y":
            if self.s._kwargs["twisted_y"]:
                move_geometry_check = self.path[nxt_block_positioned][-2][0][2] == nxt_coords[2]
            else:
                move_geometry_check = self.path[nxt_block_positioned][-2][0][2] != nxt_coords[2]
            if not move_geometry_check:
                fail_special_cube_constraints = True

        if self.tgt_zx_block.zx_type == "T":
            move_geometry_check = self.path[nxt_block_positioned][-2][0][2] > nxt_coords[2]
            if not move_geometry_check:
                fail_special_cube_constraints = True

        if self.tgt_zx_block.zx_type == "XZ":
            move_geometry_check = self.path[nxt_block_positioned][-2][0][2] < nxt_coords[2]
            if not move_geometry_check:
                fail_special_cube_constraints = True

        if fail_special_cube_constraints:
            return False

        # For all cases, return true only if standard checks clear
        if self.tgt_zx_block.zx_type == "O" or nxt_block_positioned[1].kind in self.tent_tgt_kinds:
            if self.tgt_zx_block.zx_type in ["XZ", "T", "Y"]:
                if self.tgt_zx_block.zx_type == "XZ":
                    nxt_kind = nxt_block_positioned[1].kind[:2] + "*"
                else:
                    nxt_kind = (self.tgt_zx_block.zx_type * 2) + "O"
                xz_block = ZXBlockRegistry.get_create(kind=nxt_kind)
                self.path[(nxt_coords, xz_block)] = [
                    *self.path[nxt_block_positioned][:-1],
                    (nxt_coords, xz_block),
                ]
                self.valid_paths[(nxt_coords, xz_block)] = self.path[(nxt_coords, xz_block)]
            else:
                if nxt_block_positioned not in self.valid_paths or len(
                    self.path[nxt_block_positioned]
                ) < len(self.valid_paths[nxt_block_positioned]):
                    self.valid_paths[nxt_block_positioned] = self.path[nxt_block_positioned]

            if self.tgt_zx_block.zx_type not in ["O"]:
                tgts_filled = len([p[0] for p in self.valid_paths.keys()])
                if tgts_filled >= self.tgts_to_fill:
                    return True

        return False

    def pathfinder_a_star(
        self, stretch: bool = False
    ) -> dict[PositionedZXBlock, list[PositionedZXBlock]] | None:
        """Find paths using an A* algorithm optimised for single-target cross-edges.

        Returns:
            valid_paths: All valid paths found.
            stretch: Run is part of a stretch operation.

        AI disclaimer:
            category: Coding partner (see CONTRIBUTING.md for details).
            model: Gemini 2.5 Flash.

        """
        # Extract key info into easily accessible variables
        self.src_zx_block: ZXBlock = self.s.bgraph.nodes[self.s.curr_src_id]["zx_block"]
        self.src_coords: StandardCoord = self.s.bgraph.nodes[self.s.curr_src_id]["coords"]
        self.tgt_zx_block: ZXBlock = self.s.bgraph.nodes[self.s.curr_tgt_id]["zx_block"]
        self.tgt_coords: StandardCoord = self.s.tent_coords[0]

        # Parametrise taken
        self._parametrise_taken_init()

        # Generate kinds that could in theory be assigned to the target cube
        self.tent_tgt_kinds = (
            self.tgt_zx_block.kind if self.s.cross_edge else self.tgt_zx_block.get_fulfillment_kinds
        )

        # Create bounding box to limit search space
        self._gen_bounding_box()

        # Initialise A* priority queue and tracking structures
        self._init_a_star()

        # Last recourse exit conditions
        self._gen_exit_conditions()

        # Manage pPriority queue
        hdm, break_for_success = (self.s.is_hadamard, False)
        while self.heap:
            if break_for_success:
                break

            # Unpack block with lowest f_score (f = g + h)
            _, curr_g, _, self.curr_block_positioned = heapq.heappop(self.heap)
            self.curr_coords: StandardCoord = self.curr_block_positioned[0]
            self.curr_zx_block: ZXBlock = self.curr_block_positioned[1]

            # Retrieve path for popped node
            self.curr_path = self.path[self.curr_block_positioned]

            # A* distance breach check (f-score / detour budget)
            if self._check_distance_breach_a_star(curr_g):
                continue

            # Avoid overshooting path and appending multiple special gates to same path
            if self._check_special_cases():
                continue

            # Check path for beam clashes before attempting move
            if self._check_beams_clashes():
                continue

            # Try moving in all directions
            for move in self.curr_zx_block.get_move_vectors:
                if stretch and move != (0, 0, 1):
                    continue
                # Calculate next position and update paths accordingly
                nxt_coords, curr_path_coords = get_coords_current_move(
                    self.curr_block_positioned, move, self.path
                )

                # Check if move can be skipped altogether
                if not stretch:
                    if self._check_skip_move(nxt_coords, curr_path_coords):
                        continue

                # Create a list of kinds that are valid for the next block
                possible_nxt_zx_blocks = self.curr_zx_block.nxt_kinds(move, is_hadamard=hdm)

                # Loop over all possible next types
                for possible_nxt_zx_block in possible_nxt_zx_blocks:
                    nxt_block: StandardBlock = (nxt_coords, possible_nxt_zx_block)

                    # Check clashes for extended coords of MSC blocks
                    if self._check_special_t_clash(nxt_block, curr_path_coords):
                        continue

                    # Evaluate score and add to priority queue if valid
                    self._to_visit_or_not_to_visit_a_star(nxt_block, move, curr_g)

                    # Check for success
                    if nxt_coords in self.s.tent_coords:
                        if self._check_for_success(nxt_block):
                            break_for_success = True
                            break

                if break_for_success:
                    break

            # Hadamards must be placed in very first move only
            hdm = False

        return self.valid_paths

    def _check_distance_breach_a_star(self, curr_g: int) -> bool:
        """Check if search has exceeded allowed distance tolerances."""

        # Naive distance checks for cross edges to maximise routing freedom
        if self.s.cross_edge:
            curr_manhattan = get_manhattan(self.src_coords, self.curr_coords)
            if curr_manhattan > self.src_tgt_manhattan * 3:
                return True
            if curr_manhattan > self.max_manhattan:
                return True
            return False

        # Fast detour budget (estimated total cost: f = g + h) for standard edges
        else:
            curr_h = get_manhattan(self.curr_coords, self.tgt_coords)
            curr_f = curr_g + curr_h

            detour_budget = self.s._kwargs.get("a_star_detour_budget", 8)
            if curr_f > self.src_tgt_manhattan + detour_budget:
                return True

            if get_manhattan(self.src_coords, self.curr_coords) > self.max_manhattan:
                return True

            return False

    def _init_a_star(self):
        """Initialise A* data structures and adaptive heuristic weights."""
        self.src_block_positioned = (self.src_coords, self.src_zx_block)
        self.heap_counter = count()

        initial_h = get_manhattan(self.src_coords, self.tgt_coords)

        # Adaptive weighting: For long distances (>= 15), weight heuristic higher
        # to drive beam directly to target without exploring huge 3D search space
        if initial_h >= 15:
            default_weight = 1.5
        else:
            default_weight = 1.1

        self.h_weight = self.s._kwargs.get("a_star_weight", default_weight)
        initial_g = 0
        initial_f = initial_g + (self.h_weight * initial_h)

        self.heap = []
        heapq.heappush(
            self.heap,
            (initial_f, initial_g, next(self.heap_counter), self.src_block_positioned),
        )

        self.g_scores = {(self.src_block_positioned, (0, 0, 0)): 0}
        self.visit_attempts = 0

        if self.s.cross_edge:
            self.max_visits = self.s._kwargs.get("a_star_max_visits_cross", 50000)
        else:
            self.max_visits = self.s._kwargs.get("a_star_max_visits_standard", 3000)

        self.path = {self.src_block_positioned: [self.src_block_positioned]}
        self.valid_paths = {}
        self.path_clashes, self.src_tgt_adjusts, self.out_pendings = ({}, {}, {})

    def _to_visit_or_not_to_visit_a_star(
        self,
        nxt_block: PositionedZXBlock,
        move: tuple[int, int, int],
        curr_g: int,
    ):
        """Evaluate next block state and push to heap if lower g_score found."""
        self.visit_attempts += 1

        if self.visit_attempts > self.max_visits:
            self.heap.clear()
            return

        nxt_coords = nxt_block[0]
        nxt_g = curr_g + 1
        state_key = (nxt_block, move)

        if state_key not in self.g_scores or nxt_g < self.g_scores[state_key]:
            self.g_scores[state_key] = nxt_g

            # Store as list so _check_for_success works seamlessly
            self.path[nxt_block] = [*self.path[self.curr_block_positioned], nxt_block]

            # Weighted f-score f(n) = g(n) + w * h(n)
            # w = 1.5 forces A* to move straight to target when distance > 15
            h_score = get_manhattan(nxt_coords, self.tgt_coords)
            f_score = nxt_g + (self.h_weight * h_score)

            heapq.heappush(
                self.heap,
                (f_score, nxt_g, next(self.heap_counter), nxt_block),
            )

    def pathfinder_a_star_multi_target(
        self,
    ) -> dict[PositionedZXBlock, list[PositionedZXBlock]] | None:
        """Find paths using single-target A* iterated over multiple candidate target coordinates.

        AI disclaimer:
            category: Coding partner (see CONTRIBUTING.md for details).
            model: Gemini 2.5 Flash.

        Returns:
            accumulated_paths: All valid paths found for targets meeting success rate criteria.

        """
        original_state = self.s
        src_coords: StandardCoord = original_state.bgraph.nodes[original_state.curr_src_id][
            "coords"
        ]

        # Calculate global success requirement based on the full original state
        if original_state.cross_edge:
            global_tgts_to_fill = 1
        else:
            global_tgts_to_fill = int(
                len(original_state.tent_coords) * original_state._kwargs["min_succ_rate"] / 100
            )
            global_tgts_to_fill = min(10, max(1, global_tgts_to_fill))

        # Sort targets by initial Manhattan distance (try closest candidates first)
        sorted_tent_coords = sorted(
            original_state.tent_coords,
            key=lambda tgt: get_manhattan(src_coords, tgt),
        )

        accumulated_paths: dict[PositionedZXBlock, list[PositionedZXBlock]] = {}

        try:
            for candidate_tgt in sorted_tent_coords:
                # Temporarily set single candidate target
                self.s = replace(original_state, tent_coords=[candidate_tgt])

                # Run single-target A*
                valid_paths = self.pathfinder_a_star()

                if valid_paths:
                    accumulated_paths.update(valid_paths)

                    # Stop once the required target quota is satisfied
                    if len(accumulated_paths) >= global_tgts_to_fill:
                        break
        finally:
            # Restore original frozen state
            self.s = original_state

        return accumulated_paths if accumulated_paths else None
