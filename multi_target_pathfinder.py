import time
from typing import List
from scripts.greedy_bfs_traditional import run_pathfinder
from scripts.pathfinder import bfs_extended_3d, run_bfs_for_all_potential_target_nodes
from utils.classes import StandardBlock, StandardCoord

from run_hyperparams import (
    VALUE_FUNCTION_HYPERPARAMS,
    LENGTH_OF_BEAMS,
    MAX_PATHFINDER_SEARCH_SPACE,
)

test_to_run = "run_pathfinder"

# VARS USED BY ALL TESTS
source_node: StandardBlock = ((0, 0, 0), "xzx")
tentative_positions_by_step = {
    3: [(3, 0, 0), (0, 3, 0), (0, 0, 3), (-3, 0, 0), (0, -3, 0), (0, 0, -3)],
    6: [(6, 0, 0), (0, 6, 0), (0, 0, 6), (-6, 0, 0), (0, -6, 0), (0, 0, -6)],
    9: [(9, 0, 0), (0, 9, 0), (0, 0, 9), (-9, 0, 0), (0, -9, 0), (0, 0, -9)],
}

# TEST bfs_extended_3d
if test_to_run == "core_pathfinder":

    target_families = [["xxz", "xzx", "zxx"], ["xzz", "zzx", "zxz"], ["ooo"]]

    for step, tentative_positions in tentative_positions_by_step.items():
        for fam in target_families:
            t1 = time.time()
            valid_paths = bfs_extended_3d(
                source_node,
                tentative_positions,
                fam,
                completion_target=100,
            )
            duration = time.time() - t1

            if valid_paths:
                print(
                    f"\r{source_node} -> {fam}: {len(valid_paths)} paths found in {duration:.2f} secs (step = {step})."
                )
                for path in valid_paths:
                    print("  -> ", path)


# TEST run_bfs_for_all_potential_target_nodes
if test_to_run == "run_core_pathfinder":

    target_node_zx_types = ["X", "Z", "O"]

    for step, tentative_positions in tentative_positions_by_step.items():
        for zx_type in target_node_zx_types:
            t1 = time.time()
            valid_paths = run_bfs_for_all_potential_target_nodes(
                source_node,
                tentative_positions,
                tgt_zx_type=zx_type,
            )

            duration = time.time() - t1

            if valid_paths:
                print(
                    f"\r{source_node} -> {zx_type}: {len(valid_paths)} paths found in {duration:.2f} secs (step = {step})."
                )
                for path in valid_paths:
                    print("  -> ", path)


# TEST run_bfs_for_all_potential_target_nodes
if test_to_run == "run_pathfinder":

    target_node_zx_types = ["X", "Z", "O"]

    kwargs = {
        "weights": VALUE_FUNCTION_HYPERPARAMS,
        "length_of_beams": LENGTH_OF_BEAMS,
        "max_search_space": MAX_PATHFINDER_SEARCH_SPACE,
    }

    for step, tentative_positions in tentative_positions_by_step.items():
        for zx_type in target_node_zx_types:

            t1 = time.time()

            clean_paths = run_pathfinder(
                source_node,
                zx_type,
                3,
                [],
                None,
                False,
                **kwargs,
            )

            duration = time.time() - t1

            if clean_paths:
                print(
                    f"\r{source_node} -> {zx_type}: {len(clean_paths)} paths found in {duration:.2f} secs (step = {step})."
                )
                for path in clean_paths:
                    print("  -> ", path)
