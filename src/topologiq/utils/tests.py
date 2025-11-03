"""
Contains tests that should eventually move to file-specific tests. 

This file is temporary. 

Usage:
    Run the tests in the file.
"""

import os

from datetime import datetime
from typing import List

from topologiq.scripts import pathfinder
from topologiq.utils.classes import StandardCoord
from topologiq.utils.utils_greedy_bfs import gen_tent_tgt_coords


##############
# PATHFINDER #
##############
def test_pathfinder(
    stats_dir: str, min_succ_rate: int = 60, max_test_step: int = 3, num_repetitions: int = 1
):
    """Checks runtimes for creation of paths by pathfinder algorithm.

    Args:
        - stats_dir: the directory where states are to be saved
        min_succ_rate (optional): Minimum % of tentative coordinates that must be filled for each edge.
        - max_test_step: Sets the maximum distance for target test node
        - num_repetitions: Sets the number of times that the test loop will be repeated.

    """

    if os.path.exists(f"{stats_dir}/pathfinder_tests.csv"):
        os.remove(f"{stats_dir}/pathfinder_tests.csv")

    taken: List[StandardCoord] = []
    hdm: bool = False
    src_coords: StandardCoord = (0, 0, 0)
    all_valid_start_kinds: List[str] = ["zzx", "zxz", "xzz", "xxz", "xzx", "zxx"]
    all_valid_zx_types: List[str] = ["X", "Z"]
    unique_run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "*"

    i = 0
    while i < num_repetitions:
        i += 1
        step: int = 3
        print(f"\nRunning tests. Loop {i} of {num_repetitions}.")
        while step <= max_test_step:
            tent_coords = gen_tent_tgt_coords(src_coords, step, taken)
            for start_kind in all_valid_start_kinds:
                src_block_info = ((0, 0, 0), start_kind)
                for tgt_zx_type in all_valid_zx_types:
                    clean_paths = pathfinder(
                        src_block_info,
                        tent_coords,
                        tgt_zx_type,
                        tgt_block_info=(None, None),
                        taken=taken,
                        hdm=hdm,
                        min_succ_rate=min_succ_rate,
                        log_stats_id=unique_run_id,
                    )
                    if clean_paths:
                        print(".", end="")
            step += 3
