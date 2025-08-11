import os

from typing import List
from datetime import datetime

from scripts.pathfinder import pthfinder
from utils.classes import StandardCoord
from utils.utils_greedy_bfs import gen_tent_tgt_coords

def test_pthfinder(
    stats_dir: str, min_succ_rate, max_test_step: int = 3, num_repetitions: int = 1
):
    """Checks runtimes for creation of paths by pathfinder algorithm.

    Args:
        - stats_dir: the directory where states are to be saved
        - min_succ_rate: min % of tent_coords that need to be filled, used as exit condition.
        - max_test_step: Sets the maximum distance for target test node
        - num_repetitions: Sets the number of times that the test loop will be repeated.

    """

    if os.path.exists(f"{stats_dir}/pathfinder_iterations_tests.csv"):
        os.remove(f"{stats_dir}/pathfinder_iterations_tests.csv")

    taken: List[StandardCoord] = []
    hdm: bool = False
    src_coords: StandardCoord = (0, 0, 0)
    all_valid_start_kinds: List[str] = ["xxz", "xzx", "zxx", "xzz", "zzx", "zxz"]
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
                src = ((0, 0, 0), start_kind)
                for zx_type in all_valid_zx_types:
                    clean_paths = pthfinder(
                        src,
                        tent_coords,
                        zx_type,
                        tgt=(None, None),
                        taken=taken,
                        hdm=hdm,
                        min_succ_rate=min_succ_rate,
                        log_stats_id=unique_run_id,
                    )
                    if clean_paths:
                        print(".", end="")
            step += 3
