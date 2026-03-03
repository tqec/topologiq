"""Test the inner pathfinder algorithm.

This file allows testing the inner pathfinder algorithm with a combination of
source and target kinds/zx types. Using the default setup, the pathfinder should
return 4 valid paths for any given combination of src_kind and tgt_zx_type.

Usage:
    Run the file to trigger tests.
"""

from topologiq.core.graph_manager.callers import _gen_tent_tgt_coords
from topologiq.core.graph_manager.kwargs import check_assemble_kwargs
from topologiq.core.pathfinder.pathfinder import pathfinder
from topologiq.utils.classes import StandardCoord
from topologiq.utils.core import datetime_manager


##############
# PATHFINDER #
##############
def test_pathfinder(src_kinds: list[str], tgt_zx_types: list[str], step: int = 3):
    """Checks runtimes for creation of paths by pathfinder algorithm.

    Args:
        src_kinds: List of source kinds to test.
        tgt_zx_types: List of target ZX types to test.
        step (optional): Manhattan distance between src and tgt cubes.

    """

    print("==> TEST STARTS.\n")

    # Define foundational test parameters
    taken: list[StandardCoord] = []
    hdm: bool = False
    src_coords: StandardCoord = (0, 0, 0)

    # Assemble KWARGs
    kwargs = {
        "seed": None,  # (None | int) Change to use a specific random seed across the entire algorithm
        "max_attempts": 1,  # (int) Change to limit the max number of runs for any given circuit
        "min_succ_rate": 100,  # % of paths needed to consider run successful (len(tent_coords)/len(valid paths))
        "debug": 0,  # (int: 0, 1, 2, 3) Change to turn debug mode on, with increasing level of stringency
    }
    kwargs = check_assemble_kwargs(**kwargs)

    # Generate tentative coords at max distance
    tent_coords = _gen_tent_tgt_coords(
        src_coords,
        step,
        taken,
    )
    print(f"Sending {len(tent_coords)} tentative coordinates to pathfinder.")

    # Run test on all src/tgt combinations
    if tent_coords:
        for src_kind in src_kinds:
            src_block = (src_coords, src_kind)
            for tgt_zx_type in tgt_zx_types:
                # Start timer for individual test
                t_1, _ = datetime_manager()
                print("Testing", src_block, "-->", tgt_zx_type)

                valid_paths, _ = pathfinder(
                    src_block,
                    tent_coords,
                    tgt_zx_type,
                    hdm=hdm,
                    src_tgt_ids=(0, 1),
                    **kwargs,
                )

                # End timer for individual test
                _, duration_iter = datetime_manager(t_1=t_1)

                print(
                    f"==> TEST ENDS. Received: {len(valid_paths)} valid paths. Duration: {duration_iter * 1000:.2f}ms\n",
                )


if __name__ == "__main__":
    src_kinds = ["zzx", "zxz", "xzz", "xxz", "xzx", "zxx"]
    tgt_zx_types = ["X", "Z"]
    step = 3
    test_pathfinder(src_kinds, tgt_zx_types, step=step)
