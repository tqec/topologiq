"""Example of how to use Topologiq to perform LS with random PyZX graphs and produce BGRAPH files as output."""

import random
from pathlib import Path

from topologiq.assets.pyzx_graphs import random_graph
from topologiq.input.zx_manager import ZXGraphManager

###############
# WRITE PATHS #
###############
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT_DIR / "output/bgraph"

##########
# KWARGS #
##########
# See `src/topologiq/kwargs.py` for descriptions of each field.
# Do NOT change seed for this script. Some of the post-processing steps
# require specifying the exact ID of a MSC cube (to emulate a control system),
# so if the seed changes those steps will no longer work.
kwargs = {
    "first_id_strategy": "first-spider",
    "seed": 37,
    "debug": 1,
    "size_of_chip": (12, 12),
    "k": 3,
    "graph_traverse_mode": "bfs-layers",
    "gravity": 7,
    "z_stretch": 2,
    "post_process": True,
}

#######
# RUN #
#######
if __name__ == "__main__":

    # Set seed if in KWARGS
    if "seed" in kwargs:
        random.seed(kwargs["seed"])

    # Retrieve circuit
    had_phase = True
    qubit_n, depth = (4, 20)
    circuit_name = f"random_{kwargs['seed'] if kwargs.get('seed') else 'noseed'}_{qubit_n}_{depth}"
    pyzx_graph, _ = random_graph(
        qubit_n,
        depth,
        p_t=0.2,
        draw_graph=True,
        graph_type="cnot_had_phase" if had_phase else "cnot",
        **kwargs,
    )

    # Convert ZX graph into AugmentedZXGraph
    zx_graph_manager = ZXGraphManager()
    aug_zx = zx_graph_manager.add_graph_from_pyzx(pyzx_graph, use_primary=True)

    # Run Topologiq
    bgraph_manager = aug_zx.get_blockgraph(**kwargs)

    # Visualise results
    bgraph_manager.draw_blockgraph()

    # Extend MSCs to maximum available space
    if kwargs["post_process"]:
        bgraph_manager.stretch_msc_cubes()
        bgraph_manager.draw_blockgraph()

    # Add micro-factories where possible
    if kwargs["post_process"]:
        bgraph_manager.distributed_msc_factory()
        bgraph_manager.draw_blockgraph()

    # Example of how to exchange a MSC for a factory
    # Note. Assumes the existence of a control system that
    # would flag a factory success near a MSC that hasn't succeeded
    if kwargs["post_process"]:
        bgraph_manager.msc_exchange(connect_id=261, remove_id=86)
        bgraph_manager.draw_blockgraph()

    # Example of how to stretch the computation to wait for an MSC
    if kwargs["post_process"]:
        bgraph_manager.slice_stretch(slice_at_z=20, shift_z=3)
        bgraph_manager.draw_blockgraph()

    # Example of how to stretch the computation to wait for a conditional
    if kwargs["post_process"]:
        bgraph_manager.slice_stretch(slice_at_z=29, shift_z=3)
        bgraph_manager.draw_blockgraph()

    # Animate and cleanup
    if kwargs.get("animate"):
        bgraph_manager.animate(filename_prefix=circuit_name)
