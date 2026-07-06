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
# Note. Topologiq auto-completes any KWARGs not given explicitly.
# By extension, you only need to give KWARGs that deviate from default parameters.
# All default parameters are available at the repository root: `src/topologiq/kwargs.py`.
kwargs = {
    "first_id_strategy": "centrality-majority",
    "seed": 19,  # 1330
    "debug": 1,
    "size_of_chip": (12, 12),
    "k": 3,
    "graph_traverse_mode": "tfs-cnots",
    "gravity": 3,
    "z_stretch": 2,
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
    qubit_n, depth = (4, 10)
    circuit_name = f"random_{kwargs['seed'] if kwargs.get('seed') else 'noseed'}_{qubit_n}_{depth}"
    pyzx_graph, _ = random_graph(
        qubit_n,
        depth,
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

    # Write results to file
    bgraph_manager.write_bgraph(circuit_name=circuit_name)

    # Animate and cleanup
    if kwargs.get("animate"):
        bgraph_manager.animate(filename_prefix=circuit_name)
