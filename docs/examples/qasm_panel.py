"""Example of how to use Topologiq to perform LS on a small collection of QASM circuits.

This script contains an example of how to use Topologiq to perform algorithmic lattice
surgery (LS) on a number of circuits generated randomly in PyZX and saved as QASM.
Outputs are saved to a `.bgraph` file in `./outputs/bgraph/`.

Usage:
    Run script as given.

"""

from pathlib import Path

import pyzx as zx

from topologiq.input.pyzx_manager import ZXGraphManager

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = ROOT_DIR / "src/topologiq/assets"
OUTPUT_DIR = ROOT_DIR / "output/bgraph"


##########
# KWARGS #
##########
kwargs = {
    "first_id_strategy": "centrality_random",
    "seed": None,
    "debug": 0,
}


############
# MAIN RUN #
############
if __name__ == "__main__":
    # Circuits
    circuit_names = ["qasm_random_05_05", "qasm_random_10_10"]

    # Run selected circuits on a loop, without reduction
    for circuit_name in circuit_names:
        # Update user
        print(f"\n===> START. QASM circuit: {circuit_name}")

        # Path to file
        path_to_qasm_file = ASSETS_DIR / f"{circuit_name}.qasm"

        # QASM -> PyZX
        zx_graph_manager = ZXGraphManager()
        aug_zx_in = zx_graph_manager.add_graph_from_qasm(
            path_to_qasm_file=path_to_qasm_file, graph_key=circuit_name
        )
        zx.draw(aug_zx_in.zx_graph)

        # Run Topologiq
        bgraph_manager = aug_zx_in.get_blockgraph(**kwargs)

        # Visualise results
        bgraph_manager.draw_blockgraph()

