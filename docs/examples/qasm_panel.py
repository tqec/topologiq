"""Example of how to use Topologiq with QASM circuits."""

from pathlib import Path

import pyzx as zx

from topologiq.input.zx_manager import ZXGraphManager

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = ROOT_DIR / "src/topologiq/assets"
OUTPUT_DIR = ROOT_DIR / "output/bgraph"


##########
# KWARGS #
##########
# Note. Topologiq auto-completes any KWARGs not given explicitly.
# By extension, you only need to give KWARGs that deviate from default parameters.
# All default parameters are available at the repository root: `src/topologiq/kwargs.py`.
kwargs = {"debug": 1}


#######
# RUN #
#######
if __name__ == "__main__":
    # Circuits
    circuit_names = ["qasm_random_05_05.qasm", "qasm_random_10_10.qasm"]

    # Run selected circuits on a loop, without reduction
    for circuit_name in circuit_names:
        # Update user
        print(f"\n===> START. QASM circuit: {circuit_name}")

        # Path to file
        path_to_qasm_file = ASSETS_DIR / f"{circuit_name}"

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
