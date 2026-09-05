"""Example of how to use Topologiq with QASM circuits.

Note. Special thanks to Kabir Dubey, Purva Thakre, Yilun Zhao,
    and David Yonge-Mallo, who have all contributed to enabling a robust
    QASM -> PyZX pipeline that Topologiq can rely upon.

"""

from pathlib import Path

import pyzx as zx

from topologiq.core.graph_manager.graph_manager import BlockGraphManager
from topologiq.input.zx_manager import ZXGraphManager
from topologiq.utils.zx import apply_bialgebra, rm_unnecessary_phases

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = ROOT_DIR / "src/topologiq/assets"
OUTPUT_DIR = ROOT_DIR / "output/bgraph"


##########
# KWARGS #
##########
# Note. Topologiq auto-completes any KWARGs not given explicitly.
# By extension, you only need to give KWARGs that deviate from default parameters.
# All default parameters are available at the repository root: `src/topologiq/kwargs.py`.
kwargs = {"debug": 0}


#######
# RUN #
#######
if __name__ == "__main__":
    # Circuits
    circuit_names = ["qasm_random_05_05.qasm", "qasm_random_10_10.qasm", "qasm_steane.qasm"]

    # Run selected circuits on a loop, without reduction
    for circuit_name in circuit_names:
        # Update user
        print(f"\n===> START. QASM circuit: {circuit_name}")

        # Path to file
        path_to_qasm_file = ASSETS_DIR / f"{circuit_name}"

        # QASM -> PyZX
        zx_graph_manager = ZXGraphManager()
        aug_zx = zx_graph_manager.add_graph_from_qasm(
            path_to_qasm_file=path_to_qasm_file, graph_key=circuit_name
        )

        # All graphs are now loaded and would build already
        # But we will apply some transformations to the Steane to ensure:
        # - transformations on the loaded ZX graph are possible
        # - final outputs match what we get elsewhere with manually-encoded Steane graphs.
        if "steane" in circuit_name:
            # Copy the graph instead of reassigning it to get consecutive IDs
            steane_reduced_zx = aug_zx.zx_graph_reduced.copy()

            # Turn to RG
            zx.to_rg(steane_reduced_zx)

            # Remove phases not needed for LS
            rm_unnecessary_phases(steane_reduced_zx)

            # Apply an extended bi-algebra transformation
            apply_bialgebra(steane_reduced_zx)

            # Now create a new Augmented ZX Graph containing all above transformations
            aug_zx = zx_graph_manager.add_graph_from_pyzx(
                steane_reduced_zx, graph_key=f"{circuit_name}_reduced"
            )

        # AugmentedZXGraph -> BlockGraph
        # Kwargs
        if "steane" in circuit_name:
            kwargs["graph_traverse_mode"] = "bfs-cycles"
            kwargs["twins"] = False
            kwargs["gravity"] = 7
        else:
            kwargs["graph_traverse_mode"] = "bfs-cross"
            kwargs["twins"] = True
            kwargs["gravity"] = 7
        bgraph_manager = BlockGraphManager(aug_zx, **kwargs)
        bgraph_manager.build()

        # Visualise results
        bgraph_manager.draw_blockgraph()
