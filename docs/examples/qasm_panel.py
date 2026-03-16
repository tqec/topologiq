"""Example of how to use Topologiq to perform LS on a small collection of QASM circuits.

This script contains an example of how to use Topologiq to perform algorithmic lattice
surgery (LS) on a number of circuits generated randomly in PyZX and saved as QASM.
Outputs are saved to a `.bgraph` file in `./outputs/bgraph/`.

Usage:
    Run script as given.

"""

from pathlib import Path

import pyzx as zx

from topologiq.core.graph_manager.graph_manager import runner
from topologiq.input.pyzx_manager import ZXGraphManager, pyzx_g_to_simple_g
from topologiq.utils.classes import StandardBlock

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = ROOT_DIR / "src/topologiq/assets"
OUTPUT_DIR = ROOT_DIR / "output/bgraph"


def run_single_qasm(
    circuit_name: str,
    use_reduced: bool = False,
    draw_graph: bool = False,
    **kwargs,
) -> tuple[dict[int, StandardBlock] | None, dict[tuple[int, int], list[str]] | None]:
    """Call Topologiq to perform algorithmic lattice surgery on circuit.

    Args:
        circuit_name: The name of the circuit.
        use_reduced (optional): Whether to optimise/reduce the circuit before running it or not.
        draw_graph: Whether to pop-up PyZX graph visualisation or not.
        **kwargs: See `./kwargs.py` for a comprehensive breakdown.
            NB! If an arbitrary kwarg is not given explicitly, it is created against defaults on `./src/topologiq/kwargs.py`.
            NB! By extension, it only makes sense to give the specific kwargs where user wants to deviate from defaults.

    """

    # Path to file
    path_to_qasm_file = ASSETS_DIR / f"{circuit_name}.qasm"

    # QASM -> PyZX
    zx_graph_manager = ZXGraphManager()
    aug_zx = zx_graph_manager.add_graph_from_qasm(
        path_to_qasm_file=path_to_qasm_file, graph_key=circuit_name
    )

    if draw_graph:
        zx.draw(aug_zx.zx_graph_reduced if use_reduced else aug_zx.zx_graph)

    # Convert to simple graph
    circuit_name = f"{circuit_name}_min" if use_reduced else circuit_name
    simple_graph = pyzx_g_to_simple_g(aug_zx.zx_graph_reduced if use_reduced else aug_zx.zx_graph)
    _, _, _, _ = runner(simple_graph, circuit_name, **kwargs)


# ...
if __name__ == "__main__":
    # Circuits
    print("\n===> START. QASM Panel.")
    circuit_names = ["qasm_random_05_05", "qasm_random_10_10"]

    # KWARGs
    # KWARGs not included here is autocompleted on run.
    kwargs = {
        "first_id_strategy": "centrality_random",
        "seed": None,
        "vis_options": ("final", None),
        "max_attempts": 1,  # Run 10 tests for each circuit
        "stop_on_first_success": False,  # Do NOT stop after success (if True, this setting overrides max_attempts)
        "debug": 0,
        "log_stats": False,
    }

    # Run selected circuits on a loop, without reduction
    for circuit_name in circuit_names:
        run_single_qasm(circuit_name, **kwargs)
