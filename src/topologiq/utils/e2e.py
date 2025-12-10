"""Util facilities to facilitate benchmarking and end-to-end testing.

Usage:
    Call any function/class from a separate script.

"""

import os
import random
from datetime import datetime
from pathlib import Path

import pyzx as zx

from topologiq.scripts.runner import runner
from topologiq.utils.classes import StandardBlock
from topologiq.utils.interop_pyzx import pyzx_g_to_simple_g

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
DATA_DIR = ROOT_DIR / "benchmark/data"
OUTPUT_DIR = ROOT_DIR / "output/bgraph"


#################
# FLOW MANAGERS #
#################
def test_qasm_circuit(
    circuit_name: str,
    reduce: bool = False,
    vis_options: tuple[str | None, str | None] = (None, None),
    debug: int = 0,
    random_seed: int | None = None,
    save_to_file: bool = True,
) -> tuple[
    None | dict[int, StandardBlock],
    None | dict[tuple[int, int], list[str]],
    dict[str, bool | int | float]
]:
    """Call Topologiq to perform algorithmic lattice surgery on circuit.

    Args:
        circuit_name: The random PyZX graph.
        reduce (optional): Whether to optimise/reduce the circuit before running it or not.
        vis_options (optional): Visualisation settings provided as a tuple.
        debug (optional): Debug mode (0: off, 1: graph manager, 2: pathfinder, 3: pathfinder w. discarded paths).
        random_seed (optional): A specific seed to use for a particular run.
        save_to_file (optional): True to save the results to a `.bgraph` file, else False.

    Return:
        lat_nodes: The cubes of the final space-time diagram produced by Topologiq.
        lat_edges: The pipes of the final space-time diagram produced by Topologiq.
        test_stats: Misc. statistics for test run.

    """

    # Timer, unique ID, and seed
    success = True
    t1 = datetime.now()
    if random_seed:
        random.seed(random_seed)

    # Path to file
    path_to_qasm_circuit = ASSETS_DIR / f"{circuit_name}.qasm"

    # Run circuit as given
    lat_nodes, lat_edges = run_topologiq_qasm_as_input(
        circuit_name,
        path_to_qasm_circuit,
        reduce=reduce,
        max_attempts=1,
        vis_options=vis_options,
        debug=debug,
    )
    duration = (datetime.now() - t1).total_seconds()
    success = success if (lat_nodes and lat_edges) else not success

    # Write data and results to files
    circuit_name = circuit_name if reduce else circuit_name + "_canonical"

    # Save results to file
    if save_to_file and lat_nodes and lat_edges:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path_to_bgraph_file = OUTPUT_DIR / f"{circuit_name}.bgraph"
        with open(path_to_bgraph_file, "w") as f:
            f.write("BLOCKGRAPH 0.1.0;\n")
            f.write("\nCUBES: key;(x, y, z);kind;\n")
            f.writelines(
                [f"{key};{cube_info[0]};{cube_info[1]};\n" for key, cube_info in lat_nodes.items()]
            )

            f.write("\nPIPES: (src, tgt),kind;\n")
            f.writelines([f"{key};{pipe_info[0]};\n" for key, pipe_info in lat_edges.items()])

    test_stats = {
        "success": True if success else False,
        "volume": len(lat_nodes) if lat_nodes else 0,
        "duration": duration,
    }

    return lat_nodes, lat_edges, test_stats


###########
# RUNNERS #
###########
def run_topologiq_qasm_as_input(
    circuit_name: str,
    path_to_qasm_circuit: Path,
    reduce: bool = False,
    max_attempts: int = 10,
    vis_options: tuple[str | None, str | None] = (None, None),
    log_stats: bool = False,
    debug: int = 0,
) -> tuple[
    None | dict[int, StandardBlock],
    None | dict[tuple[int, int], list[str]],
]:
    """Load a circuit from a QASM file and run Topologiq with it.

    Args:
        circuit_name: The random PyZX graph.
        path_to_qasm_circuit: The path to the qasm file containing the circuit.
        reduce (optional): Whether to optimise/reduce the circuit before running it or not.
        max_attempts (optional): How many times to repeat-run the circuit.
        vis_options (optional): Visualisation settings provided as a tuple.
        stop_on_first_success: Whether to stop after the first successful attempt.
        log_stats (optional): If True, triggers automated stats logging to CSV files in `./benchmark/data`.
        debug (optional): Debug mode (0: off, 1: graph manager, 2: pathfinder, 3: pathfinder w. discarded paths).

    Return:
        lat_nodes: The cubes of the final space-time diagram produced by Topologiq.
        lat_edges: The pipes of the final space-time diagram produced by Topologiq.

    """

    # Convert to PyZX graph
    pyzx_circuit = zx.Circuit.load(path_to_qasm_circuit)
    pyzx_graph = pyzx_circuit.to_graph()

    # Draw un-reduced PyZX graph if any visualisation mode is on
    if vis_options[0] or debug > 2:
        zx.draw(pyzx_graph, labels=True)

    # Reduce if needed
    if reduce:
        circuits_with_reduction_strategy = ["qasm", "ghz"]
        if any([circuit for circuit in circuits_with_reduction_strategy]):
            # Apply states (commented out to enable comparison)
            num_apply_state = pyzx_graph.num_inputs()
            pyzx_graph.apply_state("0" * num_apply_state)

            # Post-select
            if circuit_name == "qasm_steane":
                pyzx_graph.apply_effect("000///////")
            elif "ghz" in circuit_name:
                qubit_n = int(circuit_name.split("_")[2])
                pyzx_graph.apply_effect("/" * qubit_n)

            # Reduce
            zx.full_reduce(pyzx_graph)
            if circuit_name == "qasm_steane":
                zx.to_rg(pyzx_graph)

            # Draw reduced version  if any visualisation mode is on
            if vis_options[0] or debug > 2:
                zx.draw(pyzx_graph, labels=True)
        else:
            print("Reduction strategy for this circuit not yet defined.")

    # Call Topologiq
    simple_graph = pyzx_g_to_simple_g(pyzx_graph)
    kwargs = {"weights": (-1, -1), "length_of_beams": 99}
    _, _, lat_nodes, lat_edges = runner(
        simple_graph,
        circuit_name,
        max_attempts=max_attempts,
        vis_options=vis_options,
        log_stats=log_stats,
        debug=debug,
        fig_data=None,
        **kwargs,
    )

    return lat_nodes, lat_edges
