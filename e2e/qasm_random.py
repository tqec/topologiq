"""Run test using a series of originally-random circuits encoded as QASM.

This script tests Topologiq performance using a number of circuits generated
randomly and saved as QASM. After each run, outputs are saved to a `.bgraph`
file in `./outputs/bgraph/`.

Usage:
    Run script as given.

"""

import os
from pathlib import Path

import pyzx as zx
from pyzx.graph.base import BaseGraph
from pyzx.graph.graph_s import GraphS

from topologiq.run_hyperparams import VALUE_FUNCTION_HYPERPARAMS
from topologiq.scripts.runner import runner
from topologiq.utils.classes import Colors, StandardBlock
from topologiq.utils.interop_pyzx import pyzx_g_to_simple_g
from topologiq.utils.utils_misc import datetime_manager

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
ASSETS_DIR = ROOT_DIR / "assets"
DATA_DIR = ROOT_DIR / "benchmark/data"
OUTPUT_DIR = ROOT_DIR / "output/bgraph"


#################
# FLOW MANAGER #
#################
def manage_single_qasm_test(
    circuit_name: str,
    reduce_input_circuit: bool = False,
    **kwargs,
) -> tuple[
    dict[int, StandardBlock] | None,
    dict[tuple[int, int], list[str]] | None,
    dict[str, bool | int | float],
]:
    """Call Topologiq to perform algorithmic lattice surgery on circuit.

    Args:
        circuit_name: The name of the circuit.
        reduce_input_circuit (optional): Whether to optimise/reduce the circuit before running it or not.
        vis_options (optional): Visualisation settings provided as a tuple.
        max_attempts (optional): How many times to repeat-run the circuit.
        stop_on_first_success (optional): If True, forces exit on first successful outcome irrespective of `max_attempts`.
        debug (optional): Debug mode (0: off, 1: graph manager, 2: pathfinder, 3: pathfinder w. discarded paths).
        log_stats (optional): If True, triggers automated stats logging to CSV files in `./benchmark/data`.
        random_seed (optional): A specific seed to use for a particular run.
        save_to_file (optional): True to save the results to a `.bgraph` file, else False.
        **kwargs: !

    Return:
        lat_nodes: The cubes of the final space-time diagram produced by Topologiq.
        lat_edges: The pipes of the final space-time diagram produced by Topologiq.
        test_stats: Misc. statistics for test run.

    """

    # Timer, unique ID, and seed
    success = True
    t_1, _ = datetime_manager()

    # Path to file
    path_to_qasm_circuit = ASSETS_DIR / f"{circuit_name}.qasm"

    # Retrieve QASM as PyZX graph
    pyzx_graph = retrieve_qasm(circuit_name, path_to_qasm_circuit, reduce_input_circuit)

    # Convert to simple graph
    simple_graph = pyzx_g_to_simple_g(pyzx_graph)

    _, _, lat_nodes, lat_edges = runner(
        simple_graph,
        circuit_name,
        **kwargs,
    )

    # Stop timer
    _, t_total = datetime_manager(t_1=t_1)
    success = success if (lat_nodes and lat_edges) else not success

    # Write data and results to files
    circuit_name = circuit_name if reduce_input_circuit else circuit_name + "_canonical"

    # Save results to file
    if lat_nodes and lat_edges:
        save_test_results_to_file(circuit_name, OUTPUT_DIR, lat_nodes, lat_edges)

    # Assemble test stats
    test_stats = {
        "success": True if success else False,
        "volume": len(lat_nodes) if lat_nodes else 0,
        "duration": t_total,
    }

    return lat_nodes, lat_edges, test_stats


#######
# AUX #
#######
def retrieve_qasm(
    circuit_name: str,
    path_to_qasm_circuit: Path,
    reduce_input_circuit: bool = False,
) -> BaseGraph | GraphS:
    """Retrieve a circuit from a QASM file.

    Args:
        circuit_name: The name of the circuit.
        path_to_qasm_circuit: The path to the qasm file containing the circuit.
        reduce_input_circuit (optional): Whether to optimise/reduce the circuit before running it or not.

    Return:
        str: The retrieved QASM string.

    """

    # Retrieve QASM string from QASM file and convert to ZX graph
    pyzx_circuit = zx.Circuit.load(path_to_qasm_circuit)
    pyzx_graph = pyzx_circuit.to_graph()

    # Draw un-reduced PyZX graph if any visualisation mode is on
    if kwargs["vis_options"][0] or kwargs["debug"] > 2:
        zx.draw(pyzx_graph, labels=True)

    # Reduce if reduce mode is on
    if reduce_input_circuit:
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

            # Draw reduced version if any visualisation mode is on
            if kwargs["vis_options"][0] or kwargs["debug"] > 2:
                zx.draw(pyzx_graph, labels=True)
        else:
            print("Reduction strategy for this circuit not yet defined.")

    return pyzx_graph


def save_test_results_to_file(
    circuit_name: str,
    output_dir: Path,
    lat_nodes: dict[int, StandardBlock],
    lat_edges: dict[tuple[int, int], list[str]],
):
    """Save test results to file.

    Args:
        circuit_name: The name of the circuit.
        output_dir: The path to the directory where results should be saved.
        lat_nodes: lat_nodes: The cubes of the final space-time diagram produced by Topologiq.
        lat_edges: lat_edges: The pipes of the final space-time diagram produced by Topologiq.

    """

    os.makedirs(output_dir, exist_ok=True)
    path_to_bgraph_file = output_dir / f"{circuit_name}.bgraph"
    with open(path_to_bgraph_file, "w") as f:
        f.write("BLOCKGRAPH 0.1.0;\n")
        f.write("\nCUBES: key;(x, y, z);kind;\n")
        f.writelines(
            [f"{key};{cube_info[0]};{cube_info[1]};\n" for key, cube_info in lat_nodes.items()]
        )

        f.write("\nPIPES: (src, tgt),kind;\n")
        f.writelines([f"{key};{pipe_info[0]};\n" for key, pipe_info in lat_edges.items()])


# ...
if __name__ == "__main__":
    # Update user
    print(Colors.BLUE, "\n===> E2E QASM Test Suite. START.", Colors.RESET)

    # Circuits
    circuit_names = [
        # "qasm_random_05_05",
        "qasm_random_10_10",
        # "qasm_random_10_20",
        # "qasm_random_03_30",
        # "qasm_random_10_50",  # Still takes too long to enable by default
    ]

    # Determine if circuit should be reduced/optimised or not
    reduce_input_circuit = False

    # Adjust KWARGS
    # Only include kwargs when you want to deviate from default. Others will be autocompleted on run.
    # (Visualisation mode, Animation mode)
    kwargs = {
        "weights": VALUE_FUNCTION_HYPERPARAMS,
        "deterministic": False,
        "seed": None,
        "vis_options": (None, None),
        "max_attempts": 10,
        "stop_on_first_success": True,
        "debug": 0,
        "log_stats": True,
    }

    # Run selected circuits on a loop, without reduction
    for circuit_name in circuit_names:
        _, _, test_stats = manage_single_qasm_test(
            circuit_name,
            reduce_input_circuit,
            **kwargs,
        )

    # Update user with results
    print(
        Colors.BLUE,
        "\n===> E2E QASM->Blockgraph Test Suite. END.",
        f"{Colors.GREEN + 'SUCCESS' if test_stats['success'] else Colors.RED + 'FAIL'}",
        Colors.RESET,
        f"Duration: {test_stats['duration']:.2f}.\n",
    )
