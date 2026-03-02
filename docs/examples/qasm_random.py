"""Run test using a small collection of QASM circuits.

This script tests Topologiq performance using a number of circuits generated
randomly in PyZX and saved as QASM. A novelty of this file is that outputs are
saved to a `.bgraph` file in `./outputs/bgraph/`. Eventually, the hope is to develop
a standard that allows easy interoperability between lattice surgery tools.

Usage:
    Run script as given.

"""

import os
from pathlib import Path

import pyzx as zx
from pyzx.graph.base import BaseGraph
from pyzx.graph.graph_s import GraphS

from topologiq.core.graph_manager.graph_manager import runner
from topologiq.input.pyzx import pyzx_g_to_simple_g
from topologiq.kwargs import VALUE_FUNCTION_HYPERPARAMS
from topologiq.utils.classes import Colors, StandardBlock
from topologiq.utils.core import datetime_manager
from topologiq.utils.read_write import write_bgraph

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
ASSETS_DIR = ROOT_DIR / "src/topologiq/assets"
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

    # Create output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    # Write to bgraph file
    path_to_output_file = output_dir / f"{circuit_name}.bgraph"
    write_bgraph(path_to_output_file, lat_nodes, lat_edges)


# ...
if __name__ == "__main__":
    # Update user
    print(Colors.BLUE, "\n===> START. QASM Test Panel.", Colors.RESET)

    # Circuits
    circuit_names = ["qasm_random_05_05", "qasm_random_10_10", "qasm_random_10_20"]

    # Determine if circuit should be reduced/optimised or not
    reduce_input_circuit = False

    # Adjust KWARGs
    # KWARGs no included here is autocompleted on run.
    kwargs = {
        "weights": VALUE_FUNCTION_HYPERPARAMS,
        "first_id_strategy": "centrality_random",
        "seed": None,
        "vis_options": (None, None),
        "max_attempts": 1,  # Run 10 tests for each circuit
        "stop_on_first_success": False,  # Do NOT stop after success (if True, this setting overrides max_attempts)
        "debug": 0,
        "log_stats": False,
    }

    # Run selected circuits on a loop, without reduction
    joint_success = True
    circuit_count = 0
    for circuit_name in circuit_names:
        _, _, test_stats = manage_single_qasm_test(
            circuit_name,
            reduce_input_circuit,
            **kwargs,
        )
        circuit_count += 1
        if not test_stats["success"]:
            joint_success = False

    # Update user with results
    print(
        Colors.BLUE,
        "\n===> END. QASM Test Panel.",
        f"{Colors.GREEN + 'SUCCESS' if joint_success else Colors.RED + 'FAIL'}",
        Colors.RESET,
        f"Duration: {test_stats['duration']:.2f}.\n",
    )
