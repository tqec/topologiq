"""Run test using a couple GHZ circuits encoded as QASM.

This script tests Topologiq performance using a couple canonical GHZ circuits
saved as QASM file. After each run, outputs are saved to a `.bgraph` file in
`./outputs/bgraph/`.

Usage:
    Run script as given.

"""

import os
from pathlib import Path

import pyzx as zx
from pyzx.graph.base import BaseGraph
from qiskit import qasm2
from qiskit.circuit import QuantumCircuit

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
def manage_single_ghz_test(
    circuit_name: str,
    n_qubits: int,
    reduce_input_circuit: bool = False,
    vis_options: tuple[str | None, str | None] = (None, None),
    max_attempts: int = 10,
    debug: int = 0,
    log_stats: bool = False,
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
        n_qubits: The number of qubits in the GHZ circuit.
        vis_options (optional): Visualisation settings provided as a tuple.
        max_attempts (optional): How many times to repeat-run the circuit.
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
    t_1,_ = datetime_manager()

    # Retrieve QASM as PyZX graph
    qasm_str = ghz_to_qasm(n_qubits, circuit_name)
    pyzx_graph = qasm_to_pyzx(qasm_str)
    if reduce_input_circuit:
        pyzx_graph = pyzx_reduce(pyzx_graph)

    # Call topologiq on circuit
    simple_graph = pyzx_g_to_simple_g(pyzx_graph)
    _, _, lat_nodes, lat_edges = runner(
        simple_graph,
        circuit_name,
        max_attempts=max_attempts,
        vis_options=vis_options,
        log_stats=log_stats,
        debug=debug,
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

# ...
def ghz_to_qasm(n_qubits: int, circuit_name: str) -> str:
    """Create a n-qubits GHZ and convert it to QASM.

    Args:
        n_qubits: The number of qubits for the GHZ.
        circuit_name: The name of the circuit.

    """
    # Foundational circuit
    qc: QuantumCircuit = QuantumCircuit(n_qubits, name=circuit_name)

    # GHZ encoding
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    # Convert to QASM
    qasm_str = qasm2.dumps(qc)

    return qasm_str


def qasm_to_pyzx(qasm_str: str) -> BaseGraph:
    """Import a circuit from QASM and convert it to a PyZX graph.

    Args:
        qasm_str: A quantum circuit encoded as a QASM string.

    """
    # QASM --> PyZX circuit --> PyZX graph
    zx_circuit = zx.Circuit.from_qasm(qasm_str)
    zx_graph = zx_circuit.to_graph()

    return zx_graph


def pyzx_reduce(zx_graph: BaseGraph) -> BaseGraph:
    """Reduce a PyZX graph after applying states to all inputs.

    Args:
        zx_graph: The input ZX graph, given as a PyZX graph.

    """

    # Work with copy
    zx_graph_copy = zx_graph

    # Apply states
    num_apply_state = zx_graph_copy.num_inputs()
    zx_graph_copy.apply_state("0" * num_apply_state)

    # Reduce & draw
    zx.full_reduce(zx_graph_copy)

    return zx_graph_copy

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
    n_qubits = [16, 200]

    # Adjustable parameters
    reduce_input_circuit = True
    vis_options = ("final", None)
    max_attempts = 10
    debug = 0
    log_stats = False

    # KWARGS
    kwargs = {
        "weights": VALUE_FUNCTION_HYPERPARAMS,
        "deterministic": False,
        "seed": None,
    }

    for n in n_qubits:
        circuit_name = f"ghz_{n}"
        _, _, test_stats = manage_single_ghz_test(
            circuit_name,
            n,
            reduce_input_circuit,
            vis_options=vis_options,
            max_attempts=max_attempts,
            debug=debug,
            log_stats=log_stats,
            **kwargs,
        )

    # Update user with results
    print(
        Colors.BLUE,
        "\n===> E2E Qiskit GHZ Test Suite. END.",
        f"{Colors.GREEN + 'SUCCESS' if test_stats['success'] else Colors.RED + 'FAIL'}",
        Colors.RESET,
        f"Duration: {test_stats['duration']:.2f}.\n",
    )
