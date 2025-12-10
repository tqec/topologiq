"""Run test using a series of originally-random circuits encoded as QASM.

This script tests Topologiq performance using a number of circuits generated
randomly and saved as QASM. After each run, outputs are saved to a `.bgraph`
file in `./outputs/bgraph/`.

Usage:
    Run script as given.

"""

from topologiq.utils.classes import Colors
from topologiq.utils.e2e import test_qasm_circuit

# ...
if __name__ == "__main__":

    # Adjustable parameters
    generic_circuit_name = "random"
    random_seed = 0
    vis_options = (None, None)  # Change to none for GitHub mode. Enable for debugging locally.
    debug = 0
    save_to_file = True

    # Update user
    print(
        Colors.BLUE,
        f"\n===> E2E QASM->Blockgraph Suite for {generic_circuit_name}. START.",
        Colors.RESET,
    )
    # Run circuits on a loop, without reduction
    circuit_names = ["qasm_random_05_05", "qasm_random_10_10"]  # Moderate runtimes.
    # circuit_names = ["qasm_random_05_05", "qasm_random_10_10", "qasm_random_10_20"]  # Significant runtimes.
    # circuit_names = ["qasm_random_05_05", "qasm_random_10_10", "qasm_random_03_30", "qasm_random_10_50"]  # VERY significant runtimes.
    reduce_mode = False
    for circuit_name in circuit_names:
        _, _, test_stats = test_qasm_circuit(
            circuit_name,
            reduce=reduce_mode,
            vis_options=vis_options,
            debug=debug,
            random_seed=random_seed,
            save_to_file=save_to_file,
        )

    # Update user with results
    success = test_stats["success"]
    duration = test_stats["duration"]
    print(
        Colors.BLUE,
        f"\n===> E2E QASM->Blockgraph for {generic_circuit_name}. END.",
        f"{Colors.GREEN + 'SUCCESS' if success else Colors.RED +'FAIL'}{Colors.RESET}. Duration: {duration:.2f}.\n",
    )
