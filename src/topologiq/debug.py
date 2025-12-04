"""Run edge cases directly from command-line.

Script looks for cases in `./assets/stats/debug.csv` and prompts user to
select a specific case to run.

Usage:
    debug.py --graph:<name_of_graph> [options]

Options:
    -h, --help              Show this help message and exit.

Notes:
    The following file is required: `./assets/stats/debug.csv`. If this file
        does not exist, run Topologiq using `--stat_logs` option to enable stats logs.
        If a run fails or uses key atypical parameters, the run will be logged to this file.
    Chosen circuit must exist in `src/topologiq/assets/graphs/pyzx_graphs.py` or
        `src/topologiq/assets/graphs/simple_graphs.py`. If the circuit does not exist in
        either of these two files, it is better to call Topologiq directly using the optional
        `first_cube` parameter. Topologiq uses the information in this optional parameter
        to replicate the specific case.

"""

import sys
from pathlib import Path

from topologiq.assets import pyzx_graphs, simple_graphs
from topologiq.run_hyperparams import LENGTH_OF_BEAMS, VALUE_FUNCTION_HYPERPARAMS
from topologiq.scripts.runner import runner
from topologiq.utils.classes import SimpleDictGraph
from topologiq.utils.interop_pyzx import pyzx_g_to_simple_g
from topologiq.utils.simple_grapher import simple_graph_vis
from topologiq.utils.utils_misc import get_debug_cases


#################
# DEBUG MANAGER #
#################
def run_debug():
    """Pick up the list of edge cases and run selected case.

    This function serves as main entry point for command-line debugging. It relies
    on the existence of edge cases in `./assets/stats/debug.csv`. When it runs,
    it gathers all unique edge cases in the debug file, and prompts user to choose
    one of these cases to run.

    Outputs:
        The purpose of this function is to produce visualisations and file-based outputs.
        The function will:
            * Save a TXT file with detailed information about the run.
            * Pop up an interactive 3D viewer after each edge to examine progress step by step.
            * Save a GIF with an animated summary of the full run (not including any non-completed edge).

    Returns:
        None: This function does not produce objects for programmatic use.

    """

    # Misc options
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__)
        sys.exit(0)

    # Preliminaries
    circuit_as_graph_dict: SimpleDictGraph = {"nodes": [], "edges": []}
    circuit_name: str = None
    min_success_rate = 60
    first_id, first_kind = (None, None)
    value_fn_weights = (VALUE_FUNCTION_HYPERPARAMS,)
    len_of_beams = LENGTH_OF_BEAMS

    # Get list of edge cases
    path_to_stats = Path(__file__).parent.resolve() / "assets/stats/debug.csv"

    if not path_to_stats.exists():
        raise FileNotFoundError(f"File `{path_to_stats}` must exist.\n")

    debug_cases = list(set(get_debug_cases(path_to_stats)))

    if not debug_cases:
        print(f"\nNO EDGE CASES FOUND IN `{path_to_stats}`.\n")

    # Ask user to select case to run
    else:
        print("\n==> EDGE CASES AVAILABLE FOR DIRECT RUN")
        print(
            "[case number] circuit_name, first_id, first_kind, min_success_rate, value_fn_weights, len_of_beams."
        )
        for i, case in enumerate(debug_cases):
            print(f"[{i}] {str(case)[1:-1]}.")
        print(f"[{i + 1}] Exit debug mode.")

        # Loop until user makes valid choice or exits
        while True:
            case_number = input("\nTo run a specific case, enter a [case number] here: ")

            try:
                case_number = int(case_number)
                if case_number > i:
                    print("Exiting debug mode.\n")
                    break
                (
                    circuit_name,
                    first_id,
                    first_kind,
                    min_success_rate,
                    value_fn_weights,
                    len_of_beams,
                ) = debug_cases[case_number]
                break
            except (ValueError, KeyError, IndexError):
                print("You must choose a valid [case number] or type 'EXIT' to exit")

        if circuit_name is not None:
            # Update user
            print("\nLAUNCHING CASE")
            print(
                "[case number] circuit_name, first_id, first_kind, min_success_rate, value_fn_weights, len_of_beams."
            )
            print(
                f"[{case_number}]",
                circuit_name,
                first_id,
                first_kind,
                min_success_rate,
                value_fn_weights,
                len_of_beams,
            )

            # Assemble KWARGS
            kwargs = {
                "weights": value_fn_weights,
                "length_of_beams": len_of_beams,
            }

            # Retrieve circuit
            # NB! Function will NOT work if circuit is not saved to a circuit files

            # Look for name match in simple graphs
            if circuit_name in dir(simple_graphs):
                circuit_as_graph_dict = getattr(simple_graphs, circuit_name)
                fig_data = simple_graph_vis(circuit_as_graph_dict)

            # Look for name match in pyzx graphs
            if circuit_name in dir(pyzx_graphs):
                pyzx_function = getattr(pyzx_graphs, circuit_name)
                g, fig_data = pyzx_function(draw_graph=True)
                circuit_as_graph_dict = pyzx_g_to_simple_g(g)

            # Run Topologiq
            if circuit_as_graph_dict["nodes"] and circuit_as_graph_dict["edges"]:
                _, _, _, _ = runner(
                    circuit_as_graph_dict,
                    circuit_name,
                    min_succ_rate=min_success_rate,
                    strip_ports=False,
                    hide_ports=False,
                    max_attempts=1,
                    stop_on_first_success=True,
                    vis_options=("final", "GIF"),
                    log_stats=False,
                    debug=3,
                    fig_data=fig_data,
                    first_cube=(first_id, first_kind),
                    **kwargs,
                )


if __name__ == "__main__":
    run_debug()
