# This file facilitates direct testing of specific cases.
#
# It picks up info for any (failed) edge cases from `./assets/stats/outputs.csv`
# and asks users which case they want to run. 
# 
# The chosen circuit is called with override values for first_id and first_kind, 
# which suffices to replicate the entire case because topologiq is deterministic
# given random choice of initial cube and its kind. 
#
# NB! The file requires the chosen circuit to exist in either
# `assets/graphs/simple_graphs.py` or `assets/graphs/pyzx_graphs.py`


from pathlib import Path

from topologiq.assets.graphs import simple_graphs, pyzx_graphs
from topologiq.run_hyperparams import VALUE_FUNCTION_HYPERPARAMS, LENGTH_OF_BEAMS
from topologiq.scripts.runner import runner
from topologiq.utils.interop_pyzx import pyzx_g_to_simple_g
from topologiq.utils.classes import SimpleDictGraph
from topologiq.utils.simple_grapher import simple_graph_vis
from topologiq.utils.utils_misc import get_edge_cases


####################
# MAIN RUN MANAGER #
####################
def run_debug():
    """Pick up list of any failed edge cases from output stats and run selected case.

    Returns:
        - n/a. All outputs below result from separate functions at different parts of the process.
            - Pops up an interactive 3D viewer for user to examine the space-time diagram produced as output.
            - Saves a TXT file with detailed results to `.outputs/txt/` folder.
            - Saves a GIF or MP4 animation to `.outputs/media/` folder.
            - IMPORTANT! This is NOT the function needed for programmatic use – see `./scripts/runner.py` for that.

    """

    # ASSEMBLE KWARGS
    kwargs = {
        "weights": VALUE_FUNCTION_HYPERPARAMS,
        "length_of_beams": LENGTH_OF_BEAMS,
    }

    # HELPER VARIABLES
    circuit_name: str = None
    circuit_as_graph_dict: SimpleDictGraph = {"nodes": [], "edges": []}

    # GET LIST OF EDGE CASES CURRENTLY IN OUTPUT STATS LOG
    # Path to file 
    path_to_output_logs = Path(__file__).parent.resolve() / "assets/stats/outputs.csv"

    # Raise if file does not exist at specified location
    if not path_to_output_logs.exists():
        raise FileNotFoundError("File `assets/stats/outputs.csv` must exist.\n")
    
    # Read all edge cases in outputs log
    edge_cases = list(set(get_edge_cases(path_to_output_logs)))

    # Announce no edge cases found if no edge cases found
    if not edge_cases:
        print("\nNO EDGE CASES FOUND IN `assets/stats/outputs.csv`.\n")
    
    # Print list of edge cases and ask users to select case to run
    else:    
        print("\n==> EDGE CASES AVAILABLE FOR DIRECT RUN")
        print("[case number] name_of_circuit, first_id, first_kind.")
        for i, case in enumerate(edge_cases):
            print(f"[{i}] {case[0]}, {case[1]}, {case[2]}.")
        print(f"[{i+1}] Exit debug mode.")
        # Loop until user chooses to exit or selects a valid case
        while True:
            case_number = input("\nTo run a specific case, enter a [case number] here: ")

            try:
                case_number = int(case_number)

                if case_number > i:
                    print("Exiting debug mode.\n")
                    break
                
                circuit_name, first_id, first_kind = edge_cases[case_number]
                break

            except (ValueError, KeyError, IndexError):
                print("You must choose a valid [case number] or type 'EXIT' to exit")        


        if circuit_name is not None:

            print(f"\nLAUNCHING CASE\nName: {circuit_name} (first_id:{first_id}, first_kind:{first_kind}).\n")
            # FIND AND RETRIEVE THE APPROPRIATE CIRCUIT
            # Look for name match in circuits saved as simple graphs
            if circuit_name in dir(simple_graphs):
                print("simple graph")
                circuit_as_graph_dict = getattr(simple_graphs, circuit_name)
                fig_data = simple_graph_vis(circuit_as_graph_dict)

            # Look for name match in pyzx circuits/graphs
            if circuit_name in dir(pyzx_graphs):
                print("pyzx graph")
                pyzx_function = getattr(pyzx_graphs, circuit_name)
                g, fig_data = pyzx_function(draw_graph=True)
                circuit_as_graph_dict = pyzx_g_to_simple_g(g)

            # RUN TOPOLOGIQ
            if circuit_as_graph_dict["nodes"] and circuit_as_graph_dict["edges"]:

                _, _, _, _ = runner(
                    circuit_as_graph_dict,
                    circuit_name,
                    min_succ_rate=60,
                    strip_ports=False,
                    hide_ports=False,
                    max_attempts=1,
                    stop_on_first_success=True,
                    visualise=("detail", "GIF"),
                    log_stats=True,
                    debug=True,
                    fig_data=fig_data,
                    first_cube=(first_id, first_kind),
                    **kwargs
                )


if __name__ == "__main__":
    run_debug()
