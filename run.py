import sys
import time

from scripts.two_stage_greedy_bfs import main
from utils import examples
from grapher.grapher import visualise_3d_graph
from grapher.animation import create_animation
from utils.classes import Colours


####################
# MAIN RUN MANAGER #
####################
def run():

    # START GENERAL TIMER
    t1 = time.time()

    # GET CIRCUIT FROM PYTHON COMMAND
    circuit_name: str = ""
    for arg in sys.argv:
        if arg.startswith("--"):
            circuit_name = arg.replace("--", "")

    # GET CIRCUIT FROM PyZX
    if circuit_name == "":
        pass

    # CALL ALGORITHM ON CIRCUIT
    if circuit_name != "":

        # Aux variables
        i: int = 0
        errors_in_result: bool = False

        # Get circuit as a dictionary
        circuit_graph_dict = getattr(examples, circuit_name)

        # Loop until success or time limit
        while i < 10:

            # Communicate loop iteration
            i += 1
            print(
                Colours.GREEN,
                "\n\nCALLING ALGORITHM ON CIRCUIT OF NAME:",
                circuit_name,
                f". (Attempt # {i})",
                Colours.RESET
            )
            t1_inner = time.time()

            # Call algorithm
            _, edge_paths, new_nx_graph, c = main(
                circuit_graph_dict, circuit_name=circuit_name
            )

            for key, edge_path in edge_paths.items():
                if edge_path["edge_type"] == "error":
                    errors_in_result = True

            if not errors_in_result:

                duration_total = (time.time() - t1) / 60
                print(
                    Colours.GREEN,
                    f"SUCCESS! Total run time: {duration_total} min",
                    Colours.RESET,
                )
                
                print("\nResults:")
                for key, edge_path in edge_paths.items():
                    print(f"  {key}: {edge_path['path_nodes']}")
                    
                # VISUALISE FINAL LATTICE SURGERY
                visualise_3d_graph(new_nx_graph)
                visualise_3d_graph(new_nx_graph, save_to_file=True, filename=f"steane{c}")

                # CREATE A GIF FROM THE VISUALISATIONS
                create_animation(filename_prefix=circuit_name, restart_delay=5000, duration=2500)
                
                print("\n", Colours.BLUE, "\nTHAT'S IT. THANKS FOR FLYING WITH US!", Colours.RESET)

                break
            else:
                print(
                    Colours.RED,
                    "\nUNSUCCESFUL RUN. Algorithm will run again unless run limits have been exceeded.",
                    Colours.RESET,
                    f"\n- This run took: {((time.time() - t1_inner) / 60):.2f} min",
                    f"\n- Total run time thus far: {((time.time() - t1) / 60):.2f} min",
                )

            errors_in_result = False


if __name__ == "__main__":
    run()
