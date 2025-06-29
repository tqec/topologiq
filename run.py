import os
import sys
import time

from scripts.two_stage_greedy_bfs import main
from utils import examples
from grapher.grapher import visualise_3d_graph
from grapher.animation import create_animation
from utils.classes import Colors

from run_hyper_params import (
    VALUE_FUNCTION_HYPERPARAMS,
    LENGTH_OF_BEAMS,
    MAX_PATHFINDER_SEARCH_SPACE,
)


####################
# MAIN RUN MANAGER #
####################
def run():

    # START TIMER
    t1 = time.time()

    # ASSEMBLE KWARGS
    kwargs = {
        "weights": VALUE_FUNCTION_HYPERPARAMS,
        "length_of_beams": LENGTH_OF_BEAMS,
        "max_search_space": MAX_PATHFINDER_SEARCH_SPACE,
    }

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

            print(
                "\n\n####################################################",
                "\nSTARTING ALGORITHM FROM CLEAN SLATE",
                f"\nAttempt {i + 1}",
                "\n####################################################",
            )

            # Update counters
            i += 1
            t1_inner = time.time()

            # Call algorithm
            _, edge_paths, new_nx_graph, c = main(
                circuit_graph_dict, circuit_name=circuit_name, **kwargs
            )

            for key, edge_path in edge_paths.items():
                if edge_path["edge_type"] == "error":
                    errors_in_result = True

            if not errors_in_result:

                duration_total = (time.time() - t1) / 60
                print(
                    Colors.GREEN,
                    f"\n\nALGORITHM SUCCEEDED! Total run time: {duration_total} min",
                    Colors.RESET,
                )

                print("Results:")
                for key, edge_path in edge_paths.items():
                    print(f"  {key}: {edge_path['path_nodes']}")

                # VISUALISE FINAL LATTICE SURGERY
                visualise_3d_graph(new_nx_graph)
                visualise_3d_graph(
                    new_nx_graph, save_to_file=True, filename=f"steane{c}"
                )

                # CREATE A GIF FROM THE VISUALISATIONS
                create_animation(
                    filename_prefix=circuit_name, restart_delay=5000, duration=2500
                )

                break
            else:

                # UPDATE USER
                print(
                    Colors.RED,
                    "\nUNSUCCESFUL RUN. Will run again (unless run limits have been exceeded).",
                    Colors.RESET,
                    f"\n- This iteration took: {((time.time() - t1_inner) / 60):.2f} min",
                    f"\n- Total run time thus far: {((time.time() - t1) / 60):.2f} min",
                )

                # DELETE VISUALISATION TEMP FILES
                temp_images_folder_path = "assets/temp"
                for filename in os.listdir(temp_images_folder_path):
                    os.remove(f"./{temp_images_folder_path}/{filename}")
                os.rmdir(f"./{temp_images_folder_path}/")

            errors_in_result = False


if __name__ == "__main__":
    run()
