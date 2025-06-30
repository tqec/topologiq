import os
import time

from scripts.greedy_bfs_traditional import main
from utils.grapher import visualise_3d_graph
from utils.animation import create_animation
from utils.classes import Colors, SimpleDictGraph

####################
# MAIN RUN MANAGER #
####################
def runner(circuit_graph_dict:SimpleDictGraph, circuit_name:str, **kwargs):
    
    # START TIMER
    t1 = time.time()

    # LOOP UNTIL SUCCESS OR LIMIT
    i: int = 0
    errors_in_result: bool = False
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

        # Return result is there are no errors
        if not errors_in_result:

            duration_total = (time.time() - t1) / 60
            print(
                Colors.GREEN,
                f"\n\nALGORITHM SUCCEEDED! Total run time: {duration_total:.2f} min",
                Colors.RESET,
            )

            print("Results:")
            for key, edge_path in edge_paths.items():
                print(f"  {key}: {edge_path['path_nodes']}")

            # Visualise result
            visualise_3d_graph(new_nx_graph)
            visualise_3d_graph(
                new_nx_graph, save_to_file=True, filename=f"steane{c}"
            )

            # Create GIF or result
            create_animation(
                filename_prefix=circuit_name, restart_delay=5000, duration=2500
            )

            # End loop
            break
        
        # If there are errors in result, continue loop
        else:

            # Update user
            duration_this_run = (time.time() - t1_inner) / 60
            duration_thus_far = (time.time() - t1) / 60
            print(
                Colors.RED,
                "\nUNSUCCESFUL RUN. Will run again (unless run limits have been exceeded).",
                Colors.RESET,
                f"\n- This iteration took: {duration_this_run:.2f} min",
                f"\n- Total run time thus far: {duration_thus_far:.2f} min",
            )

            # Delete temporary files
            temp_images_folder_path = "./outputs/temp"
            for filename in os.listdir(temp_images_folder_path):
                os.remove(f"./{temp_images_folder_path}/{filename}")
            os.rmdir(f"./{temp_images_folder_path}/")

        # Reset errors flag for next loop (if there is one)
        errors_in_result = False

