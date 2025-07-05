import os
import time
from pathlib import Path
from typing import List

from scripts.greedy_bfs_traditional import main

from utils.classes import Colors, SimpleDictGraph
from utils.utils import build_newly_indexed_path_dict
from utils.utils_zx_graphs import strip_boundaries_from_zx_graph
from utils.grapher import visualise_3d_graph
from utils.animation import create_animation


####################
# MAIN RUN MANAGER #
####################
def runner(
    circuit_graph_dict: SimpleDictGraph,
    circuit_name: str,
    strip_boundaries: bool = False,
    hide_boundaries: bool = False,
    **kwargs,
):

    # START TIMER
    t1 = time.time()

    # APPLICABLE GRAPH TRANSFORMATIONS
    if strip_boundaries:
        circuit_graph_dict = strip_boundaries_from_zx_graph(circuit_graph_dict)

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
            circuit_graph_dict,
            circuit_name=circuit_name,
            hide_boundaries=hide_boundaries,
            **kwargs,
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

            lines: List[str] = []

            lines.append(f"RESULT SHEET. CIRCUIT NAME: {circuit_name}\n")
            lines.append("\n__________________________\nORIGINAL ZX GRAPH\n")
            for node in circuit_graph_dict["nodes"]:
                lines.append(f"Node ID: {node[0]}. Type: {node[1]}\n")
            lines.append("\n")
            for edge in circuit_graph_dict["edges"]:
                lines.append(f"Edge ID: {edge[0]}. Type: {edge[1]}\n")

            lines.append(
                '\n__________________________\n3D "EDGE PATHS" (Blocks needed to connect two original nodes)\n'
            )
            for key, edge_path in edge_paths.items():
                lines.append(
                    f"Edge {edge_path['src_tgt_ids']}: {edge_path['path_nodes']}\n"
                )

            lines.append(
                "\n__________________________\nLATTICE SURGERY (Given as graph)\n"
            )
            lattice_nodes, lattice_edges = build_newly_indexed_path_dict(edge_paths)
            for key, node in lattice_nodes.items():
                lines.append(f"Node ID: {key}. Type: {node}\n")
            for edge in lattice_edges:
                lines.append(f"Edge: {edge}\n")

            output_folder_path = "./outputs/txt"
            Path(output_folder_path).mkdir(parents=True, exist_ok=True)
            with open(f"{output_folder_path}/{circuit_name}.txt", "w") as f:
                f.writelines(lines)
                f.close()

            print(f"Result saved to: {output_folder_path}/{circuit_name}.txt")

            # Visualise result
            visualise_3d_graph(new_nx_graph, hide_boundaries=hide_boundaries)
            visualise_3d_graph(
                new_nx_graph,
                hide_boundaries=hide_boundaries,
                save_to_file=True,
                filename=f"steane{c}",
            )

            # Create GIF or result
            create_animation(
                filename_prefix=circuit_name,
                restart_delay=5000,
                duration=2500,
                video=False,
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
            try:
                for filename in os.listdir(temp_images_folder_path):
                    os.remove(f"./{temp_images_folder_path}/{filename}")
                os.rmdir(f"./{temp_images_folder_path}/")
            except (ValueError, FileNotFoundError) as e:
                print(
                    "Unable to delete temporary files or temp folder does not exist", e
                )

        # Reset errors flag for next loop (if there is one)
        errors_in_result = False
