import time
import shutil
from pathlib import Path
from typing import List, Tuple, Union

from scripts.greedy_bfs_traditional import main

from utils.classes import Colors, SimpleDictGraph, StandardBlock
from utils.utils_greedy_bfs import build_newly_indexed_path_dict
from utils.utils_zx_graphs import strip_boundaries_from_zx_graph
from utils.grapher import visualise_3d_graph, make_graph_from_final_lattice
from utils.animation import create_animation


####################
# MAIN RUN MANAGER #
####################
def runner(
    circuit_graph_dict: SimpleDictGraph,
    circuit_name: str,
    strip_boundaries: bool = False,
    hide_boundaries: bool = False,
    max_attempts: int = 10,
    visualise: Tuple[Union[None, str], Union[None, str]] = (None, None),
    **kwargs,
) -> Tuple[
    SimpleDictGraph,
    Union[None, dict],
    Union[None, dict[int, StandardBlock]],
    Union[None, dict[Tuple[int, int], List[str]]],
]:
    """Runs the algorithm on any circuit given to it

    Args:
        - circuit_graph_dict: a ZX circuit as a simple dictionary of nodes and edges.
        - circuit_name: name of ZX circuit.
        - strip_boundaries:
            - true: instructs the algorithm to eliminate any boundary nodes and their corresponding edges,
            - false: nodes are factored into the process and shown on visualisation.
        - hide_boundaries:
            - true: instructs the algorithm to use boundary nodes but do not display them in visualisation,
            - false: boundary nodes are factored into the process and shown on visualisation.
        - visualise: a tuple with visualisation settings:
            - visualise[0]:
                - None: no visualisation whatsoever,
                - "final" (str): triggers a single on-screen visualisation of the final result (small performance trade-off),
                - "detail" (str): triggers on-screen visualisation for each edge in the original ZX-graph (medium performance trade-off).
            - visualise[1]:
                - None: no animation whatsoever,
                - "GIF": saves step-by-step visualisation of the process in GIF format (huge performance trade-off),
                - "MP4": saves a PNG of each step/edge in the visualisation process and joins them into a GIF at the end (huge performance trade-off).

    Keyword arguments (**kwargs):
        - weights: weights for the value function to pick best of many paths.
        - length_of_beams: length of each of the beams coming out of open nodes.

    Returns:
        - circuit_graph_dict: original circuit given to function returns for easy traceability
        - edge_paths: the raw set of 3D edges found by the algorithm (with redundant blocks for start and end positions of some edges)
        - lattice_nodes: the nodes/blocks of the resulting space-time diagram (without redundant blocks)
        - lattice_edges: the edges/pipes of the resulting space-time diagram (without redundant pipes)

    """

    # PRELIMINARIES
    t1 = time.time()
    repository_root: Path = Path(__file__).resolve().parent.parent
    output_folder_path = repository_root / "outputs/txt"
    temp_folder_path = repository_root / "outputs/temp"
    Path(output_folder_path).mkdir(parents=True, exist_ok=True)

    # APPLICABLE GRAPH TRANSFORMATIONS
    if strip_boundaries:
        circuit_graph_dict = strip_boundaries_from_zx_graph(circuit_graph_dict)

    # VARS TO HOLD RESULTS
    edge_paths: Union[None, dict] = None
    lattice_nodes: Union[None, dict[int, StandardBlock]] = None
    lattice_edges: Union[None, dict[Tuple[int, int], List[str]]] = None

    # LOOP UNTIL SUCCESS OR LIMIT
    errors_in_result: bool = False
    i: int = 0
    while i < max_attempts:

        print(
            "\n\n################",
            f"\nAttempt {i + 1}",
            "\n##################",
        )

        # Update counters
        i += 1
        t1_inner = time.time()

        # Call algorithm
        _, edge_paths, new_nx_graph, c = main(
            circuit_graph_dict,
            circuit_name=circuit_name,
            hide_boundaries=hide_boundaries,
            visualise=visualise,
            **kwargs,
        )

        for key, edge_path in edge_paths.items():
            if edge_path["edge_type"] == "error":
                errors_in_result = True

        # Return result is there are no errors
        if not errors_in_result:

            # Last computations
            lattice_nodes, lattice_edges = build_newly_indexed_path_dict(edge_paths)

            # User updates
            duration_this_run = time.time() - t1_inner
            duration_total = time.time() - t1
            
            print(
                Colors.GREEN,
                "\nSUCCESS!!!",
                Colors.RESET,
                f"\n- This iteration took: {duration_this_run:.2f} secs",
                f"\n- Total run time: {duration_total:.2f} secs.",
            )
            
            if visualise[0] is not None or visualise[1] is not None:
                print("You have visualisations enabled. For faster runtimes, please disable visualisations.")

            # Write outputs
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

            lines.append("\n__________________________\nLATTICE SURGERY (Graph)\n")
            for key, node in lattice_nodes.items():
                lines.append(f"Node ID: {key}. Info: {node}\n")
            for key, edge_info in lattice_edges.items():
                lines.append(
                    f"Edge ID: {key}. Kind: {edge_info[0]}. Original edge in ZX graph: {edge_info[1]} \n"
                )

            with open(f"{output_folder_path}/{circuit_name}.txt", "w") as f:
                f.writelines(lines)
                f.close()

            print(f"\nResults saved to outputs folder.")

            # Visualise result
            if visualise[0] == "final" or visualise[0] == "detail":
                final_nx_graph, _ = make_graph_from_final_lattice(
                    lattice_nodes, lattice_edges
                )
                visualise_3d_graph(final_nx_graph, hide_boundaries=hide_boundaries)

            # Animate
            if visualise[1] == "GIF" or visualise[1] == "MP4":
                visualise_3d_graph(
                    new_nx_graph,
                    hide_boundaries=hide_boundaries,
                    save_to_file=True,
                    filename=f"{circuit_name}{c:03d}",
                )

                create_animation(
                    filename_prefix=circuit_name,
                    restart_delay=5000,
                    duration=2500,
                    video=True if visualise[1] == "MP4" else False,
                )

            # End loop
            break

        # If there are errors in result, continue loop
        else:

            # Update user
            duration_this_run = time.time() - t1_inner
            duration_total = time.time() - t1
            
            print(
                Colors.RED,
                "\nITERATION FAILED.",
                Colors.RESET,
                f"\n- This iteration took: {duration_this_run:.2f} secs",
                f"\n- Total run time: {duration_total:.2f} secs.",
            )
            
            if visualise[0] is not None or visualise[1] is not None:
                print("You have visualisations enabled. For faster runtimes, please disable visualisations.")

            # Delete temporary files
            try:
                if temp_folder_path.exists():
                    shutil.rmtree(temp_folder_path)
            except (ValueError, FileNotFoundError) as e:
                print("Unable to delete temp files or temp folder does not exist", e)

        # Reset errors flag for next loop (if there is one)
        errors_in_result = False

    # RETURN: original ZX graph, edge_paths, nodes and edges of result
    return circuit_graph_dict, edge_paths, lattice_nodes, lattice_edges
