from datetime import datetime
import os
import csv
import numpy as np

from pathlib import Path
from typing import Any, List, Tuple, Union

from topologiq.utils.classes import SimpleDictGraph, StandardBlock, StandardCoord


#############
# CONSTANTS #
#############
HEADER_BFS_MANAGER_STATS = [
    "unique_run_id",
    "run_success",
    "circuit_name",
    "len_beams",
    "num_input_nodes_processed",
    "num_input_edges_processed",
    "num_1st_pass_edges_processed",
    "num_2n_pass_edges_processed",
    "num_edges_in_edge_paths",
    "num_blocks_output",
    "num_edges_output",
    "duration_first_pass",
    "duration_second_pass",
    "duration_total",
]

HEADER_PATHFINDER_STATS = [
    "unique_run_id",
    "iter_type",
    "iter_success",
    "src_coords",
    "src_kind",
    "tgt_coords",
    "tgt_zx_type",
    "tgt_kind",
    "num_tent_coords_received",
    "num_tent_coords_filled",
    "max_manhattan_src_to_any_tent_coord",
    "len_longest_path",
    "num_visitation_attempts",
    "num_sites_visited",
    "iter_duration",
]

HEADER_PARAMS_STATS = [
    "unique_run_id",
    "circuit_name",
    "run_success",
    "run_params",
    "edge_paths",
]


##############
# SHARED OPS #
##############
def get_manhattan(src_coords: StandardCoord, tgt_coords: StandardCoord) -> int:
    """Calculate the Manhattan distance between any two (x, y, z) coordinates.
    Args:
        src_coords: The (x, y, z) coordinates for the source block.
        tgt_coords: The (x, y, z) coordinates for the target block.
    Returns:
        int: The Manhattan distance between the given coordinates.
    """

    return np.sum(np.abs(np.array(src_coords) - np.array(tgt_coords)))


def get_max_manhattan(src_coord: StandardCoord, all_coords: List[StandardCoord]) -> int:
    """Calculate the maximum Manhattan distance between a coordinate and a list of coordinates.
    Args:
        src_coord: The (x, y, z) coordinates for the source block.
        all_coords: A list of (x, y, z) coordinates of any arbitrary length, which may include src_coord.
    Returns:
        int: The max Manhattan distance between the source coordinate and all coordinates in the list of coordinates.
    """

    if all_coords:
        return max([get_manhattan(src_coord, c) for c in all_coords])

    return 0


############
# LOGGERS  #
############
def write_outputs(
    simple_graph: SimpleDictGraph,
    circuit_name: str,
    edge_paths: dict,
    lat_nodes: dict[int, StandardBlock],
    lat_edges: dict[Tuple[int, int], List[str]],
    output_dir_path: Path,
):
    """Write the final output to a TXT file.
    
    Args:
        simple_graph: The `simple_graph` form of an arbitrary ZX circuit.
        circuit_name: The name of the circuit.
        edge_paths: An edge-by-edge/block-by-block summary of the space-time diagram Topologiq builds.
        lat_nodes: The cubes of the final space-time diagram produced by Topologiq.
        lat_edges: The pipes of the final space-time diagram produced by Topologiq. 
        output_dir_path: The directory where outputs are saved.
    """

    lines: List[str] = []

    lines.append(f"RESULT SHEET. CIRCUIT NAME: {circuit_name}\n")
    lines.append("\n__________________________\nORIGINAL ZX GRAPH\n")
    for node in simple_graph["nodes"]:
        lines.append(f"Node ID: {node[0]}. Type: {node[1]}\n")
    lines.append("\n")
    for edge in simple_graph["edges"]:
        lines.append(f"Edge ID: {edge[0]}. Type: {edge[1]}\n")

    lines.append(
        '\n__________________________\n3D "EDGE PATHS" (Blocks needed to connect two original nodes)\n'
    )

    for key, edge_path in edge_paths.items():
        lines.append(f"Edge {edge_path['src_tgt_ids']}: {edge_path['path_nodes']}\n")

    lines.append("\n__________________________\nLATTICE SURGERY (Graph)\n")
    for key, node in lat_nodes.items():
        lines.append(f"Node ID: {key}. Info: {node}\n")
    for key, edge_info in lat_edges.items():
        lines.append(
            f"Edge ID: {key}. Kind: {edge_info[0]}. Original edge in ZX graph: {edge_info[1]} \n"
        )

    with open(f"{output_dir_path}/{circuit_name}.txt", "w") as f:
        f.writelines(lines)
        f.close()


def prep_stats_n_log(
    stats_type: str,
    log_stats_id: str,
    op_success: bool,
    counts: dict[str, int],
    times: dict[str, Union[datetime, None]],
    circuit_name: str = "unknown",
    edge_paths: Union[None, dict] = None,
    lat_nodes: Union[None, dict[int, StandardBlock]] = None,
    lat_edges: Union[None, dict[Tuple[int, int], List[str]]] = None,
    src_block_info: Union[None, StandardBlock] = None,
    tgt_block_info: Tuple[Union[None, StandardCoord], Union[None, str]] = (None, None),
    tgt_zx_type: Union[None, str] = None,
    visit_stats: Tuple[int, int] = (0, 0),
    run_params: dict[str, Any] = {},
):
    """Prepare incoming parameters for logging stats to file. 
    
    This function takes a number of parameters and assembles them in the order needed to log stats to file. 
    The function uses the incoming `stats_type` to determine appropriate order, and adds headers as appropriate
    using the header constants available at the top of this file.

    NB! Please note the arguments give to this function can be very different depending on whether the
    operation relates to logging statistics of one edge iteration by the pathfinder or the summary
    of the full process. Keep this in mind when reading parameter descriptions, as some refer to 
    global objects used to keep track of the overall processes
    while other refer to specific objects used in one pathfinder iteration.

    Args:
        stats_type: The desired type of logging operation.
        log_stats_id: 
        op_success: Whether Topologiq succesfully built the circuit.
        counts: A dictionary containing counts for the number of spiders/cubes and edges/pipes in input and output circuits.
        times: A dictionary containing various running times for several aspects of the process.
        circuit_name: The name of the circuit.
        edge_paths: An edge-by-edge/block-by-block summary of the space-time diagram Topologiq builds.
        lat_nodes: The cubes of the final space-time diagram produced by Topologiq.
        lat_edges: The pipes of the final space-time diagram produced by Topologiq. 
        src_block_info: The information of a source cube including its position in the 3D space and its kind.
        tgt_block_info: The information of a target cube including its position in the 3D space and its kind.
        tgt_zx_type: The ZX type of a target spider/cube
        visit_stats: Statistics about the number of visitation attempts and visits in a given pathfinder iteration.
        run_params: A number of critical parameters needed to replicate how Topologiq approached a given circuit.

    Keyword arguments (kwargs):
        See "CONSTANTS" for all possibilities.

    Returns:
        main_stats: An array containing all principal stats to log to primary stats files.
        aux_stats: An array containing secondary stats to log to auxiliary stats file, or an empty array if no secondary stats exist.
    """

    # Init arrays
    main_stats = []
    aux_stats = []

    # Fill arrays as determined by the `stats_type`
    if "graph_manager" in stats_type:

        durations = {
            "first_pass": (
                (times["t2"] - times["t1"]).total_seconds()
                if times["t2"] and times["t1"]
                else "error"
            ),
            "second_pass": (
                (times["t_end"] - times["t2"]).total_seconds()
                if times["t2"] and times["t_end"]
                else "error"
            ),
            "total": (
                (times["t_end"] - times["t1"]).total_seconds()
                if times["t1"] and times["t_end"]
                else "error"
            ),
        }

        main_stats = [
            log_stats_id,
            op_success,
            circuit_name,
            run_params["length_of_beams"],
            counts["num_input_nodes_processed"] if op_success else 0,
            counts["num_input_edges_processed"] if op_success else 0,
            counts["num_1st_pass_edges_processed"] if op_success else 0,
            counts["num_2n_pass_edges_processed"] if op_success else 0,
            len(edge_paths) if edge_paths else 0,
            len(lat_nodes.keys()) if lat_nodes else 0,
            len(lat_edges.keys()) if lat_edges else 0,
            durations["first_pass"],
            durations["second_pass"],
            durations["total"],
        ]

        try:
            edge_paths_summary = [
                {
                    edge_path["src_tgt_ids"][0]: edge_path["path_nodes"][0][1] if edge_path["path_nodes"][0][1] else None,
                    edge_path["src_tgt_ids"][1]: edge_path["path_nodes"][-1][1] if edge_path["path_nodes"][-1][1] else None,
                }
                for edge_path in edge_paths.values()] if edge_paths else ["error"]

        except Exception as e:
            print(f"Minor error with logging of aux stats: {e}")
            edge_paths_summary = [
                {
                    edge_path["src_tgt_ids"][0]: "Undefined",
                    edge_path["src_tgt_ids"][1]: "Undefined",
                }
                for edge_path in edge_paths.values()] if edge_paths else ["error"]

        aux_stats = [
            log_stats_id,
            op_success,
            circuit_name,
            run_params,
            edge_paths_summary,
        ]

        log_stats(
            aux_stats,
            f"params{'_tests' if log_stats_id.endswith('*') else ''}",
            opt_header=HEADER_PARAMS_STATS,
        )

        if op_success is not True or run_params["weights"] != (-1, -1):

            repo_root: Path = Path(__file__).resolve().parent.parent
            stats_dir_path = repo_root / "assets/stats"
            path_to_debug_file = stats_dir_path / f"debug{'_tests' if log_stats_id.endswith('*') else ''}.csv"

            if path_to_debug_file.is_file():
                debug_cases = list(set(get_debug_cases(path_to_debug_file)))
                new_case_info = tuple([circuit_name] + [list(aux_stats[4][0].keys())[0], list(aux_stats[4][0].values())[0]] + list(run_params.values()))
            if not path_to_debug_file.is_file() or (new_case_info not in debug_cases):
                log_stats(
                    aux_stats,
                    f"debug{'_tests' if log_stats_id.endswith('*') else ''}",
                    opt_header=HEADER_PARAMS_STATS,
                )

    elif "pathfinder" in stats_type:
        op_type = "creation" if not tgt_block_info[1] else "discovery"
        durations = {
            "total": (
                (times["t_end"] - times["t1"]).total_seconds()
                if times["t1"] and times["t_end"]
                else "error"
            )
        }

        main_stats = [
            log_stats_id,
            op_type,
            op_success,
            src_block_info[0] if src_block_info else "error",
            src_block_info[1] if src_block_info else "error",
            tgt_block_info[0] if tgt_block_info[0] else "TBD",
            tgt_zx_type,
            tgt_block_info[1] if tgt_block_info[1] else "TBD",
            counts["num_tent_coords"],
            counts["num_tent_coords_filled"],
            counts["max_manhattan"],
            counts["len_longest_path"],
            visit_stats[0],
            visit_stats[1],
            durations["total"],
        ]

    # Call logger
    log_stats(
        main_stats,
        f"{stats_type}{'_tests' if log_stats_id.endswith('*') else ''}",
        opt_header=(
            HEADER_BFS_MANAGER_STATS
            if "graph_manager" in stats_type
            else HEADER_PATHFINDER_STATS
        ),
    )


def log_stats(stats_line: List[Any], stats_type: str, opt_header: List[str] = []):
    """Write statistics to an arbitrary statistic files in `./assets/stats/`

    Args:
        stats_line: A full stats object formatted as a single line for a CSV data file.
        stats_type: The type of statistics being logged, which matches the name of recipient file.
        opt_header (optional): A line to use as header.
    """

    repo_root: Path = Path(__file__).resolve().parent.parent.parent.parent
    stats_dir_path = repo_root / "benchmark/data"

    if not os.path.exists(f"{stats_dir_path}/{stats_type}.csv"):
        with open(f"{stats_dir_path}/{stats_type}.csv", "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(opt_header)

    with open(f"{stats_dir_path}/{stats_type}.csv", "a", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(stats_line)
        f.close()


#################
# STATS READERS #
#################
def get_debug_cases(path_to_stats: Path) -> List[Tuple[str, int, str]]:
    """Get key replicability information for any failed case from output stats.

    This function gets information needed to replicate any failed cases logged to the corresponding
    stats file.

    Args:
        path_to_stats: The path to an existing debug stats file, with failed cases in it.

    Returns
        debug_cases: List with information needed to replicate any failed cases. 

    """

    # Extract cases from file
    debug_cases_full = []
    try: 
        with open(path_to_stats, "r") as f:
            entries = list(csv.reader(f, delimiter=';'))[1:]
            for entry in entries:
                debug_cases_full.append(entry)
        f.close()
    except FileNotFoundError:
        raise FileNotFoundError(f"File `{path_to_stats}` must exist.\n")
    except (IOError, OSError, ValueError):
        raise ValueError(f"Uknown error while reading `{path_to_stats}`")

    # Extract the specific parameters needed to replicate case
    debug_cases = []
    for case in debug_cases_full:
        circuit_name = case[2]
        min_success_rate, weights, len_of_beams = eval(case[3]).values()
        first_id, first_kind = list(eval(case[4])[0].items())[0]
        debug_cases.append((circuit_name, first_id, first_kind, min_success_rate, weights, len_of_beams))

    return debug_cases
