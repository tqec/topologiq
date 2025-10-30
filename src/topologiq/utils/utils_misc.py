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
def get_manhattan(src_c: StandardCoord, tgt_c: StandardCoord) -> int:
    """Gets the Manhattan distance between any two (x, y, z) coordinates.
    Args:
        - src_c: (x, y, z) coordinates for the source block.
        - tgt_c: (x, y, z) coordinates for the target block.
    Returns:
        - [int]: the Manhattan distance between the two incoming coordinate tuples
    """

    return np.sum(np.abs(np.array(src_c) - np.array(tgt_c)))


def get_max_manhattan(src_c: StandardCoord, all_cs: List[StandardCoord]) -> int:
    """Determines the maximum Manhattan distance and a list of (x, y, z) coordinates of any arbitrary length.
    Args:
        - src_c: (x, y, z) coordinates for the source block.
        - all_cs: a list of (x, y, z) coordinates of any arbitrary length, which may include src_c.
    Returns:
        - [int]: the max. Manhattan distance between the given source coordinate and all coordinate tuples in the list of coordinates.
    """

    if all_cs:
        return max([get_manhattan(src_c, c) for c in all_cs])

    return 0


##################
# STATS LOGGERS  #
##################
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
    """Takes a list of arguments and assembles them in the appropriate order needed to log stats to file. Uses the type of stats to determine appropriate order

    Args:
        -

    Keyword arguments (kwargs):
        - This varies per operation. See "CONSTANTS" section above in this file for all possibilities.

    Returns:
        - main_stats: array containing all principal stats to be logged to the corresponding stats file.
        - aux_stats: array containing any secondary stats to be logged to any auxiliary stats file, or an empty array if no secondary stats exist.
    """

    # INITIALISE ARRAYS
    main_stats = []
    aux_stats = []

    # FILL ARRAYS AS PER LOG TYPE
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

        aux_stats = [
            log_stats_id,
            op_success,
            circuit_name,
            run_params,
                        (
                [
                    {
                        edge_path["src_tgt_ids"][0]: edge_path["path_nodes"][0][1],
                        edge_path["src_tgt_ids"][1]: edge_path["path_nodes"][-1][1]
                    } 
                    for edge_path in edge_paths.values()
                ]
                if edge_paths
                else ["error"]
            ),
        ]

        log_stats(
            aux_stats,
            f"params{'_tests' if log_stats_id.endswith('*') else ''}",
            opt_header=HEADER_PARAMS_STATS,
        )

        if op_success is not True or run_params["length_of_beams"] != 9 or run_params["weights"] != (-1, -1):
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
    """Writes statistics to one of the statistics files in `./assets/stats/`
    Args:
        - stats_line: the line of statistics to be logged to file.
        - stats_type: the type of statistics being logged to file, which also matches the name of recipient file.
        - init_file: prompts the function to fully erase the destination file and start by writing headers to it.
    Returns:
        - n/a: stats are written to .csv files in `./assets/stats/`
    """

    repo_root: Path = Path(__file__).resolve().parent.parent
    stats_dir_path = repo_root / "assets/stats"

    if not os.path.exists(f"{stats_dir_path}/{stats_type}.csv"):
        with open(f"{stats_dir_path}/{stats_type}.csv", "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(opt_header)

    with open(f"{stats_dir_path}/{stats_type}.csv", "a", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(stats_line)
        f.close()


def write_outputs(
    c_g_dict: SimpleDictGraph,
    circuit_name: str,
    edge_paths: dict,
    lat_nodes: dict[int, StandardBlock],
    lat_edges: dict[Tuple[int, int], List[str]],
    output_dir_path: Path,
):
    """Writes the final results of the run to TXT file.
    Args:
        - c_g_dict: a ZX circuit as a simple dictionary of nodes and edges
        - circuit_name: name of ZX circuit
        - edge_paths: the raw set of 3D edges found by the algorithm (with redundant blocks for start and end positions of some edges)
        - lat_nodes: the nodes/blocks of the resulting space-time diagram (without redundant blocks)
        - lat_edges: the edges/pipes of the resulting space-time diagram (without redundant pipes)
        - output_dir_path: the directory where outputs are saved.

    Returns
        - n/a: stats are written to .csv files in `output_dir_path`
    """

    lines: List[str] = []

    lines.append(f"RESULT SHEET. CIRCUIT NAME: {circuit_name}\n")
    lines.append("\n__________________________\nORIGINAL ZX GRAPH\n")
    for node in c_g_dict["nodes"]:
        lines.append(f"Node ID: {node[0]}. Type: {node[1]}\n")
    lines.append("\n")
    for edge in c_g_dict["edges"]:
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


#################
# STATS READERS #
#################
def get_debug_cases(path_to_stats: Path) -> List[Tuple[str, int, str]]:
    """Get key replicability information for any failed case from output stats.

    Returns
        - debug_cases: list of (name, first_id, first_kind) for all failed cases in output stats log. 

    """

    # EXTRACT CASES FROM DEBUG CASES LOG FILE
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

    # EXTRACT PARAMS NEEDED TO REPRODUCE CASE
    debug_cases = []
    for case in debug_cases_full:
        circuit_name = case[2]
        min_success_rate, weights, len_of_beams = eval(case[3]).values()
        first_id, first_kind = list(eval(case[4])[0].items())[0]
        debug_cases.append((circuit_name, first_id, first_kind, min_success_rate, weights, len_of_beams))

    return debug_cases
