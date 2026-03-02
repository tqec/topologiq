"""Util facilities for logging and reading stats.

Usage:
    Call any function/class from a separate script.

"""

import csv
import os
from ast import literal_eval
from datetime import datetime
from pathlib import Path
from typing import Any

from topologiq.utils.classes import SimpleDictGraph, StandardBlock, StandardCoord

#############
# CONSTANTS #
#############
HEADER_BFS_MANAGER_STATS = [
    "unique_run_id",
    "run_success",
    "circuit_name",
    "zx_spiders_num",
    "zx_edges_num",
    "std_edges_processed",
    "cross_edges_processed",
    "num_edges_in_edge_paths",
    "num_blocks_output",
    "num_edges_output",
    "volume",
    "t_std_edges",
    "t_cross_edges",
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
    "run_success",
    "circuit_name",
    "kwargs",
    "edge_paths",
]


############
# WRITE  #
############
def write_bgraph(
    path_to_output_file: Path | str,
    lat_nodes: dict[int, StandardBlock],
    lat_edges: dict[tuple[int, int], list[str]],
    in_spiders: list[int] = [],
    out_spiders: list[int] = [],
):
    """Write final outputs to a `.bgraph` file.

    Args:
        path_to_output_file: Path to the .bgraph file being written.
        lat_nodes: The cubes of the final space-time diagram produced by Topologiq.
        lat_edges: The pipes of the final space-time diagram produced by Topologiq.
        in_spiders: A list of in/start (depth=0) boundary spiders.
        out_spiders: A list of out/end (depth=depth) boundary spiders

    """

    with open(path_to_output_file, "w") as f:
        f.write("BLOCKGRAPH 0.1.0;\n")
        f.write("\nCUBES: index;x;y;z;kind;label;\n")
        f.writelines(
            [
                f"{cube_id};{';'.join([str(c) for c in cube_info[0]])};{cube_info[1]};{add_port_label(cube_id, in_spiders, out_spiders)};\n"
                for cube_id, cube_info in lat_nodes.items()
            ]
        )

        f.write("\nPIPES: src;tgt;kind;\n")
        f.writelines(
            [
                f"{src_id!s};{src_id!s};{pipe_info[0]};\n"
                for (src_id, tgt_id), pipe_info in lat_edges.items()
            ]
        )


def write_outputs(
    simple_graph: SimpleDictGraph,
    circuit_name: str,
    edge_paths: dict,
    lat_nodes: dict[int, StandardBlock],
    lat_edges: dict[tuple[int, int], list[str]],
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

    lines: list[str] = []

    lines.append(f"RESULT SHEET. CIRCUIT NAME: {circuit_name}\n")

    lines.append("\n__________________________\nORIGINAL ZX GRAPH\n")
    lines.extend([f"Node ID: {node[0]}. Type: {node[1]}\n" for node in simple_graph["nodes"]])
    lines.append("\n")
    lines.extend([f"Edge ID: {edge[0]}. Type: {edge[1]}\n" for edge in simple_graph["edges"]])

    lines.append(
        '\n__________________________\n3D "EDGE PATHS" (Blocks needed to connect two original nodes)\n'
    )
    lines.extend(
        [
            f"Edge {edge_path['src_tgt_ids']}: {edge_path['path_nodes']}\n"
            for key, edge_path in edge_paths.items()
        ]
    )

    lines.append("\n__________________________\nLATTICE SURGERY (Graph)\n")
    lines.extend([f"Node ID: {key}. Info: {node}\n" for key, node in lat_nodes.items()])
    lines.extend(
        [
            f"Edge ID: {key}. Kind: {edge_info[0]}. Original edge in ZX graph: {edge_info[1]} \n"
            for key, edge_info in lat_edges.items()
        ]
    )

    with open(f"{output_dir_path}/{circuit_name}.txt", "w") as f:
        f.writelines(lines)
        f.close()


def prep_stats_n_log(
    stats_type: str,
    op_success: bool,
    counts: dict[str, int],
    times: dict[str, datetime | None],
    circuit_name: str = "unknown",
    edge_paths: dict | None = None,
    lat_nodes: dict[int, StandardBlock] | None = None,
    lat_edges: dict[tuple[int, int], list[str]] | None = None,
    src_block_info: StandardBlock | None = None,
    tgt_block_info: tuple[StandardCoord | None | list[StandardCoord], str | None] = (None, None),
    tgt_zx_type: str | None = None,
    visit_stats: tuple[int, int] = (0, 0),
    cross_edge: bool = False,
    **kwargs,
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
        cross_edge: False if pathfinder ran to add a new cube, True if it found edge between existing cubes.
        **kwargs: !

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
        total_cubes_in_output = len(lat_nodes.keys()) if lat_nodes else 0
        total_pipes_in_output = len(lat_edges.keys()) if lat_edges else 0
        volume = len([True for _, kind in lat_nodes.values() if kind != "ooo"]) if lat_nodes else 0

        main_stats = [
            kwargs["log_stats_id"],
            op_success,
            circuit_name,
            counts["zx_spiders_num"],
            counts["zx_edges_num"],
            counts["std_edges_processed"],
            counts["cross_edges_processed"],
            len(edge_paths) if edge_paths else 0,
            total_cubes_in_output,
            total_pipes_in_output,
            volume,
            round(times["t_std_edges"], 3),
            round(times["t_cross_edges"], 3),
            round(times["t_total"], 3),
        ]

        try:
            edge_paths_summary = (
                [
                    {
                        p["src_tgt_ids"][0] if p["src_tgt_ids"] != "error" else key[0]: p[
                            "path_nodes"
                        ][0][1]
                        if (p["path_nodes"] != "error" and p["path_nodes"][0][1])
                        else None,
                        p["src_tgt_ids"][1] if p["src_tgt_ids"] != "error" else key[1]: p[
                            "path_nodes"
                        ][-1][1]
                        if (p["path_nodes"] != "error" and p["path_nodes"][-1][1])
                        else None,
                    }
                    for key, p in edge_paths.items()
                ]
                if edge_paths
                else ["error"]
            )

        except Exception as e:
            print(f"Minor error with logging of aux stats: {e}.")
            edge_paths_summary = (
                [
                    {
                        src_id: None,
                        tgt_id: None,
                    }
                    for [src_id, tgt_id] in edge_paths.keys()
                ]
                if edge_paths
                else ["error"]
            )

        aux_stats = [
            kwargs["log_stats_id"],
            op_success,
            circuit_name,
            kwargs,
            edge_paths_summary,
        ]

        log_stats(
            aux_stats,
            f"params{'_tests' if kwargs['log_stats_id'].endswith('*') else ''}",
            opt_header=HEADER_PARAMS_STATS,
        )

        if op_success is not True:
            repo_root: Path = Path(__file__).resolve().parent.parent
            stats_dir_path = repo_root / "assets/stats"
            path_to_debug_file = (
                stats_dir_path
                / f"debug{'_tests' if kwargs['log_stats_id'].endswith('*') else ''}.csv"
            )

            if path_to_debug_file.is_file():
                debug_cases = list(set(get_debug_cases(path_to_debug_file)))
                new_case_info = tuple(
                    [
                        circuit_name,
                        list(aux_stats[4][0].keys())[0],
                        list(aux_stats[4][0].values())[0],
                        *list(kwargs.values()),
                    ]
                )
            if not path_to_debug_file.is_file() or (new_case_info not in debug_cases):
                log_stats(
                    aux_stats,
                    f"debug{'_tests' if kwargs['log_stats_id'].endswith('*') else ''}",
                    opt_header=HEADER_PARAMS_STATS,
                )

    elif "pathfinder" in stats_type:
        tgt_coords = tgt_block_info[0]
        tgt_kind = tgt_block_info[1]

        main_stats = [
            kwargs["log_stats_id"],
            "standard" if not cross_edge else "cross",
            op_success,
            src_block_info[0] if src_block_info else "error",
            src_block_info[1] if src_block_info else "error",
            tgt_coords,
            tgt_zx_type,
            tgt_kind,
            counts["num_tent_coords"],
            counts["num_tent_coords_filled"],
            counts["max_manhattan"],
            counts["len_longest_path"],
            visit_stats[0],
            visit_stats[1],
            round(times["duration_pathfinder"], 3),
        ]

    # Call logger
    log_stats(
        main_stats,
        f"{stats_type}{'_tests' if kwargs['log_stats_id'].endswith('*') else ''}",
        opt_header=(
            HEADER_BFS_MANAGER_STATS if "graph_manager" in stats_type else HEADER_PATHFINDER_STATS
        ),
    )


def log_stats(stats_line: list[Any], stats_type: str, opt_header: list[str] = []):
    """Write statistics to an arbitrary statistic files in `./assets/stats/`.

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


def add_port_label(cube_id: int, in_spiders: list[int], out_spiders: list[int]) -> str:
    """Return a label for an arbitrary cube_id depending on whether the ID corresponds to a boundary spider in the original ZX graph.

    Args:
        cube_id: The ID of the cube being examined.
        in_spiders: A list of in/start (depth=0) boundary spiders.
        out_spiders: A list of out/end (depth=depth) boundary spiders

    Returns:
        label: An appropriately formatted in or out port label, or empty string if ID is not a boundary.

    """
    label = ""
    if cube_id in in_spiders:
        label = f"in_{in_spiders.index(cube_id)}"
    if cube_id in out_spiders:
        label = f"out_{out_spiders.index(cube_id)}"
    return label


###########
# READ #
###########
def get_debug_cases(path_to_stats: Path) -> list[tuple[str, int, str]]:
    """Get key replicability information for any failed case from output stats.

    This function gets information needed to replicate any failed cases logged to the corresponding
    stats file.

    Args:
        path_to_stats: The path to an existing debug stats file, with failed cases in it.

    Returns:
        debug_cases: list with information needed to replicate any failed cases.

    """

    # Extract cases from file
    debug_cases_full = []
    try:
        with open(path_to_stats) as f:
            entries = list(csv.reader(f, delimiter=";"))[1:]
            debug_cases_full.extend(entry for entry in entries)
        f.close()
    except FileNotFoundError:
        raise FileNotFoundError(f"File `{path_to_stats}` must exist.\n")
    except (OSError, ValueError):
        raise ValueError(f"Uknown error while reading `{path_to_stats}`")

    # Extract the specific parameters needed to replicate case
    debug_cases = []
    for case in debug_cases_full:
        circuit_name = case[2]
        kwargs = literal_eval(case[3])
        first_id_strategy = kwargs["first_id_strategy"]
        seed = kwargs["seed"]
        log_stats_id = kwargs["log_stats_id"]
        first_id, first_kind = list(literal_eval(case[4])[0].items())[0]
        debug_cases.append(
            (circuit_name, first_id, first_kind, log_stats_id, first_id_strategy, seed)
        )

    return debug_cases
