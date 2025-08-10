import os
import csv
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union

from utils.classes import SimpleDictGraph, StandardBlock, StandardCoord


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


def log_stats_to_file(stats_line: List, stats_type: str, opt_header: List[str] = []):
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
            writer = csv.writer(f)
            writer.writerow(opt_header)

    with open(f"{stats_dir_path}/{stats_type}.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(stats_line)
        f.close()


def write_outputs(
    c_g_dict: SimpleDictGraph,
    c_name: str,
    edge_pths: dict,
    lat_nodes: dict[int, StandardBlock],
    lat_edges: dict[Tuple[int, int], List[str]],
    out_dir_pth: Path,
):
    """Writes the final results of the run to TXT file.
    Args:
        - c_g_dict: a ZX circuit as a simple dictionary of nodes and edges
        - c_name: name of ZX circuit
        - edge_pths: the raw set of 3D edges found by the algorithm (with redundant blocks for start and end positions of some edges)
        - lat_nodes: the nodes/blocks of the resulting space-time diagram (without redundant blocks)
        - lat_edges: the edges/pipes of the resulting space-time diagram (without redundant pipes)
        - out_dir_pth: the directory where outputs are saved.

    Returns
        - n/a: stats are written to .csv files in `out_dir_pth`
    """

    lines: List[str] = []

    lines.append(f"RESULT SHEET. CIRCUIT NAME: {c_name}\n")
    lines.append("\n__________________________\nORIGINAL ZX GRAPH\n")
    for node in c_g_dict["nodes"]:
        lines.append(f"Node ID: {node[0]}. Type: {node[1]}\n")
    lines.append("\n")
    for edge in c_g_dict["edges"]:
        lines.append(f"Edge ID: {edge[0]}. Type: {edge[1]}\n")

    lines.append(
        '\n__________________________\n3D "EDGE PATHS" (Blocks needed to connect two original nodes)\n'
    )

    for key, edge_pth in edge_pths.items():
        lines.append(f"Edge {edge_pth['src_tgt_ids']}: {edge_pth['pth_nodes']}\n")

    lines.append("\n__________________________\nLATTICE SURGERY (Graph)\n")
    for key, node in lat_nodes.items():
        lines.append(f"Node ID: {key}. Info: {node}\n")
    for key, edge_info in lat_edges.items():
        lines.append(
            f"Edge ID: {key}. Kind: {edge_info[0]}. Original edge in ZX graph: {edge_info[1]} \n"
        )

    with open(f"{out_dir_pth}/{c_name}.txt", "w") as f:
        f.writelines(lines)
        f.close()
