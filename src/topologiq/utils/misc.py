"""Misc utils of various sorts.

Usage:
    Call any function/class from a separate script.

"""

import json
from pathlib import Path

import pyzx as zx


def kind_to_zx_type(kind: str) -> str:
    """Get the ZX type corresponding to a given block or pipe kind.

    Args:
        kind: the /kind of a given block.

    Returns:
        zx_type: the ZX type corresponding to the kind.

    """

    if kind == "ooo":
        zx_type = "BOUNDARY"
    elif "o" in kind:
        zx_type = "HADAMARD" if "h" in kind else "SIMPLE"
    else:
        zx_type = min(set(kind), key=lambda c: kind.count(c)).capitalize()
    return zx_type


def write_zx_to_json_file(zx_graph: zx.Graph, path_to_output_file: Path):
    """Write a PyZX graph to a JSON file."""

    json_data = zx_graph.to_json()
    with open("zx_cnots.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)
