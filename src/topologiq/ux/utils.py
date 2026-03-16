"""Quick utils to assist UX."""

import json
from pathlib import Path

import pyzx as zx

from topologiq.assets.pyzx_graphs import cnots


def write_zx_to_json_file(zx_graph : zx.Graph, path_to_output_file: Path):
    """Write a PyZX graph to a JSON file."""

    json_data = zx_graph.to_json()
    with open("zx_cnots.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    zx_cnots, _ = cnots()
    zx.draw(zx_cnots)
    zx.full_reduce(zx_cnots)
    zx.draw(zx_cnots)

    #write_zx_to_json_file(zx_cnots)
