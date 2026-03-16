"""Topologiq's UX temporary graph.

Notes:
    This file is temporary.
    The UX will eventually plug directly into Topologiq's AugmentedNXGraph.

"""

from pathlib import Path

import networkx as nx
import pyzx as zx
from matplotlib import pyplot as plt

from topologiq.core.spider_block import SpiderBlock, SpiderBlockRegistry
from topologiq.utils.classes import StandardCoord

#########
# PATHS #
#########
UX_ROOT: Path = Path(__file__).resolve().parent


#################
# GRAPH MANAGER #
#################
class UXGraphManager:
    """Temporary Graph Object to run tests with."""

    def __init__(self):
        """Initialise graph."""
        self.G = nx.Graph()

    def add_element(self, element_id, element_type="node", **metadata):
        """Add a node or edge with flexible metadata."""
        if element_type == "node":
            self.G.add_node(element_id, **metadata)
        elif element_type == "edge":
            u, v = element_id
            self.G.add_edge(u, v, **metadata)

    def get_element(self, element_id, is_edge=False):
        """Retrieve attributes for a specific node or edge."""
        try:
            if is_edge:
                return self.G.edges[element_id]
            return self.G.nodes[element_id]
        except KeyError:
            return None

    def get_all_nodes(self):
        """Return nodes in a format easy for UI lists/grids."""
        return [{"id": n, **attr} for n, attr in self.G.nodes(data=True)]

    def get_all_edges(self):
        """Return edges in a format easy for canvas connectors."""
        return [{"source": u, "target": v, **attr} for u, v, attr in self.G.edges(data=True)]

    def delete_element(self, element_id, is_edge=False):
        """Delete element from graph."""
        if is_edge:
            self.G.remove_edge(*element_id)
        else:
            self.G.remove_node(element_id)

    def from_bgraph_file(self, path_to_input_file: Path):
        """Populate the graph using pre-parsed lists of node and edge data.

        Args:
            path_to_input_file: The path to the input `.bgraph` file.

        """

        with open(path_to_input_file) as f:
            lines = f.readlines()
            f.close()

        parsed_cubes: dict[str, tuple[StandardCoord, SpiderBlock, str]] = {}
        parsed_pipes: dict[str, tuple[StandardCoord, str, str]] = {}
        parse_mode = None
        for line in lines:
            if line.startswith("CUBES: "):
                parse_mode = "cubes"
                continue
            if line.startswith("PIPES: "):
                parse_mode = "pipes"
                continue

            if parse_mode == "cubes" and line[0].isnumeric():
                cube_id, x, y, z, kind, label = line.strip().split(";")[:-1]
                spider_block = SpiderBlockRegistry.get_create(kind=kind)
                parsed_cubes[int(cube_id)] = {
                    "coords": (int(x), int(y), int(z)),
                    "spider_block": spider_block,
                    "label": label,
                }

            if parse_mode == "pipes" and line[0].isnumeric():
                src_id, tgt_id, kind = line.strip().split(";")[:-1]
                spider_block = SpiderBlockRegistry.get_create(kind=kind)
                parsed_pipes[(int(src_id), int(tgt_id))] = {"spider_block": spider_block}

        for cube_id, metadata in parsed_cubes.items():
            self.add_element(cube_id, element_type="node", **metadata)

        for pipe_id, metadata in parsed_pipes.items():
            self.add_element(pipe_id, element_type="edge", **metadata)

    def to_zx_graph(self) -> zx.Graph:
        """Convert NX graph to ZX graph.

        Returns:
            positioned_zx: An positioned ZX graph where all cubes in the blockgraph become spiders.
            reduced_zx: A reduced version of the positioned_zx graph.

        """

        type_conversions = {
            "BOUNDARY": 0,
            "X": 2,
            "Z": 1,
            "SIMPLE": 1,
            "HADAMARD": 2,
        }

        positions_2d = nx.spring_layout(self.G)

        zx_graph = zx.Graph()
        id_conversions = {}

        for n, attrs in self.G.nodes(data=True):
            coords = attrs.get("coords")

            qubit = positions_2d[n][0] * 10
            row = positions_2d[n][1] * 10

            spider_block = attrs.get("spider_block")
            zx_type = spider_block.zx_type

            vertex = zx_graph.add_vertex(
                ty=type_conversions[zx_type],
                qubit=qubit,
                row=row,
            )
            zx_graph.set_vdata(vertex, "coords", coords)

            id_conversions[n] = vertex

        for src_id, tgt_id, attrs in self.G.edges(data=True):
            spider_block = attrs.get("spider_block")
            zx_type = spider_block.zx_type
            zx_graph.add_edge(
                (id_conversions[src_id], id_conversions[tgt_id]), edgetype=type_conversions[zx_type]
            )

        draw_zx_3d(zx_graph)

        zx.full_reduce(zx_graph)
        draw_zx_3d(zx_graph)

    def check_graph_integrity(self):
        """Visualise the undirected graph to verify connectivity."""

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        # Draw nodes
        pos = {}
        node_colors = []
        edge_colors = []

        for n, attrs in self.G.nodes(data=True):
            pos[n] = attrs.get("coords")

            spider_block = attrs.get("spider_block")
            if spider_block and hasattr(spider_block, "get_zx_color"):
                zx_color = spider_block.get_zx_color
                node_colors.append(zx_color.value)
            else:
                node_colors.append("#F10BB8")  # Fallback

        x_pos, y_pos, z_pos = zip(*[pos[n] for n in self.G.nodes()])
        ax.scatter(
            x_pos,
            y_pos,
            z_pos,
            s=100,
            c=node_colors,
            edgecolors="black",
            linewidth=1,
        )

        # Draw edges
        for src_id, tgt_id, attrs in self.G.edges(data=True):
            x_pts = [pos[src_id][0], pos[tgt_id][0]]
            y_pts = [pos[src_id][1], pos[tgt_id][1]]
            z_pts = [pos[src_id][2], pos[tgt_id][2]]

            spider_block = attrs["spider_block"]
            if spider_block and hasattr(spider_block, "get_zx_color"):
                zx_color = spider_block.get_zx_color
                edge_colors.append(zx_color.value)
            else:
                edge_colors.append("#F10BB8")  # Fallback

            ax.plot(x_pts, y_pts, z_pts, color=edge_colors[-1])

        # 4. Final Styling
        ax.set_title("UX Spatial Graph: Node-Specific Cube Colors")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Optional: Make the background dark for better hex color contrast
        ax.set_facecolor("#f0f0f0")

        plt.show()


def draw_zx_3d(g):
    """Draw a PyZX Graph using 3D coordinates."""
    # 1. Manually build the NetworkX graph
    g_nx = nx.Graph()
    pos_3d = {}
    node_colors = []

    for v in g.vertices():
        # Retrieve your custom 3D coords
        coords = g.vdata(v, "coords")
        pos_3d[v] = coords

        # Add node to NX with metadata
        g_nx.add_node(v, pos=coords, type=g.type(v))

        # Color mapping: Z=green, X=red, Boundary=gold
        t = g.type(v)
        node_colors.append("green" if t == 1 else "red" if t == 2 else "gray")

    for e in g.edges():
        u, v = e
        g_nx.add_edge(u, v, etype=g.edge_type(e))

    # 2. Plotting with Matplotlib 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw Nodes
    xs, ys, zs = zip(*[pos_3d[v] for v in g_nx.nodes()])
    ax.scatter(xs, ys, zs, c=node_colors, s=100, edgecolors="black", depthshade=False)

    # Draw Edges
    for u, v, data in g_nx.edges(data=True):
        x_line = [pos_3d[u][0], pos_3d[v][0]]
        y_line = [pos_3d[u][1], pos_3d[v][1]]
        z_line = [pos_3d[u][2], pos_3d[v][2]]

        # Style: Dashed for Hadamard (type 2), Solid for Simple (type 1)
        is_hadamard = data["etype"] == 2
        ax.plot(
            x_line,
            y_line,
            z_line,
            color="blue" if is_hadamard else "black",
            linestyle="--" if is_hadamard else "-",
            alpha=0.6,
        )

    ax.set_title("3D PyZX Visualization")
    plt.show()


if __name__ == "__main__":
    # ZX graph
    # zx_graph = UXGraphManager()
    # path_to_zx_graph_file = Path(UX_ROOT / "zx_cnots.json")
    # zx_graph.populate_from_zx_graph_file(path_to_zx_graph_file)

    # Blockgraph
    blockgraph = UXGraphManager()
    path_to_bgraph_file = Path(UX_ROOT / "cnots.bgraph")
    blockgraph.from_bgraph_file(path_to_bgraph_file)
    # blockgraph.check_graph_integrity()
    blockgraph.to_zx_graph()
