"""Topologiq's UX temporary graph. SpiderCube, Spider, and Cube class and related methods.

Notes:
    This file is temporary.
    The UX will eventually plug directly into Topologiq's AugmentedNXGraph.

"""

from pathlib import Path

import networkx as nx
from matplotlib import pyplot as plt

from topologiq.utils.classes import StandardCoord
from topologiq.ux.spider_cube import SpiderCube, SpiderCubeRegistry

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

    def populate_from_bgraph(self, path_to_input_file: Path):
        """Populate the graph using pre-parsed lists of node and edge data.

        Args:
            path_to_input_file: The path to the input `.bgraph` file.

        """

        with open(path_to_input_file) as f:
            lines = f.readlines()
            f.close()

        parsed_cubes: dict[str, tuple[StandardCoord, SpiderCube, str]] = {}
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
                spider_cube = SpiderCubeRegistry.get_create(kind=kind)
                parsed_cubes[int(cube_id)] = {
                    "coords": (int(x), int(y), int(z)),
                    "cube": spider_cube,
                    "label": label,
                }

            if parse_mode == "pipes" and line[0].isnumeric():
                src_id, tgt_id, kind = line.strip().split(";")[:-1]
                parsed_pipes[(int(src_id), int(tgt_id))] = {"kind": kind}

        for cube_id, metadata in parsed_cubes.items():
            self.add_element(cube_id, element_type="node", **metadata)

        for pipe_id, metadata in parsed_pipes.items():
            self.add_element(pipe_id, element_type="edge", **metadata)

        print("\n ==> FOUNDATIONAL GRAPH BUILT AS:")
        for item in self.get_all_nodes():
            print(item)
        for item in self.get_all_edges():
            print(item)

    def check_graph_integrity(self):
        """Visualise the undirected graph to verify connectivity."""

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        # 1. Extract positions and colors
        pos = {}
        node_colors = []

        for n, attr in self.G.nodes(data=True):
            # Position from (x, y, z) tuple
            pos[n] = attr.get("coords")

            # Color from the 'cube' object's method
            cube_obj = attr.get("cube")
            if cube_obj and hasattr(cube_obj, "get_zx_color"):
                zx_color = cube_obj.get_zx_color
                print(zx_color)
                node_colors.append(zx_color.value)
            else:
                node_colors.append("#F10BB8")  # Default Gray fallback

        # 2. Draw Edges
        for edge in self.G.edges():
            u, v = edge
            x_pts = [pos[u][0], pos[v][0]]
            y_pts = [pos[u][1], pos[v][1]]
            z_pts = [pos[u][2], pos[v][2]]
            ax.plot(x_pts, y_pts, z_pts, color="#000000")

        # 3. Draw Nodes (scatter accepts a list of hex strings)
        x_nodes, y_nodes, z_nodes = zip(*[pos[n] for n in self.G.nodes()])
        ax.scatter(
            x_nodes,
            y_nodes,
            z_nodes,
            s=100,
            c=node_colors,
            edgecolors="white",
            linewidth=0.5,
            alpha=0.9,
        )

        # 4. Final Styling
        ax.set_title("UX Spatial Graph: Node-Specific Cube Colors")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Optional: Make the background dark for better hex color contrast
        ax.set_facecolor("#f0f0f0")

        plt.show()


if __name__ == "__main__":
    graph = UXGraphManager()
    path_to_bgraph = Path(UX_ROOT / "cnots.bgraph")
    graph.populate_from_bgraph(path_to_bgraph)
    graph.check_graph_integrity()
