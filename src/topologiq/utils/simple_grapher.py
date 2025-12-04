"""Quick and simple grapher for dictionary-based graphs.

Usage:
    Call `simple_graph_vis` from a separate script.

"""

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from topologiq.utils.classes import SimpleDictGraph


def simple_graph_vis(simple_graph: SimpleDictGraph, layout_method: str = "spectral") -> Figure:
    """Visualise a graph using only Matplotlib objects.

    Args:
        simple_graph: A dictionary with 'nodes' and 'edges' keys, representing the graph.
        layout_method: The NX layout method to use to define node positions.

    """

    # Colours
    hex_map = {
        "X": "#d7a4a1",
        "Y": "#a8e6cf",
        "Z": "#b9cdff",
        "O": "#555",
        "SIMPLE": "#000",
        "HADAMARD": "#1f2df1",
    }

    # Temporary NX graph for layout purposes
    temp_g = nx.Graph()
    for node_id, node_type in simple_graph["nodes"]:
        temp_g.add_node(node_id)
    for (u, v), edge_type in simple_graph["edges"]:
        temp_g.add_edge(u, v)

    # Layout
    positions = {}
    if layout_method == "spring":
        positions = nx.spring_layout(temp_g, iterations=100)
    elif layout_method == "circular":
        positions = nx.circular_layout(temp_g)
    elif layout_method == "shell":
        positions = nx.shell_layout(temp_g)
    elif layout_method == "kamada_kawai":
        positions = nx.kamada_kawai_layout(temp_g)
    elif layout_method == "spectral":
        positions_spectral = nx.spectral_layout(temp_g)
        positions = nx.kamada_kawai_layout(temp_g, pos=positions_spectral)
    elif layout_method == "planar":
        if not nx.is_planar(temp_g):
            print("Warning: Graph is not planar. Falling back to 'spring' layout.")
            positions = nx.spring_layout(temp_g, iterations=100)
        else:
            positions_spring = nx.spring_layout(temp_g, iterations=100)
            positions = positions_spring
    else:
        print(f"Warning: Unknown layout '{layout_method}'. Using 'spring' as default.")
        positions = nx.spring_layout(temp_g, iterations=100)

    # Initialise figure
    fig, ax = plt.subplots()

    # Edges with low z-order
    for (u, v), edge_type in simple_graph["edges"]:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        line = Line2D(
            [float(x1), float(x2)],
            [float(y1), float(y2)],
            color=hex_map.get(edge_type, "black"),
            linewidth=1.5,
            zorder=1,
            alpha=0.5,
        )
        ax.add_line(line)

    # Nodes with higher z-order
    node_radius = 0.05
    for node_id, node_type in simple_graph["nodes"]:
        x, y = positions[node_id]
        circle = Circle(
            (float(x), float(y)),
            radius=node_radius,
            color=hex_map.get(node_type, "pink"),
            zorder=2,
        )
        ax.add_patch(circle)

    # Labels with highest z-order
    label_offset = 0.04
    for node_id, node_type in simple_graph["nodes"]:
        x, y = positions[node_id]
        ax.text(
            float(x + label_offset),
            float(y + label_offset),
            str(node_id),
            fontsize=10,
            ha="left",
            va="bottom",
            fontweight="bold",
            zorder=3,
        )

    ax.set_aspect("equal")
    ax.autoscale_view()
    plt.axis("off")
    plt.show()

    return fig
