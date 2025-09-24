import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.figure import Figure
from utils.classes import SimpleDictGraph


def simple_graph_vis(
    simple_graph: SimpleDictGraph, layout_method: str = "spectral"
) -> Figure:
    """
    Visualizes a graph using only Matplotlib objects.

    Args:
        simple_graph: A dictionary with 'nodes' and 'edges' keys, representing the graph.
    """

    # COLOURS
    hex_map = {
        "X": "#FF6961",
        "Y": "#A8E6CF",
        "Z": "#77DD77",
        "O": "#555555",
        "SIMPLE": "#000000",
        "HADAMARD": "#1F2DF1",
    }

    # CREATE TEMP NX GRAPH FOR LAYOUT & DETERMINE LAYOUT
    # Graph
    G_temp = nx.Graph()
    for node_id, node_type in simple_graph["nodes"]:
        G_temp.add_node(node_id)
    for (u, v), edge_type in simple_graph["edges"]:
        G_temp.add_edge(u, v)

    # Layout
    pos = {}
    if layout_method == "spring":
        pos = nx.spring_layout(G_temp, iterations=100)
    elif layout_method == "circular":
        pos = nx.circular_layout(G_temp)
    elif layout_method == "shell":
        pos = nx.shell_layout(G_temp)
    elif layout_method == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G_temp)
    elif layout_method == "spectral":
        pos_spectral = nx.spectral_layout(G_temp)
        pos = nx.kamada_kawai_layout(G_temp, pos=pos_spectral)
    elif layout_method == "planar":
        if not nx.is_planar(G_temp):
            print("Warning: Graph is not planar. Falling back to 'spring' layout.")
            pos = nx.spring_layout(G_temp, iterations=100)
        else:
            pos_spring = nx.spring_layout(G_temp, iterations=100)
            pos = pos_spring
    else:
        print(f"Warning: Unknown layout '{layout_method}'. Using 'spring' as default.")
        pos = nx.spring_layout(G_temp, iterations=100)

    # ASSEMBLE FIGURE
    fig, ax = plt.subplots()

    # Edges with low z-order
    for (u, v), edge_type in simple_graph["edges"]:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
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
        x, y = pos[node_id]
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
        x, y = pos[node_id]
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

    # RETURN FIG FOR CONSUMPTION IN OTHER VISUALISATIONS
    return fig
