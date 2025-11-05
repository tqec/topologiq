"""
Contains visualisation options for the inner pathfinder algorithm. 

This file contains functions that can help create full-detail visualisations of how
the pathfinder algorithm goes about resolving specific edges.

Usage:
    Call `visualise_pathfinder_iteration()` programmatically, with an appropriate parameter combination. 

Notes:
    The visualisations in this file are not default in standard outputs due to being quite resource intensive. 
        By extension, expect runtimes to increase considerably. 
"""

import numpy as np
import networkx as nx
#from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#from typing import Annotated, Literal, Any, Optional, Tuple, List
from typing import List
#from numpy.typing import NDArray

from topologiq.utils.grapher import render_edge
from topologiq.utils.utils_pathfinder import rot_o_kind  # check_is_exit, 
from topologiq.utils.classes import StandardBlock, StandardCoord
from topologiq.utils.grapher_common import node_hex_map, render_block
from topologiq.utils.utils_zx_graphs import kind_to_zx_type


# MAIN VISUALISATION FUNCTION
def visualise_pathfinder_evolution(
    valid_paths: dict[StandardBlock, List[StandardBlock]] | None,
    all_paths_in_round: dict[StandardBlock, List[StandardBlock]] | None,
    src_block_info: StandardBlock,
    tent_coords: List[StandardCoord],
    tent_tgt_kinds: List[str],
    taken: List[StandardCoord] = [],
    vis_discarded_paths: bool = False,
):
    """Create a visualisation of each pathfinder cycle that includes valid and discarded paths, 
    displaying the discarded (invalid) paths sequentially as an animation.

    Args:
        valid_paths: All paths found in a pathfinder round covering some or all tent_coords.
        all_paths_in_round: All paths discarded in a pathfinder round.
        src_block_info: The source block (coordinates and kind).
        tent_coords: A list of tentative target coordinates to find paths to.
        tent_tgt_kinds: list of kinds matching the zx-type of target block
        A list of all coordinates occupied by any blocks/pipes placed throughout the algorithmic process
    
    Outputs:
        None: A Matplotlib animation showing all valid_paths found in a round by the inner pathfinder
            algorithm and snapshots of paths tested and discarded in round.

    AI disclaimer:
        Category: Assisted. Categorization performed by the model according to the categories given to it.
        Model: Gemini 2.5 Flash.
    """

    # Preliminaries
    current_grapher_id = 0

    # 1. Pre-process dynamic data (all_paths_in_round)
    # ---
    # Convert all_paths_in_round into a list of (N, 3) np.array, each a path/frame.
    path_frames_data = []
    path_colors = []
    if all_paths_in_round:
        for path in all_paths_in_round.values():
            # Check if this path matches a valid one (successful)
            path_coords = np.array([block[0] for block in path])
            path_frames_data.append(path_coords)
            
            if path in valid_paths.values():
                path_colors.append('green')
            else:
                path_colors.append('red')

    num_paths = len(path_frames_data)
    num_frames = num_paths + 1 # +1 is used as hold frame

    # Calculate required interval for the target duration
    TARGET_DURATION_MS = 10000 if num_frames > 100 else 5000 if num_frames > 20 else 1000
    if num_paths > 0:
        animation_interval_ms = TARGET_DURATION_MS / num_paths
    # Default to 500ms if no paths
    else:
        animation_interval_ms = 500
    animation_interval_ms = int(animation_interval_ms)
    
    # 2. Setup static plot elements
    # ---
    # Create foundational Matplotlib objects
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", computed_zorder=False)

    # Visualise built-lattice
    size = [1.0, 1.0, 1.0]
    if taken:
        for taken_coords in taken:
            render_block(
                ax,
                current_grapher_id,
                taken_coords,
                size,
                "aaa",
                node_hex_map,
            )

    # Add source cube
    size = [1.0, 1.0, 1.0]
    src_coords, src_kind = src_block_info
    render_block(
        ax,
        current_grapher_id,
        src_coords,
        size,
        src_kind[:3],
        node_hex_map,
        edge_col = "violet",
        alpha=1,
    )
    current_grapher_id += 1

    # Add target cube
    if len(tent_tgt_kinds) == 1:
        tgt_kind = tent_tgt_kinds[0]
    else:
        zx_type = kind_to_zx_type(tent_tgt_kinds[0])
        tgt_kind = zx_type.lower()*3 if zx_type in ["X", "Y", "Z"] else "ooo"
    
    size = [1.0, 1.0, 1.0] if tgt_kind != "ooo" else [0.7, 0.7, 0.7]
    for tent_coord in tent_coords:
        render_block(
            ax,
            current_grapher_id,
            tent_coord,
            size,
            tgt_kind,
            node_hex_map,
            edge_col = "white",
            alpha=1,
        )
    current_grapher_id += 1

    # Render all valid paths (Code for valid paths remains static)
    graph = edge_paths_to_g(valid_paths)

    # Positions and types
    node_positions = nx.get_node_attributes(graph, "coords")
    node_types = nx.get_node_attributes(graph, "type")
    edge_types = nx.get_edge_attributes(graph, "pipe_type")

    # Cubes (valid paths)
    for node_id in graph.nodes():
        node_type = node_types.get(node_id)
        if (
            node_type
            and "o" not in node_type
        ):
            node_coords = node_positions.get(node_id)
            if node_coords:
                size = [1.0, 1.0, 1.0]
                edge_col = "black"
                alpha = 0.5

                if node_type == "ooo":
                    size = [0.9, 0.9, 0.9]
                    edge_col = "white"

                render_block(
                    ax,
                    node_id,
                    node_coords,
                    size,
                    node_type[:3],
                    node_hex_map,
                    alpha=alpha,
                    edge_col=edge_col,
                    border_width=0.5,
                    taken=taken,
                )

    # Edges (valid paths)
    for u, v in graph.edges():
        u_coords = np.array(node_positions.get(u))
        v_coords = np.array(node_positions.get(v))
        if u_coords is not None and v_coords is not None:
            midpoint = (u_coords + v_coords) / 2
            delta = v_coords - u_coords
            original_length = np.linalg.norm(delta)
            adjusted_length = original_length - 1.0

            if adjusted_length > 0:
                orientation = np.argmax(np.abs(delta))
                size = [1.0, 1.0, 1.0]
                size[orientation] = float(adjusted_length)

                pipe_type = edge_types.get((u, v), "gray")
                face_cols = ["gray"] * 6
                
                if pipe_type:
                    alpha = 0.5
                    edge_col = "white" if "*" in pipe_type else "black"

                    col = node_hex_map.get(pipe_type.replace("*", ""), ["gray"] * 3)
                    face_cols = [col[2]] * 2 + [col[1]] * 2 + [col[0]] * 2

                    # Hadamard rendering logic
                    if "h" in pipe_type:
                        if adjusted_length > 0:
                            yellow_length = 0.1 * adjusted_length
                            colored_length = 0.45 * adjusted_length

                            if colored_length < 0 or yellow_length < 0:
                                continue

                            size_col = [1.0, 1.0, 1.0]
                            size_yellow = [1.0, 1.0, 1.0]
                            size_col[orientation] = float(colored_length)
                            size_yellow[orientation] = float(yellow_length)

                            offset1 = np.zeros(3)
                            offset3 = np.zeros(3)

                            offset1[orientation] = -(
                                yellow_length / 2 + colored_length / 2
                            )
                            offset3[orientation] = (
                                yellow_length / 2 + colored_length / 2
                            )

                            centre1 = midpoint + offset1
                            centre2 = midpoint
                            centre3 = midpoint + offset3

                            face_cols_1 = list(face_cols)
                            face_cols_yellow = ["yellow"] * 6
                            face_cols_2 = ["gray"] * 6
                            rotated_pipe_type = rot_o_kind(pipe_type[:3]) + "h"
                            col = node_hex_map.get(rotated_pipe_type, ["gray"] * 3)
                            face_cols_2[4] = col[0] 
                            face_cols_2[5] = col[0] 
                            face_cols_2[2] = col[1] 
                            face_cols_2[3] = col[1] 
                            face_cols_2[0] = col[2] 
                            face_cols_2[1] = col[2] 

                            render_edge(ax, centre1, size_col, face_cols_1, edge_col, alpha, border_width=0.5)
                            render_edge(ax, centre2, size_yellow, face_cols_yellow, edge_col, alpha, border_width=0.5)
                            render_edge(ax, centre3, size_col, face_cols_2, edge_col, alpha, border_width=0.5)
                    else:
                        render_edge(
                            ax,
                            midpoint,
                            size,
                            face_cols,
                            edge_col,
                            alpha,
                            border_width=0.5,
                        )
        
    # 3. Plot adjustments
    # ---
    # Adjust plot for viewing (Only consider static elements for fixed bounds)
    all_static_coords = np.array(taken + tent_coords + [src_block_info[0]]) # Include source
    
    # Calculate bounds based only on static and valid path elements
    if valid_paths:
        for path in valid_paths.values():
            all_static_coords = np.vstack([all_static_coords, np.array([block[0] for block in path])])

    if all_static_coords.size > 0:
        max_x, min_x = all_static_coords[:, 0].max(), all_static_coords[:, 0].min()
        max_y, min_y = all_static_coords[:, 1].max(), all_static_coords[:, 1].min()
        max_z, min_z = all_static_coords[:, 2].max(), all_static_coords[:, 2].min()

        max_range = max((max_x - min_x), (max_y - min_y), (max_z - min_z)) / 2.0
        mid = np.array([(max_x + min_x)/2, (max_y + min_y)/2, (max_z + min_z)/2])

        ax.set_xlim(mid[0] - max_range - 1, mid[0] + max_range + 1)
        ax.set_ylim(mid[1] - max_range - 1, mid[1] + max_range + 1)
        ax.set_zlim(mid[2] - max_range - 1, mid[2] + max_range + 1)

    else:
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([-5, 5])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # 4. Animation function and execution
    # ---
    # ---
    
    # Initialize an empty list to hold all the path artists for blitting
    def update(frame):
        """Adds a new path to the visualization for each frame, keeping previous paths visible."""
        
        # Clear the previous set of lines before redrawing the cumulative history
        # Note: We must clear the AXES lines, not the path_artists list itself.
        # This is a common pattern for cumulative Matplotlib animations.
        while len(ax.lines) > 1: # Keep the first line (static boundary, if any), but clear dynamic lines
             ax.lines[-1].remove()

        artists_to_return = []
        
        # Draw all paths from index 0 up to the current frame (exclusive)
        for i in range(frame):
            if i < num_paths: 
                path_coords = path_frames_data[i]
                color = path_colors[i]
                
                # Create a NEW Line3D artist for the current path
                path_artist, = ax.plot(
                    path_coords[:, 0], 
                    path_coords[:, 1], 
                    path_coords[:, 2], 
                    color=color, 
                    linestyle="--",
                    linewidth=2,
                    zorder=8
                )
                artists_to_return.append(path_artist)

        return artists_to_return

    # Create and run the animation
    if vis_discarded_paths:
        _ = animation.FuncAnimation(
            fig, 
            update, 
            frames=num_frames,
            interval=animation_interval_ms,
            blit=False,
            repeat=False  # Stops the animation after one cycle
        )

    # Show visualisation
    plt.show()



def edge_paths_to_g(edge_paths: dict[StandardBlock, List[StandardBlock]]) -> nx.Graph:
    """Converts an edge_paths object into an nx.Graph that can be visualised with `vis_3d_g`. It is worth noting
    that the function will create a graph with potentially redundant blocks, which is irrelevant for visualisation purposes
    but does mean the function should not be used when producing final results.

    Args:
        - edge_paths: a dictionary containing a number of edge paths, i.e., full paths between two blocks, each path made of 3D blocks and pipes.

    Returns:
        - final_graph: an nx.Graph with all the information in edge_paths but in a format more amicable for visualisation

    """

    final_graph = nx.Graph()
    node_counter = 0

    for _, path_data in edge_paths.items():

        primary_node_and_edges = []
        path_nodes = path_data
        if path_nodes == "error":
            continue
        node_index_map = {}

        for coords, kind in path_nodes:
            if (coords, kind) not in node_index_map:
                node_index_map[(coords, kind)] = node_counter
                primary_node_and_edges.append([node_counter, coords, kind])
                node_counter += 1
            else:

                index_to_use = node_index_map[(coords, kind)]

                found = False
                for entry in primary_node_and_edges:
                    if entry[0] == index_to_use:
                        entry[1] = coords
                        found = True
                        break
                if not found:
                    primary_node_and_edges.append([index_to_use, coords, kind])

        # Add nodes
        for index, coords, node_type in primary_node_and_edges:
            if index not in final_graph:
                final_graph.add_node(index, coords=coords, type=node_type)

        # Add edges
        for i in range(len(primary_node_and_edges)):
            index, coords, node_type = primary_node_and_edges[i]
            if "o" in node_type:
                prev_index_path = i - 1
                next_index_path = i + 1
                if 0 <= prev_index_path < len(
                    primary_node_and_edges
                ) and 0 <= next_index_path < len(primary_node_and_edges):
                    prev_node_index = primary_node_and_edges[prev_index_path][0]
                    next_node_index = primary_node_and_edges[next_index_path][0]
                    if (
                        prev_node_index in final_graph
                        and next_node_index in final_graph
                    ):
                        final_graph.add_edge(
                            prev_node_index, next_node_index, pipe_type=node_type
                        )

    return final_graph


#for i, (block_coords, block_kind) in enumerate(edge_path):
#
    #if block_coords not in [tuple(src_coords)]:
        #alpha = 1
        #edge_col = "black"
        #size = [1.0, 1.0, 1.0] if tgt_kind != "ooo" else [0.9, 0.9, 0.9]
#
        ## Cubes
        #if "o" not in block_kind:
            #render_block(
                #ax,
                #current_grapher_id,
                #block_coords,
                #size,
                #block_kind[:3] if tgt_kind != "ooo" else "ooo",
                #node_hex_map,
                #alpha=alpha,
                #edge_col=edge_col,
                #border_width=0.5,
                #taken=taken,
            #)
            #current_grapher_id += 1
#
        ## Pipes
        #else:
            #print("==> now rendering pipe", block_coords, block_kind)
            #src_coords = edge_path[i-1][0]
            #deltas = [3 if d!=0 else 0 for d in (np.array(block_coords) - np.array(src_coords))]
            #tgt_coords = np.array(src_coords) + deltas
            #print(tgt_coords)
#            
            #midpoint = (src_coords + tgt_coords) / 2
            #delta = tgt_coords - src_coords
            #adjusted_length = 2
            #if adjusted_length > 0:
                #orientation = np.argmax(np.abs(delta))
                #size = [1.0, 1.0, 1.0]
                #size[orientation] = float(adjusted_length)
                #col = node_hex_map[block_kind]
                #face_cols = [col[2]] * 2 + [col[1]] * 2 + [col[0]] * 2
                #if "h" in block_kind:
                    ## Hadamards split into three: two coloured ends and a yellow ring at the middle
                    #if adjusted_length > 0:
                        #yellow_length = 0.1 * adjusted_length
                        #colored_length = 0.45 * adjusted_length
                        ## Skip if lengths are invalid
                        #if colored_length < 0 or yellow_length < 0:
                            #continue
                        #size_col = [1.0, 1.0, 1.0]
                        #size_yellow = [1.0, 1.0, 1.0]
                        #size_col[orientation] = float(colored_length)
                        #size_yellow[orientation] = float(yellow_length)
                        #offset1 = np.zeros(3)
                        #offset3 = np.zeros(3)
                        #offset1[orientation] = -(
                            #yellow_length / 2 + colored_length / 2
                        #)
                        #offset3[orientation] = (
                            #yellow_length / 2 + colored_length / 2
                        #)
                        #centre1 = midpoint + offset1
                        #centre2 = midpoint
                        #centre3 = midpoint + offset3
                        ## Base of hadamard
                        #face_cols_1 = list(face_cols)
                        ## Middle yellow ring
                        #face_cols_yellow = node_hex_map["hadamard"] * 6
                        ## Far end of the hadamard
                        ## Note. Keeping track of the correct rotations proved tricky
                        ## Keep this bit spread out across lines – easier
                        #face_cols_2 = ["gray"] * 6
                        #rotated_pipe_type = rot_o_kind(block_kind[:3]) + "h"
                        #col = node_hex_map[rotated_pipe_type]
                        #col = node_hex_map[block_kind[:3]]
                        #face_cols_2[4] = col[0]  # right (+x)
                        #face_cols_2[5] = col[0]  # left (-x)
                        #face_cols_2[2] = col[1]  # front (-y)
                        #face_cols_2[3] = col[1]  # back (+y)
                        #face_cols_2[0] = col[2]  # bottom (-z)
                        #face_cols_2[1] = col[2]  # top (+z)
                        #render_edge(
                            #ax,
                            #centre1,
                            #size_col,
                            #face_cols_1,
                            #edge_col,
                            #alpha,
                            #border_width=0.5,
                        #)
                        #render_edge(
                            #ax,
                            #centre2,
                            #size_yellow,
                            #face_cols_yellow,
                            #edge_col,
                            #alpha,
                            #border_width=0.5,
                        #)
                        #render_edge(
                            #ax,
                            #centre3,
                            #size_col,
                            #face_cols_2,
                            #edge_col,
                            #alpha,
                            #border_width=0.5,
                        #)
                #else:
                    #render_edge(
                        #ax,
                        #midpoint,
                        #size,
                        #face_cols,
                        #edge_col,
                        #alpha,
                        #border_width=0.5,
                    #)