"""Example of how to undertake post-processing transformations on a simple BlockGraph."""

from topologiq.assets.pyzx_graphs import ht
from topologiq.core.graph_manager.graph_manager import BlockGraphManager
from topologiq.input.zx_manager import ZXGraphManager

##########
# KWARGS #
##########
kwargs = {
    "debug": 0,  # Verbosity & visualisation control [int]
    "first_id_strategy": "first-spider",  # Strategy for choosing the first spider/cube ID [str]
    "graph_traverse_mode": "bfs-layers",  # Graph traversing strategy [str]
    # "gravity": 7,  # Pull paths towards graph centre [int]
    "post_process": False,
    # "z_stretch": 1,
}

#######
# RUN #
#######
if __name__ == "__main__":
    # Give pretty name to graph
    graph_name = "ht"
    num_t = 10
    print(f"\n#####################\nGRAPH NAME: {graph_name} * {num_t}. \n#####################\n")

    # Get graph
    pyzx_graph = ht(draw_graph=False, num_t=num_t)

    # PyZX -> AugmentedZXGraph
    zx_graph_manager = ZXGraphManager(debug=kwargs["debug"])
    aug_zx = zx_graph_manager.add_graph_from_pyzx(pyzx_graph, graph_key="input")

    # AugmentedZXGraph -> BlockGraph
    bgraph_manager = BlockGraphManager(aug_zx, **kwargs)
    bgraph_manager.build()

    # Visualise results
    bgraph_manager.draw_blockgraph()

    # Write results to file
    bgraph_manager.write_bgraph(circuit_name=graph_name)

    # Extend MSCs to maximum available space
    if kwargs["post_process"]:
        bgraph_manager.stretch_msc_cubes()
        bgraph_manager.draw_blockgraph()

    # Add micro-factories where possible
    if kwargs["post_process"]:
        bgraph_manager.distributed_msc_factory()
        bgraph_manager.draw_blockgraph()

    # Example of how to exchange a MSC for a factory
    # Note. Assumes the existence of a control system that
    # would flag a factory success near a MSC that hasn't succeeded
    if kwargs["post_process"]:
        bgraph_manager.msc_exchange(connect_id=50, remove_id=26)
        bgraph_manager.draw_blockgraph()

    # Example of how to stretch the computation to wait for an MSC
    if kwargs["post_process"]:
        bgraph_manager.slice_stretch(slice_at_z=3, shift_z=5)
        bgraph_manager.draw_blockgraph()

    # Example of how to stretch the computation to wait for a conditional
    if kwargs["post_process"]:
        bgraph_manager.slice_stretch(slice_at_z=11, shift_z=1)
        bgraph_manager.draw_blockgraph()

    # Animate and clean up
    if kwargs.get("animate"):
        bgraph_manager.animate(filename_prefix=graph_name)

    # Say good bye
    print("\n---\nThank you for flying Topologiq.\n")
