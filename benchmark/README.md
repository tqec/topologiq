# What is this folder?
This folder contains files and reports which are helpful to:
- understand and improve runtimes
- understand and improve space-time volume performance
- understand edge cases that might helpf in debugging.

## Why are there no data files in the data folder?
The data folder sometimes contains and sometimes does not contain files. It depends on whether there is work ongoing to generate new benchmarks. 

Ideally, one day this will be optimised so that benchmarks are always actualised for every update of the codebase.

If you see no files but still want to play around with some data about Topologiq's performance, give the main `runner.py` function the optional parameter `log_stats: bool = True`. This will trigger the creation of the following data files:
- graph_manager.csv: statistics about the outer BFS cycle,
- pathfinder.csv: statistics about each individual edge path sent to the inner pathfinder algorithm,
- params.csv: outputs and run parameters for specific runs.
- debug.csv: output and run parameters for specific runs that can assist debugging (failed runs or runs using atypical beam lengths or atypical hyperparameters).

Please note. Statistics are NOT logged by default. A regular run will NOT be logged unless `log_stats` is set to `True`.

## Schemas
Schemas for the data files inside the data folder are as follows:

### Graph manager (one log per cycle/circuit run)
- unique_run_id: the unique identifier for the specific run of the full algorithmic flow,
- run_success: whether the process was successful, i.e., it produced a complete lattice surgery/space-time diagram of the original circuit,
- circuit_name: the name of the circuit,
- len_beams: the length of the beams used for the specific run,
- num_input_nodes_processed: the number of nodes in the original/input ZX graph that were processed successfully,
- num_input_edges_processed: the number of edges in the original/input ZX graph that were processed successfully,
- num_1st_pass_edges_processed: the number of normal edges (edges between a "placed" block and a "new" one yet to be assigned coordinates and kind) in the original/input ZX graph that were processed successfully,
- num_2n_pass_edges_processed: the number of special edges (edges between nodes placed as part of the "first pass") in the original/input ZX graph that were processed successfully,
- num_edges_in_edge_paths: the total number of edges in the final set of edges produced by the algorithm (same as `num_input_edges_processed` if `run_success==True`),
- num_blocks_output: the number of 3D spider blocks (cubes) in the output,
- num_edges_output: the number of 3D edge blocks (pipes) in the output,
- duration_first_pass: the total duration of the first pass of the algorithm, i.e., the "while" loop that takes care of "first pass" edges,
- duration_second_pass: the total duration of the second pass of the algorithm, i.e., the "while" loop that takes care of "second pass" edges,
- duration_total: the total duration of the entire algorithmic lattice surgery process for this circuit, irrespective of `run_success`. 

### Pathfinder (one log per iteration)
- unique_run_id: the unique identifier for the specific run of the full algorithmic flow,
- iter_type: the type of operation asked, i.e., whether the pathfinder needs to "create" a path between a block that has already been placed in the 3D space and a new one or whether it needs to "discover" a valid path between two blocks that were already placed as part of previous operations,
- iter_success: whether the pathfinder algorithm was able to find one or more topologically correct path(s) for the specific edge in the iteration.
- src_coords: the coordinates for the source block,
- src_kind: the kind of the source block,
- tgt_coords: if `iter_type` is discovery (target block placed as part of previous operations), the coordinates of the target block, else "TBD",
- tgt_zx_type: the ZX type of the target block, irrespective of `iter_type`,
- tgt_kind: if `iter_type` is discovery (target block placed as part of previous operations), the kind of the target block, else "TBD",
- num_tent_coords_received: the number of tentative coordinates received (many if `iter_type` is creation, one if `iter_type` is discovery),
- num_tent_coords_filled: the number of tentative coordinates that the pathfinder algorithm was able to find topologically correct paths to,
- max_manhattan_src_to_any_tent_coord: the Manhattan distance between the source and the tentative coordinate that is furthest away from it,
- len_longest_path: the lenght of the longest path returned by the algorithm, in grid units counting the lenght of all cubes/pipes in path,
- num_visitation_attempts: the number of sites (coordinate, kind, direction) the pathfinder tested to determine if a new block could be placed there,
- num_sites_visited: the number of times that, having tested a given site, the algorithm "visited" the site,
- iter_duration: the total duration of the specific iteration of the pathfinder algorithm (and to be clear, there will be many iterations of the pathfinder algorithm per circuit run).

### Params (one log per cycle/circuit run)
- unique_run_id: the unique identifier for the specific run of the full algorithmic flow,
- run_success: whether the process was successful, i.e., it produced a complete lattice surgery/space-time diagram of the original circuit,
- circuit_name: the name of the circuit,
- run_params: the main parameters needed to replicate the run.
- edge_paths: a list of the original ZX edges processed successfully in the order in which the algorithm processed them.

### Debug (optional logs as needed)
Same as "params" but only for runs identified as edge cases.
