# Algorithmic Lattice Surgery
A family of algorithms (ok, there's currently *one* full algorithm, but it is a foundation to make a family of them) to convert ZX circuits into logical versions of themselves.

It produces logical computations like the one below (video examples available in `./assets/media`).

![GIF animation of an example using Steane code](./assets/media/steane.gif)

*Figure 1. Example output (Stene code).*

***Note.*** This is work in progress.

***Note.*** Hadamards are NOT supported yet.

***Note.*** A better README is on the way.

## Process
Several things happen when you run the algorithm.

**In first place,** the algorithm will look for an incoming ZX graph and, if needed and possible, convert it into a native format.
- ***Native format:*** A simple dictionary of nodes and edges (see `assets/graphs/simple_graphs.py` for examples).
- ***PyZX interoperability:*** PyZX graphs supported (check `run.py` for a blueprint of the conversion process and `assets/graphs/pyzx_graphs.py` for examples).

**After,** the algorithm will traverse the ZX graph transforming each node into a 3D equivalent "block" and positioning the resulting blocks in a way that honours the original edges in the graph (may involve a need to add intermediate blocks). This second part of the process is itself divided into several stages:
- ***Positioning:*** organises the process of placing each node into a number of tentative positions.
  - Step currently follows a greedy Breadth First Search (BFS) approach.
  - Additional strategies will be explored in due course. 
- ***Pathfinding:*** explores a 4D space (x, y, z, block type) to determine which tentative positions allow topologically-correct paths.
  - Step currently uses a slightly-modified Dijkstra.
  - Additional strategies may be explored in due course mainly because Dijkstra is an inherently-slow algorithm.
  - That said, the priority is to optimise the existing approach to ensure maximal robustness.
- ***Value function:*** Chooses best path from the pathfinding algorithm based on given hyperparameters attached to the positioning algorithm.
  - Hyperparameters currently set to values that increase the odds of finding a successful solution in test runs. 
  - This does **NOT** mean the algorithm will produce optimal results with current hyperparameters.
  - An automated approach to discovering optimal hyperparameters will eventually be added, but this is not currently available.
  - To vary hyperparameters manually, edit `run_hyper_params.py`.

## Examples
For examples of what the algorithms can currently do, run any of the following commands from the root of the repository. The algorithm will stop when it finds a succesfull solution or run up to ten times.

``` bash
# A CNOT, using PyZX.
python -m run --pyzx:cnot

# Random series of CNOTs, using PyZX: CNOT_HAD_PHASE_circuit().
python -m run --pyzx:cnots

# Random circuit, using PyZX: zx.generate.cliffordT().
python -m run --pyzx:random

# Random circuit, optimised, using PyZX: zx.generate.cliffordT(), zx.simplify.phase_free_simp().
python -m run --pyzx:random_optimised

# A 7-qubit Steane code, from a non-descript graph. 
# Ps. This one takes longest. It is a tightly-packed/highly-optimised circuit, so a few rounds are often needed to find a successful solution.
python -m run --graph:steane

```

There are also several visualisation options that can be appended to any command.

``` bash

# "BOUNDARIES" considered and visualised.
python -m run --pyzx:cnot

# "BOUNDARIES" considered by algorithm but NOT visible in visualisations.
python -m run --pyzx:cnot --hide_boundaries

# "BOUNDARIES" stripped prior performing any operations.
python -m run --pyzx:cnot --strip_boundaries

```


It would be great to hear of tests using other circuits. You can use a non-descript ZX graph defined as a dictionary of nodes and edges (see `assets/graphs/simple_graphs.py` for examples) or a PyZX graph (check `run.py` for a blueprint of the process needed and `assets/graphs/pyzx_graphs.py` for examples of graphs).

That said, please note a degree of failed results is expected. The current goal is to inform developmental priorities by identifying types of circuits for which the current implementation performs good, less good, bad, and awfully.

## Outputs
A succesfull result will produce:
- a 3D interactive graph (pops up)
- a GIF animation of the process (saves to `./outputs/media/`) (videos possible, but FFmpeg must be installed)
- a TXT file with information about the initial ZX graph, intermediate state, and final result (saves to `./outputs/txt/`).

The information printed to the TXT file is also available for programmatic use.

## Pending
Everything is pending, but below a list of highest priorities:
- Enable automatic selection and variation of hyperparameters
- Add support for Hadamards
- Improve PyZX support
- Improve run-times.
