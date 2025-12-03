> NB!!! I have just merged a major folder re-structuring PR aimed at facilitating multiple contributors working in parallel across different aspects of Topologiq. I tested things, of course, and mmost things should work just fine. However, if something breaks, go back one merge and try again if you're looking to test Topologiq or open an Issue about and suggest a fix if you're looking to contribute. 

# Topologiq: Algorithmic Lattice Surgery
**Topologiq** is tool to convert ZX circuits into logical versions of themselves. It is based on the surface code and lattice surgery.

Interoperable with:
- PyZX: [click here for an example](docs/examples/pyzx_cnots.ipynb).
- Qiskit: [click here for an example](docs/examples/qiskit_ghz.py).

<br>

> **Work in progress**. Check "main" for latest stable checkpoint. Venture into open branches for the latest updates.

## Summary
As visualised in the animated GIF below, **Topologiq** uses the connectivity information in a ZX-graph to produce a topologically-correct lattice surgery / space-time diagram. It picks a starting point randomly based on a centrality measure. It builds incrementally from that starting point, one edge at a time.

**Topologiq**'s outputs can be used as inputs for TQEC/tqec and, *theoretically*, other similar tools.

<br>

![Algorithmic lattice surgery of a CNOT](./docs/media/cnots.gif)

*Figure 1. Algorithmic lattice surgery of a CNOT.*

## Install
Currently, the best way to test **Topologiq** is to clone the repository, recreate the environment, and install dependencies.

**Step 1.** Clone. 
```bash
git clone https://github.com/jbolns/topologiq.git
```

**Step 2.** Recreate environment.

Choose between an UV or a pip installation (one or the other, not both).

*Using UV.*
```bash
# 1. Create & Sync
uv sync
```

*Using pip.*
```bash
# 1. Create environment
python -m venv .venv

# 2. Activate environment
.venv\Scripts\activate.bat  # Windows
source .venv/bin/activate  # Linux

# 3. Install dependencies
pip install -r requirements.txt
```

**For contributors.** Contributors should opt for an editable installation.

```bash
# After running steps 1 of UV method or steps 1 & 2 of pip method
uv pip install -e

# OR (and never both)
pip install -e
```

## Examples
For examples, run any of the commands below from the root of the repository. The algorithm will run and stop when it finds a succesfull solution or run up to ten times.

A succesfull result will produce a TXT file with information about the initial ZX graph, intermediate state, and final result (saves to `./src/topologiq/outputs/txt/`) (all information printed to this TXT file is also available for programmatic use). There are also optional parameters to trigger a variety of visualisations and summary animations.

``` bash
# A CNOT, using PyZX.
uv run src/topologiq/run.py --pyzx:cnot --vis:final
python3 src/topologiq/run.py --pyzx:cnot --vis:final  # Requires active .venv

# Random series of CNOTs, using PyZX.
uv run src/topologiq/run.py --pyzx:cnots --vis:final
python3 src/topologiq/run.py --pyzx:cnots --vis:final  # Requires active .venv

# A medium-size circuit with three interconnected lines, using PyZX.
uv run src/topologiq/run.py --pyzx:simple_mess --vis:final
python3 src/topologiq/run.py --pyzx:simple_mess --vis:final  # Requires active .venv

# Line of hadamards, using a non-descript ZX-graph.
uv run src/topologiq/run.py --graph:hadamard_line --vis:final
python3 src/topologiq/run.py --graph:hadamard_line --vis:final  # Requires active .venv

# Circuit with Hadamards on bends, using a non-descript ZX-graph.
uv run src/topologiq/run.py --graph:hadamard_bend --vis:final
python3 src/topologiq/run.py --graph:hadamard_bend --vis:final  # Requires active .venv

# A 7-qubit Steane code, using a non-descript ZX-graph. 
uv run src/topologiq/run.py --graph:steane --vis:final
python3 src/topologiq/run.py --graph:steane --vis:final  # Requires active .venv

# A mess of hadamards, using a non-descript ZX-graph.
uv run src/topologiq/run.py --graph:hadamard_mess --vis:final
python3 src/topologiq/run.py --graph:hadamard_mess --vis:final  # Requires active .venv
```

There are also additional options that can be appended to any command for debugging and statistical purposes.

``` bash
# Run a circuit normally and log stats for all attempts to complete the specific circuit.
uv run src/topologiq/run.py --pyzx:simple_mess --log_stats
python3 src/topologiq/run.py --pyzx:simple_mess --log_stats  # Requires active .venv

# Run a specific circuit a single time irrespective of outcome.
uv run src/topologiq/run.py --graph:hadamard_mess --repeat:1
python3 src/topologiq/run.py --graph:steane --repeat:1  # Requires active .venv

# Run a circuit a given number of times and log log stats for all 50 cycles.
uv run src/topologiq/run.py --pyzx:cnots --repeat:50
python3 src/topologiq/run.py --pyzx:cnots --repeat:50  # Requires active .venv

# Turn debug mode on (valid modes: 1, 2, 3) (adds incrementally detailed logs and visualisations).
uv run src/topologiq/run.py --pyzx:cnots --vis:detail --debug:1
python3 src/topologiq/run.py --pyzx:cnots --vis:detail --debug:1  # Requires active .venv

# Pick up edge cases and prompt user to run specific edge case (output logs must exist).
uv run src/topologiq/debug.py
python3 src/topologiq/debug.py  # Requires active .venv
```

And it is possible to set several visualisation options also via command.

``` bash
# No visualisations
uv run src/topologiq/run.py --pyzx:cnot
python3 src/topologiq/run.py --pyzx:cnot  # Requires active .venv

# Final outcome visualised.
uv run src/topologiq/run.py --pyzx:cnots --vis:final
python3 src/topologiq/run.py --pyzx:cnots --vis:final  # Requires active .venv

# Each edge-placement is visualised / a series of progress visualisations.
uv run src/topologiq/run.py --pyzx:cnots --vis:detail
python3 src/topologiq/run.py --pyzx:cnots --vis:detail  # Requires active .venv

# "BOUNDARIES" stripped prior performing any operations and therefore not considered.
uv run src/topologiq/run.py --pyzx:cnot --vis:final --strip_boundaries
python3 src/topologiq/run.py --pyzx:cnot --vis:final --strip_boundaries  # Requires active .venv

# A GIF or MP4 summary animation of the process is saved to `/outputs/media`.
uv run src/topologiq/run.py --pyzx:cnots --animate:GIF
uv run src/topologiq/run.py --pyzx:cnots --animate:MP4
python3 src/topologiq/run.py --pyzx:cnots --animate:GIF  # Requires active .venv
python3 src/topologiq/run.py --pyzx:cnots --animate:MP4  # Requires active .venv. Requires FFmpeg.
```

## Yeah, but how does it work, really?
A detailed insight into the algorithm and, hopefully, a paper, is in progress. Meanwhile, below, a quick overview of the inner workings of the algorithm. 

**In first place,** the algorithm will look for an incoming ZX graph and, if needed and possible, convert it into a native format.
- ***Native format:*** A simple dictionary of nodes and edges (see `./src/topologiq/assets/graphs/simple_graphs.py` for examples).
- ***PyZX interoperability:*** PyZX graphs supported (check `run.py` for a blueprint of the conversion process and `./src/topologiq/assets/graphs/pyzx_graphs.py` for examples).
  - Note. If using a random PyZX circuit for testing, ensure all qubit lines are interconnected. If a qubit line is not interconnected, the graph has subgraphs. The algorithm treats subgraphs as separate logical computations, and will focus on one subgraph only.

**After,** the algorithm will traverse the ZX graph transforming each node into an equivalent "primitive" and position it in a way that honours the original graph. This second part of the process is itself divided into several stages:
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

## Contributing
Pull requests and issues are more than welcomed!

See [CONTRIBUTING](./CONTRIBUTING.md) for specific instructions to start contributing.

## License
Topologiq is licensed under an [Apache 2.0 license](./LICENSE).

The [`ETHICAL_NOTICE.md`](ETHICAL_NOTICE.md) contains additional **ethical use** pointers.

## Community
Every Wednesday at 8:30am PST, we hold [meetings](https://meet.jit.si/TQEC-design-automation) to discuss project progress and conduct educational talks related to TQEC.

Here are some helpful links to learn more about the TQEC community and Topologiq:
- Overview of state of the art 2D QEC: [Slides](https://docs.google.com/presentation/d/1xYBfkVMpA1YEVhpgTZpKvY8zeOO1VyHmRWvx_kDJEU8/edit?usp=sharing)/[Video](https://www.youtube.com/watch?v=aUtH7wdwBAM&t=2s)
- Introduction to surface code quantum computation: [Slides](https://docs.google.com/presentation/d/1GxGD9kzDYJA6X47BXGII2qjDVVoub5BsSVrGHRZINO4/edit?usp=sharing)
- Programming a quantum computer using SketchUp: [Slides](https://docs.google.com/presentation/d/1MjFuODipnmF-jDstEnQrqbsOtbSKZyPsuTOMo8wpSJc/edit?usp=sharing)/[Video](https://drive.google.com/file/d/1o1LMiidtYDcVoEFZXsJPb7XdTkZ83VFX/view?usp=drive_link)
- Overview of Topologiq: [Video](https://drive.google.com/file/d/1C9Kke4qSYd0lX5qO_yvUX88DsPt8kyaP/view?usp=drive_link)
- Qiskit->QASM->Topologiq interoperability: [Video](https://drive.google.com/file/d/1tFYNmvvyNDT04BK6U3ESRZVXB1PrObGd/view).

All the resources and group meeting recordings are available at [this link](https://docs.google.com/spreadsheets/d/11DSA2wzKLOrfTGNHunFvzsMYeO7jZ8Ny8kpzoC_wKQg/edit?usp=sharing&resourcekey=0-PdGFkp5s-4XWihMSxk0UIg).

Please join the [Google group](https://groups.google.com/g/tqec-design-automation) to receive more updates and information!
