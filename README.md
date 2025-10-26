# Topologiq: Algorithmic Lattice Surgery
***Topologiq*** is tool to convert ZX circuits into logical versions of themselves. It is based on the surface code and lattice surgery.

In essence, as illustrated in the GIF animation below (more examples [here](./src/topologiq/assets/media/)), ***topologiq*** uses the connectivity information in a ZX-graph to produce a topologically-correct lattice surgery / space-time diagram that can more easily be consumed by other quantum error correction (QEC) software/tools.

![Algorithmic lattice surgery of a CNOT](./src/topologiq/assets/media/cnots.gif)

*Figure 1. Algorithmic lattice surgery of a CNOT.*

> ***Note.*** Work in progress. Check "main" for latest stable checkpoint, or venture into any other open branches for the latest updates.

> ***Note.*** ***Topologiq*** is compatible with [PyZX](https://github.com/zxcalc/pyzx) (incl. [phases & T-gates](./src/topologiq/assets/notebooks/pyzx_phases_t_gates.ipynb), [Pauli webs](./src/topologiq/assets/notebooks/pyzx_pauli_webs.ipynb), and [QASM interop](./src/topologiq/assets/notebooks/qasm_via_pyzx.ipynb)) and, theoretically, any other ZX tool able to produce a similar ZX graphs.

## Background
ZX-calculus<sup>[1-7]</sup> is a helpful and intuitive way to represent and manipulate design quantum circuits. Virtues notwithstanding, ZX-circuits/graphs are not immediately amicable to QEC. Barring unexpected developments on the hardware front, there is a need to convert them into logical computations resilient to errors.

A leading approach to building logical quantum computations that are seemingly resilient to errors is the surface code,<sup>[8-14]</sup> a planar entanglement of qubit operations that join many qubits into a single logical computation. Lattice surgery<sup>[15-23]</sup> is the process of merging and splitting surface code patches to create continuous logical computations, often visualised as space-time diagrams like Figure 1.

Researchers have found a number of basic "primitive" lattice surgery operations that can be combined to form logical computations.<sup>[18-20, 24-27]</sup> The blocks have been validated as valid instances of surface code operations in an ongoing open-source effort to develop “automation software for representing, constructing and compiling large-scale fault-tolerant quantum computations based on surface code and lattice surgery”.<sup>[28]</sup>

When you name these "primitives" by reference to both the underlying topological features of the surface code *and* the coordinate space, the names become "symbolic" in a very mathematical sense:
- The names can be manipulated using symbolic operations
- The names can be used to establish potential placements for linked operations
- The names can be used to ensure the outcome of a lattice surgery is topologically correct.

***Topologiq*** uses the topological properties of these "primitives" to traverse a ZX graph, place its nodes in a 3D space, and devise the operations needed to deliver a topologically-correct space-time diagram.

## Install
Currently, the best way to test ***topologiq*** is to clone the repository, recreate the environment, and install dependencies.

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
uv run src/topologiq/run.py --graph:hadamards_mess --vis:final
python3 src/topologiq/run.py --graph:hadamards_mess --vis:final  # Requires active .venv
```

There are also additional options that can be appended to any command for debugging and statistical purposes.

``` bash
# Run a circuit normally and log stats for all attempts to complete the specific circuit.
uv run src/topologiq/run.py --pyzx:cnots --log_stats
python3 src/topologiq/run.py --pyzx:cnots --log_stats  # Requires active .venv

# Run a specific circuit a single time irrespective of outcome.
uv run src/topologiq/run.py --graph:steane --repeat:1
python3 src/topologiq/run.py --graph:steane --repeat:1  # Requires active .venv

# Run a circuit a given number of times and log log stats for all 50 cycles.
uv run src/topologiq/run.py --pyzx:cnots --repeat:50
python3 src/topologiq/run.py --pyzx:cnots --repeat:50  # Requires active .venv

# Run topologiq in full debug mode (output logs must exist).
uv run src/topologiq/debug.py
python3 src/topologiq/debug.py  # Requires active .venv
```

And it is possible to set several visualisation options also via command.

``` bash
# No visualisations (best for programmatic use)
uv run src/topologiq/run.py --pyzx:cnot
python3 src/topologiq/run.py --pyzx:cnot  # Requires active .venv

# Final outcome visualised.
uv run src/topologiq/run.py --pyzx:cnots --vis:final
python3 src/topologiq/run.py --pyzx:cnots --vis:final  # Requires active .venv

# Each edge-placement is visualised / a series of progress visualisations.
uv run src/topologiq/run.py --pyzx:cnots --vis:detail
python3 src/topologiq/run.py --pyzx:cnots --vis:detail  # Requires active .venv

# "BOUNDARIES" considered by algorithm but NOT shown in visualisations.
uv run src/topologiq/run.py --pyzx:cnot --vis:final --hide_boundaries
python3 src/topologiq/run.py --pyzx:cnot --vis:final --hide_boundaries  # Requires active .venv

# "BOUNDARIES" stripped prior performing any operations and therefore not considered.
uv run src/topologiq/run.py --pyzx:cnot --vis:final --strip_boundaries
python3 src/topologiq/run.py --pyzx:cnot --vis:final --strip_boundaries  # Requires active .venv

# Run visualisations on debug mode (additional details shown) (only helpful if combined with detail visualisations or animations).
uv run src/topologiq/run.py --pyzx:cnots --vis:detail --debug
python3 src/topologiq/run.py --pyzx:cnots --vis:detail --debug  # Requires active .venv

# A GIF or MP4 summary animation of the process is saved to `/outputs/media`.
uv run src/topologiq/run.py --pyzx:cnots --animate:GIF
uv run src/topologiq/run.py --pyzx:cnots --animate:MP4
python3 src/topologiq/run.py --pyzx:cnots --animate:GIF  # Requires active .venv
python3 src/topologiq/run.py --pyzx:cnots --animate:MP4  # Requires active .venv. Requires FFmpeg.
```

## Use your own circuits
It would be great to hear of tests using other circuits.
- You can use a non-descript ZX graph defined as a dictionary of nodes and edges. See `./src/topologiq/assets/graphs/simple_graphs.py` for examples.
  - No visualisation of input graphs currently available.
- You can also use a PyZX graph. See check `run.py` for a blueprint of the process needed and `./src/topologiq/assets/graphs/pyzx_graphs.py` for examples of graphs.
  - Feed PyZX's Matplotlib figure (`fig = zx.draw_matplotlib(...)`) to the algorithm using the `fig_data` optional parameter of `./src/topologiq/scripts/runner.py/`'s `runner()` function to get the PyZX graph overlayed over final output visualisations.

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
Everything is pending, but below a list of highest priorities:
- Comprehensive testing, including and starting by a way to easily launch the algorithm for any specific circuit using a pre-specified node processing orders (needed for testing and debugging edge cases). 
- Up-close table and visualisations of a single pathfinding iteration.
- Improve PyZX support.
- Better documentation.
- More example scripts (no more notebooks, there's too many already), especially for integration of external systems.
- Improve run-times further, especially for the inner pathfinder algorithm when two pre-existing nodes are very far from one another.

Having said that, at the moment, a pre-requisite to accepting contributions is to develop all the automated workflows needed to manage contributions robustly. To avoid accidents on "main", other contributions are unlikely to be accepted before these workflows are available. It would be awesome if someone wants to contribute the workflows. Else, they'll be ready when they're ready. 

## License
This repository is open source software. All code in the repository is under an Apache 2.0 license.

## References
1. Coecke, B. & Duncan, R. Interacting Quantum Observables. In *Automata, Languages and Programming* (eds. Aceto, L. et al.) 298–310 (Springer, Berlin, Heidelberg, 2008).  
2. Duncan, R. & Perdrix, S. Graph States and the Necessity of Euler Decomposition. In *Mathematical Theory and Computational Practice* (eds. Ambos-Spies, K., Löwe, B. & Merkle, W.) 167–177 (Springer, Berlin, Heidelberg, 2009).  
3. Coecke, B. & Duncan, R. Interacting quantum observables: categorical algebra and diagrammatics. *New J. Phys*. 13, 043016 (2011).  
4. Backens, M. The ZX-calculus is complete for stabilizer quantum mechanics. *New J. Phys*. 16, 093021 (2014).  
5. Backens, M. Making the stabilizer ZX-calculus complete for scalars. *Electron. Proc. Theor. Comput. Sci*. 195, 17–32 (2015).  
6. Wetering, J. van de. ZX-calculus for the working quantum computer scientist. PrePrint (2020).  
7. Kissinger, A. & Wetering, J. van de. Universal MBQC with generalised parity-phase interactions and Pauli measurements. *Quantum* 3, 134 (2019).  
8. Kitaev, A. Yu. Quantum Error Correction with Imperfect Gates. In *Quantum Communication, Computing, and Measurement* (eds. Hirota, O., Holevo, A. S. & Caves, C. M.) 181–188 (Springer US, Boston, MA, 1997).  
9. Kitaev, A. Y. Fault-tolerant quantum computation by anyons. *Ann. Phys*. 303, 2–30 (2003).  
10. Bravyi, S. B. & Kitaev, A. Y. Quantum codes on a lattice with boundary. Preprint (1998).  
11. Dennis, E., Kitaev, A., Landahl, A. & Preskill, J. Topological quantum memory. *J. Math. Phys*. 43, 4452–4505 (2002).  
12. Fowler, A. G., Stephens, A. M. & Groszkowski, P. High threshold universal quantum computation on the surface code. *Phys. Rev*. A 80, 052312 (2009).  
13. Fowler, A. G., Mariantoni, M., Martinis, J. M. & Cleland, A. N. Surface codes: Towards practical large-scale quantum computation. *Phys. Rev*. A 86, 032324 (2012).  
14. Acharya, R. et al. Quantum error correction below the surface code threshold. *Nature* 638, 920–926 (2025).  
15. Horsman, D., Fowler, A. G., Devitt, S. & Meter, R. V. Surface code quantum computing by lattice surgery. *New J. Phys*. 14, 123011 (2012).  
16. Litinski, D. & Oppen, F. von. Lattice Surgery with a Twist: Simplifying Clifford Gates of Surface Codes. *Quantum* 2, 62 (2018).  
17. Landahl, A. J. & Ryan-Anderson, C. Quantum computing by color-code lattice surgery. Preprint (2014).  
18. Fowler, A. G. & Gidney, C. Low overhead quantum computation using lattice surgery. Preprint (2019).  
19. Gidney, C. & Fowler, A. G. Efficient magic state factories with a catalyzed |CCZ〉-> |T〉transformation. *Quantum* 3, 135 (2019).  
20. Gidney, C. & Fowler, A. G. Flexible layout of surface code computations using AutoCCZ states. Preprint (2019).  
21. Tan, D. B., Niu, M. Y. & Gidney, C. A SAT Scalpel for Lattice Surgery: Representation and Synthesis of Subroutines for Surface-Code Fault-Tolerant Quantum Computing. In *2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)* 325–339 (2024).  
22. Shaw, A. T. E., Bremner, M. J., Paler, A., Herr, D. & Devitt, S. J. Quantum computation on a 19-qubit wide 2d nearest neighbour qubit array. Preprint (2022).  
23. Gehér, G. P., McLauchlan, C., Campbell, E. T., Moylett, A. E. & Crawford, O. Error-corrected Hadamard gate simulated at the circuit level. *Quantum* 8, 1394 (2024).  
24. Paetznick, A. & Fowler, A. G. Quantum circuit optimization by topological compaction in the surface code. Preprint (2013).  
25. Paler, A., Devitt, S. J. & Fowler, A. G. Synthesis of Arbitrary Quantum Circuits to Topological Assembly. *Sci. Rep*. 6, 30600 (2016).  
26. Fowler, A. G. Computing with fewer qubits: Pitfalls and tools to keep you safe [Conference Presentation]. *Munich Quantum Software Forum* (2023).  
27. Fowler, A. G. Programming a quantum computer using SketchUp [Conference Presentation]. *Munich Quantum Software Forum* (2024). 
28. TQEC Community. TQEC (Topological Quantum Error Correction): Design Automation Software Tools for Topological Quantum Error Correction. https://github.com/tqec/tqec (2025).