> NB!!! There have been 3 recent major merges (folder restructuring, dependency clean-up, and Ruff) aimed at facilitating things for potential contributors. All seems to be running in order. However, if something does not work, go back a few merges and try again if you're looking to test Topologiq or open an Issue about and suggest a fix if you're looking to contribute.

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

### Using UV
```bash
# 1. Clone repository. 
git clone https://github.com/jbolns/topologiq.git

# 2. Sync environment.
uv sync

# Additional steps needed only for contributors
# 3. Opt for an editable installation
uv pip install -e .
```

### Using PIP
```bash
# 1. Clone repository. 
git clone https://github.com/jbolns/topologiq.git

# 2. Recreate environment.
# 2.1. Environment creation
python -m venv .venv

# 2.2. Environment activation
.venv\Scripts\activate.bat  # Windows
source .venv/bin/activate  # Linux

# 2.3. Install dependencies
pip install -r requirements.txt

# Additional steps needed only for contributors
# 3. Opt for an editable installation
pip install -e .
```

## Examples
You can quickly test **Topologiq** from the terminal using pre-defined circuit examples.

Check `./outputs/txt/` for results. All information in TXT outputs is also available for programmatic use.

``` bash
# PyZX examples (Available examples: cnot, cnots, simple_mess)
uv run src/topologiq/run.py --pyzx:<circuit_name>
python3 src/topologiq/run.py --pyzx:<circuit_name>  # Requires active .venv

# Dict-based graph examples (Available examples: hadamard_line, hadamard_bend, steane, steane_obfs, or hadamard_mess)
uv run src/topologiq/run.py --graph:<circuit_name>
python3 src/topologiq/run.py --graph:<circuit_name>  # Requires active .venv
```

There are also optional parameters to enable 3D visuals, animations, and debug options.

``` bash
# Enable "final" result or "detail" edge-by-edge progress visualisations
uv run src/topologiq/run.py --pyzx:<circuit_name> --vis:<final|detail>
python3 src/topologiq/run.py --pyzx:<circuit_name> --vis:<final|detail>  # Requires active .venv

# Produce a "GIF" or "MP4" animation of the process (MP4 requires FFmpeg).
uv run src/topologiq/run.py --pyzx:<circuit_name> --animate:<GIF|MP4>
python3 src/topologiq/run.py --pyzx:<circuit_name> --animate:<GIF|MP4>  # Requires active .venv

# Run a circuit a specific number of times irrespective of outcome for each inidividual run
uv run src/topologiq/run.py --pyzx:<circuit_name> --repeat:50
python3 src/topologiq/run.py --pyzx:<circuit_name> --repeat:50  # Requires active .venv

# Log run-time and performance statistics
uv run src/topologiq/run.py --pyzx:<circuit_name> --log_stats
python3 src/topologiq/run.py --pyzx:<circuit_name> --log_stats  # Requires active .venv

# Enable debug mode (incrementally detailed logs and visuals) (available modes: 1, 2, 3, 4) .
uv run src/topologiq/run.py --pyzx:<circuit_name> --vis:detail --debug:1
python3 src/topologiq/run.py --pyzx:<circuit_name> --vis:detail --debug:1  # Requires active .venv
```

There is also an accessible debug facility to quickly run any edge case encountered while running Topologiq with `log_stats` enabled.
``` bash
# Pick up and replicate any available edge case (i.e. circuits Topologiq failed to build).
uv run src/topologiq/debug.py
python3 src/topologiq/debug.py  # Requires active .venv

# NB! Case must have been logged to stats, which only happens when Topologiq runs with `log_stats` enabled.
# NB! Currently available only for example graphs (the foundational graph must exist in file to replicate it).
```

## Yeah, but how does it work, really?
Detailed insight into **Topologiq** and, hopefully, a paper, is in progress. Meanwhile, below, a quick overview of what goes on under the hood. 

**Input.** **Topologiq** will look for an incoming ZX graph and, if needed and possible, convert it into a native format.
- ***Native format:*** A simple dictionary of nodes and edges (see `./src/topologiq/assets/graphs/simple_graphs.py` for examples).
- ***PyZX interoperability:*** PyZX graphs supported (check `docs/examples/pyzx_cnots.ipynb` for an example).
  - NB. If using a random PyZX circuit, ensure all qubit lines are interconnected. Else, the graph has independent subgraphs, which Topologiq does not support.

**Process.** **Topologiq** will traverse the ZX graph transforming each spider into an equivalent lattice surgery "primitive" positioned in a 3D space.
- ***Positioning:*** define a number of tentative 3D positions for each spider.
- ***Pathfinding:*** determine which tentative positions allow topologically-correct paths.
- ***Value function:*** choose best of any number of topologically-correct paths from previous step

The final choice considers the length of each topologically-correct path found in during pathfinding, as well as their relative impact to the feasibility of future placements. Having said that, all steps in the process share objects and undertake checks that avoid overloading the final step with many unreasonably suboptimal paths.

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
