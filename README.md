# Algorithmic Lattice Surgery
A family of algorithms (ok, there's currently *one* full algorithm, but it is a foundation to make a family of them) to convert ZX circuits into logical versions of themselves.

It produces logical computations like the ones below (individual examples available in `./assets/media`). 

<video width="850" height="480" src="https://github.com/user-attachments/assets/624df80d-7ed6-42e6-9567-fc59798f70c8"></video>

*Video 1. A sample of outputs.*

***Note.*** This is work in progress.

***Note.*** Hadamards are NOT supported yet.

***Note.*** A better README is on the way.

## Examples
To run an example, run any of the following commands from the root of the repository. The algorithm will stop when it finds a succesfull solution or run up to ten times.

``` bash
# A CNOT, using PyZX.
python -m run --pyzx:cnot

# Random series of CNOTs, using PyZX: CNOT_HAD_PHASE_circuit()
python -m run --pyzx:cnots

# Random circuit, using PyZX: zx.generate.cliffordT()
python -m run --pyzx:random

# Random circuit, optimised, using PyZX: zx.generate.cliffordT(), zx.simplify.phase_free_simp()
python -m run --pyzx:random_optimised

# A 7-qubit Steane code, from a non-descript graph. 
# Ps. This one takes longest. It is a tightly-packed/highly-optimised circuit, so a few rounds are often needed to find a successful solution.
python -m run --graph:steane

```

It would also be fantastic to hear of tests using other circuits. You can use a non-descript ZX graph defined as a dictionary of nodes and edges (see `assets/graphs/simple_graphs.py` for examples) or a PyZX graph (check `run.py` for a blueprint and `assets/graphs/pyzx_graphs.py` for examples).

That said, please note a degree of failed results is expected. The current goal is to inform developmental priorities by identifying types of circuits for which the current implementation performs good, less good, bad, and awfully.

***IMPORTANT!*** Hyperparameters are currently set to values that more or less increase the odds of finding a successful solution in a test run with the examples above. This does **NOT** mean the algorithm will produce an optimal result by simply running it with current hyperparameters. The idea is, of course, to add an automated approach to discovering optimal hyperparameters for different graphs, but this is not currently available. To vary hyperparameters manually, edit `run_hyper_params.py`.

## Outputs
A succesfull result will produce:
- a 3D interactive graph (pops up)
- a GIF animation of the process (saves to `./outputs/gif/`)
- a TXT file with information about the initial ZX graph, intermediate state, and final result (saves to `./outputs/txt/`).

The information printed to the TXT file is also available for programmatic use.

## Pending
Everything is pending, but below a list of highest priorities:
- Enable automatic selection and variation of hyperparameters
- Add support for Hadamards
- Improve PyZX support
- Improve run-times.
