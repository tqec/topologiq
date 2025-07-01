# Algorithmic Lattice Surgery
A family of algorithms (ok, there's currently *one* full algorithm, but it is a foundation to make a family of them) to convert ZX circuits into logical versions of themselves.

It produces logical computations like the ones below (individual examples available as animated GIFs in the media folder). 


<video width="480" src="./assets/media/combo.mp4"></video>

*Video 1. A sample of outputs.*

***Note.*** This is work in progress.

***Note.*** Hadamards are NOT supported yet.

***Note.*** A better README is on the way.

## Examples
To run an example, run any of the following commands from the root of the repository. The algorithm will stop when it finds a succesfull solution (most times), or run up to ten times (sometimes).

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
You can also generate PyZX circuits programatically and convert them into the type of graphs needed for the algorithm to run. Check `run.py` for a blueprint.

***IMPORTANT!*** Hyperparameters are currently set to values that more or less increase the odds of finding a successful solution in a test run with the examples above. This does **NOT** mean the algorithm will produce an optimised (or any) computation by simply running it with current hyperparameters. The idea is, of course, to add an automated approach to discovering optimal hyperparameters for different graphs, but this is not currently available. To vary hyperparameters manually, edit `run_hyper_params.py`.

## Outputs
A succesfull result will produce a 3D interactive graph (pops up), a GIF animation of the process (saves to `./outputs/gif/`) and a dictionary of edges containing the different blocks needed to build the logical computation.

## Pending
Everything is pending, but below a list of highest priorities:
- Enable automatic selection and variation of hyperparameters.
- Add support for Hadamards.
- Improve PyZX support.
- Improve run-times.

