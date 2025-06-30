# Algorithmic Lattice Surgery
A family of algorithms (ok, there's currently *one* full algorithm, but it is a foundation to make a family of them) to convert ZX circuits into logical versions of themselves.

***Note.*** This is work in progress.

***Note.*** A better README is on the way.

## Example
To run an example, run any of the following commands from the root of the repository. The algorithm will stop when it finds a succesfull solution (most times), or run up to ten times (sometimes).

A 7-qubit Steane code, from a non-descript graph. 

```
python -m run --simple:steane
```

A CNOT, from a PyZX graph.

```
python -m run --pyzx:cnot

```

A succesfull result will produce a GIF animation like the one below (saved to `./outputs/gif/`) and a dictionary of edges containing the different blocks needed to build the logical computation.

![GIF animation of an example run of the algorithm](assets/media/steane.gif)   
*Figure 1. GIF animation of the process to build a Steane code (example run).*

***Note.*** Currently, hyper-parameters are set to values that more or less increase the odds of finding a successful solution in a test run using Steane and in a reasonable amount of time (in tests carried out locally, 5 min in average, with super lucky runs taking less than a minute). To vary hyper-parameters manually, edit `run_hyper_params.py`, which will enable results of varied quality and significant variation in run-times. There is, currently, no automatic method of varying or smartly-determining hyper-parameters.

***Note.*** For the moment, the only pre-built PyZX test graph available is a CNOT, but more will be added in due time. You can in theory also build your own PyZX graph and run it using an approach similar to what is currently available in `run.py`.

## Pending
Everything is pending, but below a list of highest priorities:
- Improve the runner function in `run.py` to enable automatic selection and variation of hyper-parameters.
- Improve run-times.
- Improve support for PyZX circuits.
- Add support for circuits with Hadamards.
