# 3D Solenoidal Turbulence Reproduction

This repository contains a standalone implementation for reproducing the results of atmospheric/astrophysical turbulence studies focusing on transverse-dominant anisotropic dispersion and transient trapping in 3D solenoidal turbulence.

## Features
- **Lagrangian Tracer Integration**: High-fidelity RK4 integration of passive tracers.
- **Anisotropic Dispersion Analysis**: Calculation of parallel and perpendicular mean squared displacements.
- **Vortex Residence Time**: Identification of trapping events and calculation of tau_Q.
- **Displacement PDF Evolution**: Statistical analysis of tracer displacements across multiple time lags.
- **Chaos and FTLE Analysis**: Correlation between chaotic stretching and vortex trapping.

## Installation
Dependencies are listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```
*Note: Ensure `numpy`, `h5py`, and `matplotlib` are installed in your environment.*

## Usage
The main entry point is `main.py`.

### Running Experiments
To run a specific experiment (e.g., EXP1) in smoke mode:
```bash
python main.py EXP1 --mode smoke --raw-data-dir /path/to/vtk/data
```

To run the full benchmark suite:
```bash
python main.py full-benchmark --mode full --raw-data-dir /path/to/vtk/data
```

### Modes
- `smoke`: Runs a lightweight version of the experiments for verification.
- `full`: Runs the full reproduction suite as described in the paper.

## CLI Arguments
- `experiment`: Choice of `EXP1`, `EXP2`, `EXP3`, `EXP4`, `EXP5`, or `full-benchmark`.
- `--mode`: `full` (default) or `smoke`.
- `--raw-data-dir`: Path to the directory containing VTK snapshots.
- `--output-dir`: Path to the directory where results will be saved (default: `./outputs`).
- `--n-tracers`: Override the number of tracers.
- `--domain-size`: Override the domain size L.

## Tests
Comprehensive end-to-end tests are provided in the `tests/` directory.
To run tests:
```bash
pytest tests/test_comprehensive.py
```

## License
MIT
