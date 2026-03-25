# Neural Operator Learning

This directory contains small, self-contained examples of **neural operators** for the seminar. Unlike PINNs, which learn a single PDE solution constrained by the governing equation, neural operators learn a map between entire functions.

These examples are now **GPU-only** and include a Dash interface following the same seminar style as the existing `PINN` material.

The toy task used here is the 1D Poisson problem

```text
-u''(x) = f(x),  x in (0, 1)
u(0) = u(1) = 0
```

Both examples learn the operator

```text
f(x) -> u(x)
```

on a fixed grid, where each forcing function `f(x)` is generated as a random combination of sine modes. The exact solution is computed analytically in the sine basis, so the dataset is cheap to generate and easy to explain.

## Files

- `operator_data.py`: dataset generation and shared utilities.
- `operator_models.py`: shared DeepONet/FNO architectures and GPU training utilities.
- `deeponet_example.py`: simple DeepONet with branch and trunk networks.
- `fno_example.py`: simple 1D Fourier Neural Operator example.
- `operator_dash_app.py`: Dash interface for training and comparing both methods.
  Includes a `DeepONet Evaluation` tab to load saved checkpoints, select forcing profiles, and run inference.
- `pyproject.toml`: minimal dependencies for running the examples.
- `assets/operator_learning.css`: styling for the dashboard.

## How To Run

From this directory:

```bash
python deeponet_example.py
python fno_example.py
python operator_dash_app.py
```

Or with `uv`:

```bash
uv run deeponet_example.py
uv run fno_example.py
uv run operator_dash_app.py
```

## Teaching Notes

- `DeepONet` separates the input function encoding (`branch net`) from the query location encoding (`trunk net`).
- `FNO` learns directly on the full discretized function and updates features in Fourier space.
- The same Poisson operator is used in both scripts so the comparison focuses on architecture rather than on different datasets.
- A CUDA-capable GPU is required; the scripts intentionally fail fast instead of silently falling back to CPU.
- During DeepONet dashboard training, the best checkpoint by validation MSE is saved in `weights/deeponet/<run_name>/best_model.pt` with metadata in `best_model.json`.
- Dashboard training uses multiple forcing functions (`n_train` samples), not a single forcing.
- The DeepONet Evaluation tab uses additional forcing profiles that are separate from the training/validation/test seeds.
- Dashboard training now includes early stopping (`patience=30`) and learning-rate reduction on validation plateau.
- Changing `Grid Points` or `Sine Modes` in the training tabs updates the forcing overlay and training-point-location plots immediately.
