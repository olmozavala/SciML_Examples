# PINN Explorer

This directory contains a codebase and an interactive Dash application for exploring **Physics-Informed Neural Networks (PINNs)**. It demonstrates how to solve partial differential equations (PDEs), specifically the Heat Equation and Burgers' Equation, using various neural network-based approaches.

## Features

- **Standard PINN**: Evaluates the strong form of the PDE pointwise using a standard collocation approach.
- **Variational PINN (VPINN)**: Uses a weak formulation and integration against test functions to deal with discontinuities and shocks (e.g., in Burgers' equation).
- **Conservative PINN (cPINN)**: Enforces physical conservation laws (like flux) to ensure greater physical consistency.
- **Curriculum Learning**: Progressively expands the considered temporal domain, starting with a simpler short-time subproblem and gradually increasing the time interval to capture complex behaviors effectively.
- **Transfer Learning**: Starts from a warm state using previously saved models to dramatically accelerate convergence.
- **Interactive Dash Dashboard (`pinn_dash_app.py`)**: An elaborate web application (written in Dash, using Plotly dark themes) for configuring and training the PINNs. Includes dynamic visualizations for loss trajectories, true analytical solutions, collocation point distribution, and model comparison.
- **Adaptive Techniques**: Built-in support for adaptive resampling (moving training points to regions with high error) and adaptive loss weighting (dynamic balance between PDE, IC, and BC losses).

## Code Structure

- `pinn_core.py`: The core backend containing the `PINN` neural network architecture, automatic differentiation routines, specialized loss functions (conservation loss, weak-form VPINN loss), and the training loop.
- `pinn_dash_app.py`: The main GUI application. Provides standard, VPINN, cPINN, and curriculum learning tabs layout and runs background threads for training.
- `heat_equation.py` & `burgers_equation.py`: Equations parameters, boundaries, initial conditions, and specific PDE residual computations tailored to each equation.
- `reference_solutions.py`: Ground truth data generating functions (analytical or numerical approximations) to compare the neural network predictions against.
- `weights_utils.py`: Helpers to handle saving model state dictionaries and metadata, enabling transfer learning and comparisons.
- `pyproject.toml`, `uv.lock`, `torchenv.yml`: Project dependency descriptors and environment definitions (often managed via `uv`).

## How to Run

Activate the environment (e.g. `particleviz` or the one prescribed in the lockfile) and run the Dash application:

```bash
uv run pinn_dash_app.py
```
Or with standard Python:
```bash
python pinn_dash_app.py
```

Then navigate to the local server URL indicated in your terminal to view the dashboard.
