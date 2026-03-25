# Neural ODE Examples & Dashboard

This directory contains examples and core components for continuous-depth architectures using Neural Ordinary Differential Equations (Neural ODEs), built with PyTorch and `torchdiffeq`. An interactive Dash application is provided for visualizing the training and predictions of these models.

## Project Structure

- **`neural_ode_core.py`**: Contains the core PyTorch module definitions for the `NeuralODE` block and `NeuralODEClassifier`. It utilizes the adjoint method (`odeint_adjoint`) to ensure constant-memory backpropagation.
- **`spiral_example.py`**: Generates synthetic ground truth data for a 2D non-linear spiral trajectory governed by a known differential equation. This is used to demonstrate how Neural ODEs can accurately discover and regress dynamical system vector fields.
- **`classifier_example.py`**: Generates a 2D concentric circles dataset (via `sklearn`) to demonstrate how continuous-depth networks can naturally separate latent states and perform binary classification.
- **`neural_ode_dash_app.py`**: An interactive web frontend built with Plotly Dash. The application allows users to selectively train the models on a background thread while monitoring loss tracking and visualizing prediction trajectories (for the spiral) or decision boundaries (for the classifier) in real time. It uses Dash Bootstrap Components to provide a clean, modern user interface.

## Requirements

Ensure you have the following installed in your environment (e.g., your `particleviz` environment):
- `torch`
- `torchdiffeq`
- `scikit-learn`
- `dash`
- `dash-bootstrap-components`
- `plotly`

## Running the Application

To run the interactive dashboard, execute the application script directly:

```bash
uv run neural_ode_dash_app.py
```
*(Or `python neural_ode_dash_app.py` depending on the environment you are currently using).*

Navigate to `http://127.0.0.1:8052` in your browser. The dashboard contains an interactive playground with configurations for epochs, solver methods (e.g., DOPRI5, RK4, Euler), and learning rate adjustments for both task types.
