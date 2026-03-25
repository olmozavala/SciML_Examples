# SciML Examples - Physics-Informed & Neural Differential Equations

A collection of interactive toolkits for Scientific Machine Learning (SciML), featuring Dash dashboards for training, visualizing, and comparing modern architecture paradigms.

## 🚀 Overview

This repository contains three major SciML modules:

1.  **Neural ODEs (Ordinary Differential Equations)**: Learning continuous-time dynamics and continuous-depth classifiers.
2.  **PINNs (Physics-Informed Neural Networks)**: Solving PDEs (Burgers, Heat) by embedding physical constraints into the loss function.
3.  **Neural Operator Learning**: Learning mappings between function spaces using architectures like FNO (Fourier Neural Operators) and DeepONet.

---

## 🧠 Neural ODEs (`/NeuralODE`)
Focuses on parameterizing the derivative of the hidden state using a neural network.

- **Tasks**: 
    - **Spiral Discovery**: 2D trajectory regression.
    - **Concentric Classifier**: Continuous-depth image classification logic.
- **Key Features**:
    - Subprocess-isolated training (keeping the UI responsive).
    - Adaptive depth (dynamic ODE solvers like `dopri5`).
    - Model comparison dashboard for previously trained weights.
- **To run**: `python NeuralODE/neural_ode_dash_app.py`

## 🌊 PINNs (`/PINN`)
Implements Neural Networks that respect physical laws defined by PDEs.

- **Supported Equations**: Burgers' Equation, Heat Equation.
- **Variants**: Standard PINN, Adaptive Sampling, Variational PINN (VPINN).
- **Key Features**:
    - Interactive domain and boundary condition setup.
    - Loss history visualization (PDE loss vs Boundary loss).
    - One-click model deletion and refresh.
- **To run**: `python PINN/pinn_dash_app.py`

## 📡 Neural Operator Learning (`/Neural_Operator_Learning`)
Learning operators that map input functions to output functions, invariant to discretization.

- **Models**: FNO (Fourier Neural Operator), DeepONet.
- **To run**: `python Neural_Operator_Learning/operator_dash_app.py`

---

## 🛠 Setup & Environment

The project relies on `torch`, `torchdiffeq`, and `dash`.

```bash
# Recommended environment setup
uv venv
source .venv/bin/activate
uv pip install torch torchdiffeq dash dash-bootstrap-components plotly scikit-learn
```

### Note on Training
Both Neural ODE and PINN apps offload training to background workers to ensure the dashboard remains perfectly fluid during high-compute tasks.

## 📈 Author
**Olmo Zavala**
