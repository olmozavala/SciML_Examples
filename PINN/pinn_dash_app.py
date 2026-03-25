"""PINN Dash interface: training tab and comparison tab.

Run with: uv run pinn_dash_app.py (from PINN dir) or python pinn_dash_app.py
"""
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, MATCH, ALL
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from pinn_core import PINN, get_training_points_plotly, train_model
from heat_equation import (
    X_DOMAIN as HEAT_X_DOMAIN,
    T_DOMAIN as HEAT_T_DOMAIN,
    NUM_HIDDEN as HEAT_NUM_HIDDEN,
    DIM_HIDDEN as HEAT_DIM_HIDDEN,
    initial_condition as heat_ic,
    compute_loss as heat_compute_loss,
)
from burgers_equation import (
    X_DOMAIN as BURGERS_X_DOMAIN,
    T_DOMAIN as BURGERS_T_DOMAIN,
    NUM_HIDDEN as BURGERS_NUM_HIDDEN,
    DIM_HIDDEN as BURGERS_DIM_HIDDEN,
    initial_condition as burgers_ic,
    compute_loss as burgers_compute_loss,
)
from reference_solutions import heat_exact_solution, burgers_reference_solution
from weights_utils import list_weights, load_metadata, save_model, create_run_dir, update_model

LOSS_HISTORY_PATH = Path(__file__).resolve().parent / "loss_history.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Default training params
DEFAULT_N_X = 40
DEFAULT_N_T = 40
DEFAULT_EPOCHS = 2000
DEFAULT_LR = 0.025

PLOTLY_TEMPLATE = "plotly_dark"

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    assets_folder=str(Path(__file__).resolve().parent / "assets"),
)

# ── LaTeX strings ─────────────────────────────────────────────────────────────
HEAT_LOSS_LATEX = r"""
$$
\mathcal{L} = \mathcal{L}_{PDE} + \mathcal{L}_{IC} + \mathcal{L}_{BC}
$$
- $\mathcal{L}_{PDE} = \frac{1}{N}\sum_i \left(\frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2}\right)^2$, $\alpha = 1.1$
- $\mathcal{L}_{IC} = \frac{1}{N}\sum_i \left(u(x,0) - u_0(x)\right)^2$
- $\mathcal{L}_{BC} = \frac{1}{N}\sum_i \left(u(x_{bnd},t)\right)^2$
"""

BURGERS_LOSS_LATEX = r"""
$$
\mathcal{L} = \mathcal{L}_{PDE} + \mathcal{L}_{IC} + \mathcal{L}_{BC}
$$
- $\mathcal{L}_{PDE} = \frac{1}{N}\sum_i \left(\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2}\right)^2$, $\nu = 0.01/\pi$
- $\mathcal{L}_{IC} = \frac{1}{N}\sum_i \left(u(x,0) - u_0(x)\right)^2$, $u_0(x) = -\sin(\pi x)$
- $\mathcal{L}_{BC} = \frac{1}{N}\sum_i \left(u(x_{bnd},t)\right)^2$
"""

HEAT_EQUATION_LATEX = r"""
$$
\frac{\partial u}{\partial t} - 1.1 \frac{\partial^2 u}{\partial x^2} = 0
$$
"""

BURGERS_EQUATION_LATEX = r"""
$$
\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0,
\quad \nu = \frac{0.01}{\pi}
$$
"""



# ── Helper: empty dark figure ─────────────────────────────────────────────────
def _empty_fig(title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=title,
        paper_bgcolor="#141929",
        plot_bgcolor="#0d1120",
        margin=dict(l=40, r=20, t=40, b=40),
        font=dict(color="#a0b4d8"),
    )
    return fig


def _apply_dark_layout(fig: go.Figure, title: str = "", xaxis_title: str = "", yaxis_title: str = "") -> None:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=title, font=dict(size=13, color="#c0d0f0")),
        paper_bgcolor="#141929",
        plot_bgcolor="#0d1120",
        margin=dict(l=45, r=20, t=45, b=45),
        font=dict(color="#a0b4d8", size=11),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )


# ── Training backend ──────────────────────────────────────────────────────────
def _get_loss_history_path(method_id: str) -> Path:
    return Path(__file__).resolve().parent / f"loss_history_{method_id}.json"


_loss_lock = threading.Lock()

def _write_loss_history(data: dict, method_id: str) -> None:
    with _loss_lock:
        with open(_get_loss_history_path(method_id), "w") as f:
            json.dump(data, f, indent=2)


def _read_loss_history(method_id: str) -> dict:
    p = _get_loss_history_path(method_id)
    if not p.exists():
        return {"status": "idle", "history": []}
    with _loss_lock:
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"status": "idle", "history": []}


def _run_training(
    method_id: str,
    equation: str,
    n_x: int,
    n_t: int,
    epochs: int,
    lr: float,
    strategy: str = "standard",
    learning_type: str = "standard",
    adaptive_toggles: list[str] = None,
) -> None:
    if adaptive_toggles is None:
        adaptive_toggles = []

    if equation == "heat":
        x_domain, t_domain = HEAT_X_DOMAIN, HEAT_T_DOMAIN
        num_hidden, dim_hidden = HEAT_NUM_HIDDEN, HEAT_DIM_HIDDEN
        compute_loss = heat_compute_loss
    else:
        x_domain, t_domain = BURGERS_X_DOMAIN, BURGERS_T_DOMAIN
        num_hidden, dim_hidden = BURGERS_NUM_HIDDEN, BURGERS_DIM_HIDDEN
        compute_loss = burgers_compute_loss

    x_np = np.linspace(x_domain[0], x_domain[1], n_x, dtype=np.float32)
    t_np = np.linspace(t_domain[0], t_domain[1], n_t, dtype=np.float32)
    x_raw = torch.tensor(x_np, requires_grad=True).to(device)
    t_raw = torch.tensor(t_np, requires_grad=True).to(device)
    grids = torch.meshgrid(x_raw, t_raw, indexing="ij")
    x_train = grids[0].flatten().reshape(-1, 1).to(device)
    t_train = grids[1].flatten().reshape(-1, 1).to(device)

    n_val = max(1, int(0.25 * n_x * n_t))
    rng = np.random.default_rng()
    x_val_np = rng.uniform(x_domain[0], x_domain[1], n_val).astype(np.float32)
    t_val_np = rng.uniform(t_domain[0], t_domain[1], n_val).astype(np.float32)
    x_val = torch.tensor(x_val_np, requires_grad=True).reshape(-1, 1).to(device)
    t_val = torch.tensor(t_val_np, requires_grad=True).reshape(-1, 1).to(device)

    U_nn = PINN(num_hidden, dim_hidden).to(device)

    # Transfer Learning: Load weights if they exist
    if learning_type == "transfer":
        from weights_utils import list_weights
        opts = list_weights(equation)
        if opts:
            latest_weights = opts[0]
            try:
                if Path(latest_weights).exists():
                    U_nn.load_state_dict(torch.load(latest_weights, map_location=device))
                    print(f"Loaded transfer weights from {latest_weights}")
                else:
                    print(f"Transfer weights file not found: {latest_weights}")
            except Exception as e:
                print(f"Failed to load transfer weights: {e}")

    # Pre-generate test functions if VPINN strategy is used
    if strategy == "vpinn":
        from pinn_core import get_vpinn_test_functions, compute_vpinn_loss
        test_fns = get_vpinn_test_functions(5, 5, x_domain, t_domain)
        
        def loss_fn(model: PINN) -> tuple[torch.Tensor, dict]:
            pde_loss = compute_vpinn_loss(model, equation, x_train, t_train, test_fns)
            _, components = compute_loss(model, x_train, t_train, return_components=True)
            components["pde"] = pde_loss
            total_loss = pde_loss + components["ic"] + components["bc"]
            return total_loss, components
    elif strategy == "cpinn":
        from pinn_core import compute_conservation_loss
        def loss_fn(model: PINN) -> tuple[torch.Tensor, dict]:
            pde_total_loss, components = compute_loss(model, x_train, t_train, return_components=True)
            cons_loss = compute_conservation_loss(model, equation, x_domain, t_train)
            components["conservation"] = cons_loss
            # Weighting conservation slightly higher to ensure physical consistency
            total_loss = pde_total_loss + 1.0 * cons_loss 
            return total_loss, components
    else:
        def loss_fn(model: PINN) -> torch.Tensor | tuple[torch.Tensor, dict]:
            return compute_loss(model, x_train, t_train, return_components=True)

    def on_epoch(ep: int, train_loss_val: float) -> None:
        data = _read_loss_history(method_id)
        if data.get("status") != "training":
            return

        val_loss_tensor = compute_loss(U_nn, x_val, t_val)
        val_loss_val = float(val_loss_tensor.detach())

        history = data.get("history", [])
        history.append({"epoch": ep, "train_loss": train_loss_val, "val_loss": val_loss_val})
        data["history"] = history
        data["current_epoch"] = ep
        data["total_epochs"] = epochs

        best_val = data.get("best_val_loss", float("inf"))
        if val_loss_val < best_val:
            meta = {
                "equation_type": equation,
                "n_x": n_x, "n_t": n_t, "epochs": epochs,
                "num_hidden": num_hidden, "dim_hidden": dim_hidden,
                "strategy": strategy, "learning_type": learning_type,
                "best_epoch": ep, "best_val_loss": val_loss_val,
                "adaptive_options": adaptive_toggles,
            }
            weights_path, meta_path = update_model(run_dir, U_nn, meta)
            data["best_val_loss"] = val_loss_val
            data["best_weights_path"] = weights_path
            data["best_metadata_path"] = meta_path

        _write_loss_history(data, method_id)

    # Create a single fixed run directory for this training run.
    # The best model will be OVERWRITTEN here on each improvement.
    run_dir = create_run_dir(equation, strategy, learning_type)

    _write_loss_history({
        "status": "training",
        "history": [],
        "best_val_loss": float("inf"),
        "current_epoch": 0,
        "total_epochs": epochs,
    }, method_id)

    try:
        def val_loss_fn(model: PINN) -> float:
            return float(compute_loss(model, x_val, t_val).detach())

        adaptive_weights = "weights" in adaptive_toggles
        do_resampling = "resampling" in adaptive_toggles
        is_curriculum = (learning_type == "curriculum")

        if do_resampling or is_curriculum:
            # Resampling or curriculum every 500 epochs
            res_interval = 500
            for chunk_start in range(0, epochs, res_interval):
                chunk_epochs = min(res_interval, epochs - chunk_start)
                
                # Curriculum learning: expand time domain
                if is_curriculum:
                    frac = min(1.0, (chunk_start + res_interval) / epochs)
                    t_max_curr = t_domain[0] + frac * (t_domain[1] - t_domain[0])
                    
                    mask = (grids[1].flatten() <= t_max_curr)
                    x_train = grids[0].flatten()[mask].reshape(-1, 1).detach().requires_grad_(True).to(device)
                    t_train = grids[1].flatten()[mask].reshape(-1, 1).detach().requires_grad_(True).to(device)

                # Update training points based on residual
                if do_resampling and chunk_start > 0:
                    from pinn_core import get_adaptive_points
                    def residual_only(m, x, t):
                        return compute_loss(m, x, t, return_components=True)[1]["pde"]
                    
                    x_res, t_res = get_adaptive_points(
                        U_nn, residual_only, x_domain, t_domain, len(x_raw) * len(t_raw) // 2
                    )
                    # Mix original grid with adaptive points (50/50)
                    # x_train, t_train refer to the flattened grid or curriculum subset
                    half_len_x = max(1, len(x_train) // 2)
                    half_len_t = max(1, len(t_train) // 2)
                    x_train = torch.cat([x_train[:half_len_x], x_res], dim=0)
                    t_train = torch.cat([t_train[:half_len_t], t_res], dim=0)
                
                train_model(
                    U_nn, loss_fn,
                    learning_rate=lr,
                    max_epochs=chunk_epochs,
                    on_epoch_callback=lambda ep, loss: on_epoch(ep + chunk_start, loss),
                    callback_interval=10,
                    patience=10 if chunk_start == 0 else 0, # Only early stop on first chunk for simplicity
                    val_loss_fn=val_loss_fn,
                    adaptive_weights=adaptive_weights
                )
        else:
            train_model(
                U_nn, loss_fn,
                learning_rate=lr,
                max_epochs=epochs,
                on_epoch_callback=on_epoch,
                callback_interval=10,
                patience=10,
                val_loss_fn=val_loss_fn,
                adaptive_weights=adaptive_weights,
            )

        data = _read_loss_history(method_id)
        history = data.get("history", [])
        final_epoch = history[-1]["epoch"] if history else epochs
        stopped_early = final_epoch < epochs - 1
        data["status"] = "complete"
        data["stopped_early"] = stopped_early
        _write_loss_history(data, method_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        data = _read_loss_history(method_id)
        data["status"] = "error"
        data["error"] = str(e)
        _write_loss_history(data, method_id)


# ── Layout components ─────────────────────────────────────────────────────────
def _section_label(text: str) -> html.Div:
    return html.Div(text, className="section-header")

_graph_config = {"responsive": True, "displayModeBar": False}

def _plot_card(title: str, graph_id: dict) -> html.Div:
    return html.Div(
        [
            html.Div(title, className="plot-title"),
            dcc.Graph(id=graph_id, config=_graph_config, style={"height": "280px"}),
        ],
        className="plot-card",
    )

METHODS_CONFIG = [
    {
        "id": "standard",
        "name": "Standard PINN",
        "eq": "heat",
        "nx": 40, "nt": 40, "epochs": 2000, "lr": 0.01,
        "strategy": "standard", "learning_type": "standard",
        "description": "Standard Physics-Informed Neural Network evaluating the strong form of the Heat equation directly at collocation points."
    },
    {
        "id": "vpinn",
        "name": "Variational PINN (VPINN)",
        "eq": "burgers",
        "nx": 25, "nt": 25, "epochs": 2000, "lr": 0.01,
        "strategy": "vpinn", "learning_type": "standard",
        "description": "VPINN uses a weak formulation integrated against test functions, making it more resilient to the discontinuities/shocks formed by Burgers' Equation."
    },
    {
        "id": "cpinn",
        "name": "Conservative PINN (cPINN)",
        "eq": "heat",
        "nx": 50, "nt": 50, "epochs": 2000, "lr": 0.01,
        "strategy": "cpinn", "learning_type": "standard",
        "description": "cPINN explicitly enforces physical conservation (like flux/mass constraints). This improves physical consistency.",
    },
    {
        "id": "curriculum",
        "name": "Curriculum Learning",
        "eq": "burgers",
        "nx": 40, "nt": 40, "epochs": 2500, "lr": 0.01,
        "strategy": "standard", "learning_type": "curriculum",
        "description": "Starts learning on a small initial sub-domain where dynamics are simpler, progressively expanding in time to capture complex behaviors effectively.",
    }
]

def create_method_tab(config: dict) -> dbc.Container:
    method_id = config["id"]
    sidebar = html.Div([
        _section_label("Configuration"),
        dcc.Markdown(f"**{config['name']}**\n{config['description']}", mathjax=True, style={"fontSize": "13px", "color": "#a0b4d8", "marginBottom": "15px"}),
        
        _section_label("Equation"),
        dcc.Dropdown(
            id={"type": "eq-select", "method": method_id},
            options=[
                {"label": "Heat Equation", "value": "heat"},
                {"label": "Burgers Equation", "value": "burgers"},
            ],
            value=config["eq"],
            clearable=False,
            style={"marginBottom": "8px"},
        ),
        dcc.Markdown(id={"type": "equation-markdown", "method": method_id}, mathjax=True, style={"fontSize": "14px", "minHeight": "60px"}),
        html.Hr(style={"borderColor": "#2a3555", "margin": "12px 0"}),

        _section_label("Grid Resolution"),
        html.Div("n_x (spatial points)", className="control-label"),
        dcc.Slider(
            id={"type": "n-x", "method": method_id}, min=10, max=100, step=10, value=config["nx"],
            marks={i: {"label": str(i), "style": {"color": "#5577aa", "fontSize": "10px"}} for i in [10, 40, 70, 100]},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div("n_t (temporal points)", className="control-label mt-2"),
        dcc.Slider(
            id={"type": "n-t", "method": method_id}, min=10, max=100, step=10, value=config["nt"],
            marks={i: {"label": str(i), "style": {"color": "#5577aa", "fontSize": "10px"}} for i in [10, 40, 70, 100]},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Hr(style={"borderColor": "#2a3555", "margin": "12px 0"}),

        _section_label("Optimizer"),
        html.Div("Epochs", className="control-label"),
        dcc.Slider(
            id={"type": "epochs", "method": method_id}, min=500, max=10000, step=500, value=config["epochs"],
            marks={i: {"label": str(i), "style": {"color": "#5577aa", "fontSize": "10px"}} for i in [500, 2000, 5000, 10000]},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div("Learning Rate", className="control-label mt-2"),
        dcc.Input(
            id={"type": "lr", "method": method_id}, type="number", value=config["lr"], step=0.001, min=0.0001,
            className="w-100", style={"background": "#1e2842", "border": "1px solid #2a3555", "color": "#e8eaf0", "borderRadius": "6px", "padding": "6px 10px", "marginBottom": "12px"},
        ),
        
        dcc.Store(id={"type": "strategy-select", "method": method_id}, data=config["strategy"]),
        dcc.Store(id={"type": "learning-type-select", "method": method_id}, data=config["learning_type"]),
        
        dbc.Checklist(
            id={"type": "adaptive-toggles", "method": method_id},
            options=[
                {"label": "Adaptive Loss Weights", "value": "weights"},
                {"label": "Adaptive Resampling", "value": "resampling"},
            ],
            value=[],
            inline=False,
            switch=True,
            style={"fontSize": "13px", "color": "#8899bb"},
        ),
        html.Hr(style={"borderColor": "#2a3555", "margin": "12px 0"}),

        _section_label("Actions"),
        dbc.Row([
            dbc.Col(dbc.Button("🚀 Train", id={"type": "train-btn", "method": method_id}, n_clicks=0, className="w-100 btn-train", color="primary"), width=7),
            dbc.Col(dbc.Button("🗑 Clear", id={"type": "clear-btn", "method": method_id}, n_clicks=0, className="w-100 btn-clear", color="secondary", outline=True), width=5),
        ], className="mb-2"),
        dbc.Progress(id={"type": "train-progress", "method": method_id}, value=0, max=100, striped=True, animated=True, className="mb-2", style={"height": "6px"}),
        dcc.Loading(
            type="circle",
            children=html.Div(id={"type": "train-status", "method": method_id}),
        ),

        html.Hr(style={"borderColor": "#2a3555", "margin": "12px 0"}),
        _section_label("Loss Definition"),
        dcc.Markdown(id={"type": "loss-latex", "method": method_id}, mathjax=True, style={"fontSize": "13px", "marginBottom": "15px"}),
        
        html.Hr(style={"borderColor": "#2a3555", "margin": "12px 0"}),
        _section_label("Compare Models"),
        dcc.Dropdown(id={"type": "weights-select", "method": method_id}, options=[], value=None, style={"marginBottom": "8px"}),
        dbc.Button(
            "🔄 Refresh List",
            id={"type": "refresh-models-btn", "method": method_id},
            n_clicks=0,
            color="secondary",
            outline=True,
            size="sm",
            className="w-100",
            style={"fontSize": "0.75rem"},
        ),
        dbc.Button(
            "🗑 Delete All Models",
            id={"type": "delete-models-btn", "method": method_id},
            n_clicks=0,
            color="danger",
            outline=True,
            size="sm",
            className="w-100 mt-1",
            style={"fontSize": "0.75rem", "borderColor": "#7f1d1d", "color": "#fca5a5"},
        ),
        html.Div(id={"type": "delete-status", "method": method_id}, style={"fontSize": "0.72rem", "color": "#fca5a5", "marginTop": "4px", "minHeight": "18px"}),
    ], className="pinn-sidebar p-3", style={"height": "100%", "overflowY": "auto"})

    return dbc.Container([
        dbc.Row([
            dbc.Col(sidebar, width=3, style={"minWidth": "280px"}),
            dbc.Col([
                html.H4("Training Plots", className="section-header mt-2"),
                dbc.Row([
                    dbc.Col(_plot_card("Initial Condition  u(x, 0)", {"type": "ic-plot", "method": method_id}), width=6),
                    dbc.Col(_plot_card("Training Points", {"type": "train-points-plot", "method": method_id}), width=6),
                ]),
                dbc.Row([
                    dbc.Col(_plot_card("Ground-Truth Solution  u(x, t)", {"type": "true-solution-plot", "method": method_id}), width=6),
                    dbc.Col(_plot_card("Loss vs Epoch", {"type": "loss-plot", "method": method_id}), width=6),
                ]),
                html.Hr(style={"borderColor": "#2a3555", "margin": "20px 0"}),
                html.H4("Model Comparison", className="section-header"),
                dbc.Row([
                    dbc.Col(html.Div(id={"type": "model-meta-card", "method": method_id}), width=12),
                ]),
                dbc.Row([
                    dbc.Col(html.Div([
                        dcc.Graph(id={"type": "compare-plot", "method": method_id}, config=_graph_config, style={"height": "320px"})
                    ], className="plot-card"), width=12)
                ], className="mt-3")
            ], width=9)
        ], className="mt-2 g-3"),
        dcc.Store(id={"type": "loss-store", "method": method_id}, data={"history": []}),
        dcc.Interval(id={"type": "loss-interval", "method": method_id}, interval=1000, n_intervals=0, disabled=True),
    ], fluid=True, className="mt-3")

THEORY_MARKDOWN = r"""
### Physics-Informed Neural Networks (PINNs)

Physics-Informed Neural Networks involve constructing a neural network $u_\theta(x, t)$ to approximate the solution of a PDE by incorporating the physical laws directly into the loss function. The total loss $\mathcal{L}$ is typically defined as:
$$
\mathcal{L} = \mathcal{L}_{PDE} + \lambda_{IC}\mathcal{L}_{IC} + \lambda_{BC}\mathcal{L}_{BC}
$$

- **$\mathcal{L}_{PDE}$**: Evaluated on collocation points within the domain, it penalizes the residual of the governing PDE (e.g., $\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0$). The derivatives appearing in the PDE operator are obtained via automatic differentiation of the neural network with respect to its inputs.
- **$\mathcal{L}_{IC}$ & $\mathcal{L}_{BC}$**: Enforce the initial and boundary conditions, guiding the solution to fit the specific constraints of the problem.

---

### Learning Strategies

**1. Standard PINN (Strong Form)**
The standard PINN evaluates the strong form of the PDE pointwise across a set of scattered collocation points. It requires the network to be smooth enough to compute the necessary high-order differential operators, aiming for the residual of the PDE to be zero at every point.

**2. VPINN (Variational PINNs / Weak Form)**
Instead of evaluating the strong form of the PDE, VPINNs are based on the weak formulation. The PDE is multiplied by a set of test functions $v_k$ and integrated over the domain (usually computed via numerical quadrature). By transferring some spatial derivatives onto the test functions via integration by parts, VPINNs reduce the order of required network derivatives. This formulation is much more suitable for problems with discontinuities, shocks, or solutions with low regularity.

**3. cPINN (Conservative PINNs / Domain Decomposition)**
cPINNs split the computational domain into non-overlapping subdomains and enforce physical conservation laws strictly (such as flux continuity across interfaces). Each subdomain can have its own sub-network. This explicit enforcement of conservation improves training stability, handles large variations in the solution better, and ensures global physical correctness (e.g., mass or momentum conservation).

---

### Learning Types

**1. Standard Learning**
The neural network is trained on the entire spatio-temporal domain simultaneously using a fixed set of collocation, initial, and boundary points for all epochs.

**2. Curriculum Learning**
The network starts learning on a simpler sub-problem, such as a very short initial time interval $t \in [0, \delta t]$. Once the network accurately captures the initial dynamics, the "curriculum" progressively expands the domain (e.g., taking larger time slices) until the entire domain is covered. This is critical for convective PDEs (like Burgers' Equation) where shocks form over time, as it prevents the training from getting stuck in poor local minima caused by complex long-term dynamics.

**3. Transfer Learning**
The neural network is initialized with weights pre-trained on a similar, previously solved problem (e.g., predicting on a coarser grid, a different time span, or for slightly different PDE coefficients). Starting from a "warm" state dramatically accelerates convergence and often improves the final accuracy compared to random initialization.

---

### Adaptive Toggles

**1. Adaptive Loss Weights**
In multi-objective optimization (such as satisfying PDE, IC, and BCs simultaneously), the gradients from different loss terms can vary by orders of magnitude. For instance, the boundary loss gradient might dominate the PDE loss gradient, stalling training. Adaptive weighting methods dynamically scale the weights ($\lambda_{IC}, \lambda_{BC}$) throughout the training process based on the variance or maximum values of the backpropagated gradients. This ensures balanced learning across all objectives.

**2. Adaptive Resampling**
Using a static grid of collocation points can be highly inefficient, as the neural network might easily satisfy the PDE in smooth regions but fail dramatically in challenging ones. Adaptive resampling periodically evaluates the PDE residual across the domain and generates a new set of collocation points densely clustered in regions where the error is highest (e.g., near steep gradients or shocks). This dynamically focuses the network's learning capacity exactly where it struggles most.
"""

theory_tab = dbc.Container([
    dbc.Row(dbc.Col(
        html.Div([
            dcc.Markdown(THEORY_MARKDOWN, mathjax=True, className="p-4", style={"color": "#a0b4d8", "fontSize": "15px", "lineHeight": "1.6"})
        ], style={"background": "#141929", "borderRadius": "8px", "border": "1px solid #2a3555", "marginTop": "20px", "marginBottom": "20px"})
    ))
], fluid=True)

app.layout = html.Div(
    [
        # Hidden dummy Markdown to force global MathJax initialization
        html.Div(dcc.Markdown("$$ $$", mathjax=True), style={"display": "none"}),
        html.Div(
            [
                html.Span("⚛", style={"fontSize": "1.6rem", "marginRight": "0.5rem"}),
                html.Span("PINN Explorer", className="pinn-title"),
                html.Span("Physics-Informed Neural Networks", className="pinn-subtitle"),
            ],
            className="pinn-navbar d-flex align-items-center",
        ),
        dbc.Container(
            dbc.Tabs(
                id="tabs",
                active_tab="standard",
                children=[
                    dbc.Tab(label=cfg["name"], tab_id=cfg["id"], children=[create_method_tab(cfg)])
                    for cfg in METHODS_CONFIG
                ] + [
                    dbc.Tab(label="Theory & Configurations", tab_id="theory", children=[theory_tab]),
                ],
            ),
            fluid=True,
            className="px-3 pt-3",
        ),
    ],
    style={"minHeight": "100vh", "backgroundColor": "#0f1117"},
)

# ── Callbacks ─────────────────────────────────────────────────────────────────
@callback(
    Output({"type": "ic-plot", "method": MATCH}, "figure"),
    Output({"type": "train-points-plot", "method": MATCH}, "figure"),
    Output({"type": "true-solution-plot", "method": MATCH}, "figure"),
    Output({"type": "equation-markdown", "method": MATCH}, "children"),
    Output({"type": "loss-latex", "method": MATCH}, "children"),
    Input({"type": "eq-select", "method": MATCH}, "value"),
    Input({"type": "n-x", "method": MATCH}, "value"),
    Input({"type": "n-t", "method": MATCH}, "value"),
)
def update_training_plots(equation: str, n_x: int | None, n_t: int | None):
    if equation == "heat":
        x_domain, t_domain = HEAT_X_DOMAIN, HEAT_T_DOMAIN
        ic_fn = heat_ic
        eq_latex = HEAT_EQUATION_LATEX
        latex = HEAT_LOSS_LATEX
    else:
        x_domain, t_domain = BURGERS_X_DOMAIN, BURGERS_T_DOMAIN
        ic_fn = burgers_ic
        eq_latex = BURGERS_EQUATION_LATEX
        latex = BURGERS_LOSS_LATEX

    n_x = n_x or DEFAULT_N_X
    n_t = n_t or DEFAULT_N_T

    x_np = np.linspace(x_domain[0], x_domain[1], 300)
    x_ic = torch.tensor(x_np, dtype=torch.float32)
    u_ic = ic_fn(x_ic).detach().numpy().flatten()

    ic_fig = go.Figure()
    ic_fig.add_trace(go.Scatter(
        x=x_np.tolist(), y=u_ic.tolist(),
        mode="lines", name="u(x,0)",
        line=dict(color="#7eb3ff", width=2.5),
        fill="tozeroy", fillcolor="rgba(126,179,255,0.12)",
    ))
    _apply_dark_layout(ic_fig, "Initial Condition", "x", "u(x, 0)")

    x_train = np.linspace(x_domain[0], x_domain[1], n_x, dtype=np.float32)
    t_train = np.linspace(t_domain[0], t_domain[1], n_t, dtype=np.float32)
    traces = get_training_points_plotly(t_domain, x_domain, x_train, t_train)

    n_val = min(500, int(n_x * n_t / 2))
    rng = np.random.default_rng()
    x_val = rng.uniform(x_domain[0], x_domain[1], n_val)
    t_val = rng.uniform(t_domain[0], t_domain[1], n_val)

    pts_fig = go.Figure(data=traces)
    pts_fig.add_trace(go.Scatter(
        x=x_val, y=t_val, mode="markers", name="Validation",
        marker=dict(size=4, color="#f59e0b", symbol="x", opacity=0.7),
    ))
    _apply_dark_layout(
        pts_fig, f"Training Points ({n_x * n_t} int · {n_val} val)", "x", "t",
    )

    n_x_plot, n_t_plot = 120, 120
    x_plot = np.linspace(x_domain[0], x_domain[1], n_x_plot, dtype=np.float32)
    t_plot = np.linspace(t_domain[0], t_domain[1], n_t_plot, dtype=np.float32)

    if equation == "heat":
        X_plot, T_plot = np.meshgrid(x_plot, t_plot)
        z_true = heat_exact_solution(X_plot, T_plot)
    else:
        u_true = burgers_reference_solution(x_plot, t_plot)
        z_true = u_true.T

    z_abs_max = float(np.abs(z_true).max()) or 1.0
    true_fig = go.Figure(data=go.Heatmap(
        x=x_plot, y=t_plot, z=z_true,
        colorscale="RdBu", zmid=0, zmin=-z_abs_max, zmax=z_abs_max,
        colorbar=dict(title="u(x,t)", tickfont=dict(size=10)),
    ))
    _apply_dark_layout(true_fig, "Ground-Truth Solution", "x", "t")
    true_fig.update_layout(paper_bgcolor="#141929", plot_bgcolor="#141929")

    return ic_fig, pts_fig, true_fig, eq_latex, latex

@callback(
    Output({"type": "train-btn", "method": MATCH}, "disabled"),
    Output({"type": "train-status", "method": MATCH}, "children"),
    Output({"type": "train-progress", "method": MATCH}, "value"),
    Output({"type": "train-progress", "method": MATCH}, "animated"),
    Output({"type": "loss-interval", "method": MATCH}, "disabled"),
    Input({"type": "train-btn", "method": MATCH}, "n_clicks"),
    Input({"type": "clear-btn", "method": MATCH}, "n_clicks"),
    Input({"type": "loss-interval", "method": MATCH}, "n_intervals"),
    State({"type": "eq-select", "method": MATCH}, "value"),
    State({"type": "n-x", "method": MATCH}, "value"),
    State({"type": "n-t", "method": MATCH}, "value"),
    State({"type": "epochs", "method": MATCH}, "value"),
    State({"type": "lr", "method": MATCH}, "value"),
    State({"type": "strategy-select", "method": MATCH}, "data"),
    State({"type": "learning-type-select", "method": MATCH}, "data"),
    State({"type": "adaptive-toggles", "method": MATCH}, "value"),
)
def handle_train(n_clicks, clear_clicks, n_intervals, equation, n_x, n_t, epochs, lr, strategy, learning_type, adaptive_toggles):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, _status_badge("idle"), 0, False, True

    method_id = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["method"]
    data = _read_loss_history(method_id)

    if '"clear-btn"' in ctx.triggered[0]["prop_id"] and clear_clicks:
        _write_loss_history({"status": "idle", "history": []}, method_id)
        return False, _status_badge("idle"), 0, False, True

    if '"train-btn"' in ctx.triggered[0]["prop_id"] and n_clicks > 0:
        if data.get("status") == "training":
            return True, _status_badge("training"), _progress(data), True, False
        equation = equation or "heat"
        n_x = int(n_x or DEFAULT_N_X)
        n_t = int(n_t or DEFAULT_N_T)
        epochs = int(epochs or DEFAULT_EPOCHS)
        lr = float(lr or DEFAULT_LR)
        threading.Thread(
            target=_run_training,
            args=(method_id, equation, n_x, n_t, epochs, lr, strategy, learning_type, adaptive_toggles),
            daemon=True,
        ).start()
        return True, _status_badge("training"), 0, True, False

    if '"loss-interval"' in ctx.triggered[0]["prop_id"]:
        status = data.get("status", "idle")
        if status == "complete":
            weights_path = data.get("best_weights_path")
            label = f"Saved: {Path(weights_path).name}" if weights_path else "Complete"
            return False, _status_badge("complete", label), 100, False, True
        if status == "error":
            return False, _status_badge("error", data.get("error", "Unknown")), 0, False, True
        if status == "training":
            return True, _status_badge("training"), _progress(data), True, False

    return False, _status_badge("idle"), 0, False, True

def _progress(data: dict) -> int:
    ep = data.get("current_epoch", 0)
    total = data.get("total_epochs", 1) or 1
    return min(100, int(100 * ep / total))

def _status_badge(status: str, extra: str = "") -> html.Span:
    config = {
        "idle":     ("Idle",     "secondary"),
        "training": ("Training…","warning"),
        "complete": ("Complete", "success"),
        "error":    ("Error",    "danger"),
    }
    label, color = config.get(status, ("Unknown", "secondary"))
    display = f"{label}  {extra}" if extra else label
    return dbc.Badge(display, color=color, className="status-badge")

@callback(
    Output({"type": "loss-plot", "method": MATCH}, "figure"),
    Input({"type": "loss-interval", "method": MATCH}, "n_intervals"),
    Input({"type": "train-btn", "method": MATCH}, "n_clicks"),
    Input({"type": "clear-btn", "method": MATCH}, "n_clicks"),
)
def update_loss_plot(n_intervals, train_clicks, clear_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return _empty_fig()
    method_id = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["method"]
    data = _read_loss_history(method_id)
    history = data.get("history", [])

    fig = _empty_fig()
    if history:
        epochs_list = [h["epoch"] for h in history]
        if "train_loss" in history[0]:
            train_losses = [h["train_loss"] for h in history]
            val_losses = [h["val_loss"] for h in history]
        else:
            train_losses = [h["loss"] for h in history]
            val_losses = None

        fig.add_trace(go.Scatter(
            x=epochs_list, y=train_losses,
            mode="lines", name="Train Loss",
            line=dict(color="#7eb3ff", width=2),
        ))
        if val_losses:
            fig.add_trace(go.Scatter(
                x=epochs_list, y=val_losses,
                mode="lines", name="Val Loss",
                line=dict(color="#f59e0b", width=2, dash="dash"),
            ))

            fig.add_trace(go.Scatter(
                x=epochs_list + epochs_list[::-1],
                y=val_losses + train_losses[::-1],
                fill="toself", fillcolor="rgba(245,158,11,0.07)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ))

        if val_losses:
            best_idx = int(np.argmin(val_losses))
            fig.add_vline(
                x=epochs_list[best_idx], line_dash="dot",
                line_color="rgba(100,220,100,0.5)",
                annotation_text=f"best ep={epochs_list[best_idx]}",
                annotation_font_size=10, annotation_font_color="#80cc80",
            )

    _apply_dark_layout(fig, "Loss vs Epoch", "Epoch", "Loss (log)")
    fig.update_layout(yaxis_type="log")

    return fig

@callback(
    Output({"type": "weights-select", "method": MATCH}, "options"),
    Output({"type": "weights-select", "method": MATCH}, "value"),
    Output({"type": "delete-status", "method": MATCH}, "children"),
    Input({"type": "eq-select", "method": MATCH}, "value"),
    Input({"type": "strategy-select", "method": MATCH}, "data"),
    Input({"type": "learning-type-select", "method": MATCH}, "data"),
    Input({"type": "delete-models-btn", "method": MATCH}, "n_clicks"),
    Input({"type": "refresh-models-btn", "method": MATCH}, "n_clicks"),
)
def update_weights_dropdown(equation, strategy, learning_type, delete_clicks, refresh_clicks):
    from weights_utils import WEIGHTS_DIR
    ctx = dash.callback_context
    status_msg = ""

    # Handle delete button
    triggered_ids = [t["prop_id"] for t in ctx.triggered] if ctx.triggered else []
    if any('"delete-models-btn"' in tid for tid in triggered_ids) and delete_clicks:
        deleted = 0
        for entry in WEIGHTS_DIR.iterdir():
            try:
                if entry.is_dir():
                    shutil.rmtree(entry)
                elif entry.is_file():
                    entry.unlink()
                deleted += 1
            except Exception as e:
                status_msg = f"Error: {e}"
        if not status_msg:
            status_msg = f"Deleted {deleted} model(s)."
        return [], None, status_msg

    if not equation:
        return [], None, status_msg
    paths = list_weights(equation)
    
    filtered_paths = []
    for p in paths:
        meta = load_metadata(p)
        match_strat = (not strategy) or (strategy == "any") or (meta.get("strategy") == strategy)
        match_type = (not learning_type) or (learning_type == "any") or (meta.get("learning_type") == learning_type)
        if match_strat and match_type:
            filtered_paths.append(p)

    opts = [{"label": Path(p).name, "value": p} for p in filtered_paths]
    val = opts[0]["value"] if opts else None
    return opts, val, status_msg

@callback(
    Output({"type": "compare-plot", "method": MATCH}, "figure"),
    Output({"type": "model-meta-card", "method": MATCH}, "children"),
    Input({"type": "eq-select", "method": MATCH}, "value"),
    Input({"type": "weights-select", "method": MATCH}, "value"),
)
def update_compare_plot(equation: str | None, weights_path: str | None):
    if not equation or not weights_path:
        fig = _empty_fig("Select a model and weights file to compare")
        return fig, None

    meta = load_metadata(weights_path)
    if equation == "heat":
        num_hidden = meta.get("num_hidden", HEAT_NUM_HIDDEN)
        dim_hidden = meta.get("dim_hidden", HEAT_DIM_HIDDEN)
        x_domain, t_domain = HEAT_X_DOMAIN, HEAT_T_DOMAIN
    else:
        num_hidden = meta.get("num_hidden", BURGERS_NUM_HIDDEN)
        dim_hidden = meta.get("dim_hidden", BURGERS_DIM_HIDDEN)
        x_domain, t_domain = BURGERS_X_DOMAIN, BURGERS_T_DOMAIN

    n_x, n_t = 120, 120
    x_np = np.linspace(x_domain[0], x_domain[1], n_x, dtype=np.float32)
    t_np = np.linspace(t_domain[0], t_domain[1], n_t, dtype=np.float32)

    if equation == "heat":
        X, T = np.meshgrid(x_np, t_np)
        u_ref = heat_exact_solution(X, T)
    else:
        u_ref = burgers_reference_solution(x_np, t_np)

    X_mg, T_mg = np.meshgrid(x_np, t_np)
    x_tensor = torch.tensor(X_mg.flatten().astype(np.float32)).reshape(-1, 1).to(device)
    t_tensor = torch.tensor(T_mg.flatten().astype(np.float32)).reshape(-1, 1).to(device)

    model = PINN(num_hidden, dim_hidden).to(device)
    if not Path(weights_path).exists():
        fig = _empty_fig(f"Weights file not found: {Path(weights_path).name}")
        return fig, html.Div("Selected weights file is missing.", style={"color": "#ef4444"})

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    with torch.no_grad():
        u_ml = model(x_tensor, t_tensor).cpu().numpy().reshape(n_t, n_x)

    u_ref_2d = u_ref if equation == "heat" else u_ref.T
    u_err = np.abs(u_ml - u_ref_2d)

    z_abs_max = float(max(np.abs(u_ref_2d).max(), np.abs(u_ml).max())) or 1.0

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Ground Truth", "PINN Model", "Absolute Error |GT − ML|"),
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Heatmap(x=x_np, y=t_np, z=u_ref_2d, coloraxis="coloraxis", name="Ground Truth"), row=1, col=1)
    fig.add_trace(go.Heatmap(x=x_np, y=t_np, z=u_ml, coloraxis="coloraxis", name="PINN Model"), row=1, col=2)
    fig.add_trace(go.Heatmap(x=x_np, y=t_np, z=u_err, coloraxis="coloraxis2", name="|Error|"), row=1, col=3)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#141929", plot_bgcolor="#0d1120",
        font=dict(color="#a0b4d8"),
        margin=dict(l=40, r=30, t=55, b=40),
        height=320,
        coloraxis=dict(
            colorscale="RdBu", cmid=0, cmin=-z_abs_max, cmax=z_abs_max,
            colorbar=dict(title=dict(text="u", side="right"), thickness=14, len=0.85, x=0.64, tickfont=dict(size=9)),
        ),
        coloraxis2=dict(
            colorscale="YlOrRd", cmin=0, cmax=float(u_err.max()) or 1.0,
            colorbar=dict(title=dict(text="|err|", side="right"), thickness=14, len=0.85, x=1.01, tickfont=dict(size=9)),
        ),
    )
    for col in [1, 2, 3]:
        fig.update_xaxes(title_text="x", row=1, col=col)
        fig.update_yaxes(title_text="t", row=1, col=col)

    best_val = meta.get("best_val_loss")
    best_ep = meta.get("best_epoch")
    strategy = meta.get("strategy", "standard")
    l_type = meta.get("learning_type", "standard")
    adaptive = meta.get("adaptive_options", [])
    adaptive_str = ", ".join(adaptive) if adaptive else "None"

    meta_card = html.Div(
        [
            dbc.Row([
                dbc.Col([html.Div("Strategy", className="control-label"), html.Div(f"{strategy.upper()}", style={"color": "#7eb3ff", "fontWeight": "600", "fontSize": "0.85rem"})], width=3),
                dbc.Col([html.Div("Type", className="control-label"), html.Div(f"{l_type.upper()}", style={"color": "#f59e0b", "fontWeight": "600", "fontSize": "0.85rem"})], width=3),
                dbc.Col([html.Div("Adaptive", className="control-label"), html.Div(f"{adaptive_str}", style={"color": "#a0b4d8", "fontSize": "0.75rem"})], width=3),
                dbc.Col([
                    html.Div("Best Epoch", className="control-label"), html.Div(f"{best_ep or '—'}", style={"color": "#7eb3ff", "fontWeight": "600"})
                ], width=1),
                dbc.Col([
                    html.Div("Best Val Loss", className="control-label"), html.Div(f"{best_val:.2e}" if best_val else "—", style={"color": "#f59e0b", "fontWeight": "600"}),
                ], width=2),
            ]),
        ],
        className="p-2 mb-2",
        style={"background": "#141929", "borderRadius": "8px", "border": "1px solid #2a3555"},
    )

    return fig, meta_card

if __name__ == "__main__":
    app.run(
        debug=True,
        port=8050,
        dev_tools_hot_reload=True,
        dev_tools_hot_reload_interval=1500,
        dev_tools_hot_reload_watch_interval=1.5
    )
