"""Neural ODE Dash interface.

Run with: uv run neural_ode_dash_app.py (from NeuralODE dir) or python neural_ode_dash_app.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, MATCH
import numpy as np
import plotly.graph_objects as go
import torch

from neural_ode_core import NeuralODE, NeuralODEClassifier, device
from spiral_example import get_spiral_data
from classifier_example import get_concentric_circles

DEFAULT_EPOCHS = 1000
DEFAULT_LR = 0.01

PLOTLY_TEMPLATE = "plotly_dark"

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    assets_folder=str(Path(__file__).resolve().parent / "assets"),
)

# ── LaTeX strings ─────────────────────────────────────────────────────────────
SPIRAL_EQUATION_LATEX = r"""
**Spiral Dynamics Inverse Problem**

$$
\frac{d}{dt} \mathbf{x}(t) = \mathbf{A}(\mathbf{x}(t))^3 
$$

Goal: Learn $\mathbf{f}_{\theta}(\mathbf{x}) \approx \mathbf{A}\mathbf{x}^3$ by training a neural network over sequential data points.
"""

CLASSIFIER_EQUATION_LATEX = r"""
**Continuous-Depth Classifier**

$$
\mathbf{h}(0) = \text{Encoder}(\mathbf{x})
$$

$$
\frac{d \mathbf{h}(t)}{dt} = \mathbf{f}_{\theta}(\mathbf{h}(t), t) \quad \text{for} \quad t \in [0, 1]
$$

$$
\hat{y} = \text{Decoder}(\mathbf{h}(1))
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

def _apply_dark_layout(fig: go.Figure, title: str = "", xaxis_title: str = "", yaxis_title: str = "", equal_aspect: bool = False) -> None:
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
    if equal_aspect:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

# ── Training backend (file I/O helpers imported from train_worker) ────────────
from train_worker import _path as _get_loss_history_path, _write as _write_loss_history, _read as _read_loss_history

_WORKER = str(Path(__file__).resolve().parent / "train_worker.py")
_PYTHON = sys.executable

# ── Weights helpers ─────────────────────────────────────────────────────────
_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

def _list_weights(task: str) -> list[str]:
    """Return .pt paths for a given task, newest first."""
    _WEIGHTS_DIR.mkdir(exist_ok=True)
    pts = [p for p in _WEIGHTS_DIR.rglob("*.pt") if p.name.startswith(task + "_")]
    pts.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return [str(p) for p in pts]

def _load_meta(pt_path: str) -> dict:
    p = Path(pt_path).with_suffix(".json")
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


# ── Pre-load training data once (avoid blocking Flask on every interval) ─────
# We load clean for the 'clean' trajectory line and noisy for training markers
_t_spiral, _true_y_spiral_clean = get_spiral_data(noise_std=0.0)
_, _true_y_spiral = get_spiral_data(noise_std=0.05)
_X_circles, _y_circles = get_concentric_circles()

# ── Layout components ─────────────────────────────────────────────────────────
def _section_label(text: str) -> html.Div:
    return html.Div(text, className="section-header")

_graph_config = {"responsive": True, "displayModeBar": False}

def _plot_card(title: str, graph_id: dict) -> html.Div:
    return html.Div(
        [
            html.Div(title, className="plot-title"),
            dcc.Graph(id=graph_id, config=_graph_config, style={"height": "560px"}),
        ],
        className="plot-card mb-3",
    )

METHODS_CONFIG = [
    {
        "id": "spiral",
        "name": "Dynamical System Discovery",
        "task": "spiral",
        "epochs": 1000, "lr": 0.01,
        "description": "Learn a continuous-time 2D spiral by regressing the vector field using a Neural ODE over sampled trajectories."
    },
    {
        "id": "classifier",
        "name": "Continuous-Depth Classifier",
        "task": "classifier",
        "epochs": 500, "lr": 0.05,
        "description": "Solve a classification problem across continuous layers (resnets approaching infinite depth). A latent state is evolved by the continuous vector field, separated, and read out by a dense layer."
    }
]

def create_method_tab(config: dict) -> dbc.Container:
    method_id = config["id"]
    sidebar = html.Div([
        _section_label("Configuration"),
        dcc.Markdown(f"**{config['name']}**\n{config['description']}", mathjax=True, style={"fontSize": "13px", "color": "#a0b4d8", "marginBottom": "15px"}),
        
        _section_label("Task Setup"),
        dcc.Store(id={"type": "task-select", "method": method_id}, data=config["task"]),
        dcc.Markdown(SPIRAL_EQUATION_LATEX if config["task"] == "spiral" else CLASSIFIER_EQUATION_LATEX, mathjax=True, style={"fontSize": "14px", "minHeight": "60px"}),
        
        # Noise Slider (always present to avoid MATCH callback errors)
        html.Div([
            html.Div("Data Noise", className="control-label mt-2"),
            dcc.Slider(
                id={"type": "noise-std", "method": method_id}, min=0.0, max=0.2, step=0.01, value=0.05,
                marks={0: "0", 0.1: "0.1", 0.2: "0.2"},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Div("Batch Time (Segment Length)", className="control-label mt-2"),
            dcc.Slider(
                id={"type": "batch-time", "method": method_id}, min=2, max=100, step=1, value=20,
                marks={2: "2", 20: "20", 50: "50", 100: "100"},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ], id={"type": "noise-container", "method": method_id}),
        
        html.Hr(style={"borderColor": "#2a3555", "margin": "12px 0"}),

        _section_label("Solver"),
        dcc.Dropdown(
            id={"type": "solver-select", "method": method_id},
            options=[
                {"label": "DOPRI5 (Adaptive RK45)", "value": "dopri5"},
                {"label": "RK4 (Fixed Step)", "value": "rk4"},
                {"label": "Euler (Fixed Step)", "value": "euler"},
            ],
            value="dopri5",
            clearable=False,
            className="dash-dropdown",
            style={"marginBottom": "8px"},
        ),
        html.Hr(style={"borderColor": "#2a3555", "margin": "12px 0"}),

        _section_label("Optimizer"),
        html.Div("Epochs", className="control-label"),
        dcc.Slider(
            id={"type": "epochs", "method": method_id}, min=100, max=5000, step=100, value=config["epochs"],
            marks={i: {"label": str(i), "style": {"color": "#5577aa", "fontSize": "10px"}} for i in [100, 1000, 2500, 5000]},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div("Learning Rate", className="control-label mt-2"),
        dcc.Input(
            id={"type": "lr", "method": method_id}, type="number", value=config["lr"], step=0.001, min=0.0001,
            className="w-100", style={"background": "#1e2842", "border": "1px solid #2a3555", "color": "#e8eaf0", "borderRadius": "6px", "padding": "6px 10px", "marginBottom": "12px"},
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
        html.Div("Compare Models", className="section-header"),
        dcc.Dropdown(
            id={"type": "weights-select", "method": method_id},
            options=[], value=None,
            placeholder="Select saved run…",
            className="dash-dropdown",
            style={"marginBottom": "8px"},
        ),
        dbc.Button(
            "🔄 Refresh List",
            id={"type": "refresh-models-btn", "method": method_id},
            n_clicks=0, color="secondary", outline=True, size="sm",
            className="w-100", style={"fontSize": "0.75rem"},
        ),
        dbc.Button(
            "🗑 Delete All Models",
            id={"type": "delete-models-btn", "method": method_id},
            n_clicks=0, color="danger", outline=True, size="sm",
            className="w-100 mt-1",
            style={"fontSize": "0.75rem", "borderColor": "#7f1d1d", "color": "#fca5a5"},
        ),
        html.Div(id={"type": "delete-status", "method": method_id},
                 style={"fontSize": "0.72rem", "color": "#fca5a5", "marginTop": "4px", "minHeight": "18px"}),
    ], className="pinn-sidebar p-3", style={"height": "100%", "overflowY": "auto", "minWidth": "300px"})

    return dbc.Container([
        dbc.Row([
            dbc.Col(sidebar, width=3, style={"minWidth": "300px"}),
            dbc.Col([
                html.H4("Training Plots", className="section-header mt-2"),
                dbc.Row([
                    dbc.Col(_plot_card("Loss vs Epoch", {"type": "loss-plot", "method": method_id}), width=6),
                    dbc.Col(_plot_card("Predictions Overview", {"type": "pred-plot", "method": method_id}), width=6),
                ]),
                html.Hr(style={"borderColor": "#2a3555", "margin": "12px 0"}),
                html.H4("Model Comparison", className="section-header"),
                _plot_card("Saved Model vs Target", {"type": "compare-plot", "method": method_id}),
            ], width=9)
        ], className="mt-2 g-3"),
        dcc.Store(id={"type": "loss-store", "method": method_id}, data={"history": []}),
        dcc.Interval(id={"type": "loss-interval", "method": method_id}, interval=500, n_intervals=0, disabled=True),
    ], fluid=True, className="mt-3")

THEORY_MARKDOWN = r"""
### Neural Ordinary Differential Equations

Neural ODEs extend classical sequence modeling architectures (like ResNets and RNNs) into a continuous domain. Instead of specifying a discrete sequence of hidden layers, a Neural ODE parameterizes the derivative of the hidden state using a neural network and evaluates it using a black-box differential equation solver.

$$
\frac{dh(t)}{dt} = f_{\theta}(h(t), t)
$$

This allows continuous-time dynamics to be modeled directly, bringing significant benefits such as adaptive computation, continuous time interpolation, and reversibility, which ultimately uses less memory for forward-backward passes by using the adjoint method.

---

### Adjoint Method

To optimize the model, instead of backpropagating completely through the solver steps (which scales linearly in memory with depth), the adjoint method solves an augmented continuous-time ODE backwards in time. This provides constant memory requirements, enabling much heavier models to be optimized effectively.

---

### Dynamical System Modeling

Neural ODEs are superb for regressing physical state representations given trajectory arrays over time, even with irregular sampling gaps. We showcase this with the **Spiral Classifier Tool** capable of uncovering a cubic nonlinear relationship creating a 2D spiral state map.
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
        html.Div(dcc.Markdown("$$ $$", mathjax=True), style={"display": "none"}),
        html.Div(
            [
                html.Span("⚛", style={"fontSize": "1.6rem", "marginRight": "0.5rem"}),
                html.Span("Neural ODE Explorer", className="pinn-title", style={"color": "#ffaa77", "fontWeight": "bold", "fontSize": "1.4rem"}),
                html.Span("Continuous-Depth Architectures", className="pinn-subtitle", style={"color": "#a0b4d8", "marginLeft": "15px"}),
            ],
            className="pinn-navbar d-flex align-items-center mb-3",
        ),
        dbc.Container(
            dbc.Tabs(
                id="tabs",
                active_tab="spiral",
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
    Output({"type": "train-btn", "method": MATCH}, "disabled"),
    Output({"type": "train-status", "method": MATCH}, "children"),
    Output({"type": "train-progress", "method": MATCH}, "value"),
    Output({"type": "train-progress", "method": MATCH}, "animated"),
    Output({"type": "loss-interval", "method": MATCH}, "disabled"),
    Input({"type": "train-btn", "method": MATCH}, "n_clicks"),
    Input({"type": "clear-btn", "method": MATCH}, "n_clicks"),
    Input({"type": "loss-interval", "method": MATCH}, "n_intervals"),
    State({"type": "task-select", "method": MATCH}, "data"),
    State({"type": "epochs", "method": MATCH}, "value"),
    State({"type": "lr", "method": MATCH}, "value"),
    State({"type": "solver-select", "method": MATCH}, "value"),
    State({"type": "noise-std", "method": MATCH}, "value"),
    State({"type": "batch-time", "method": MATCH}, "value"),
)
def handle_train(n_clicks, clear_clicks, n_intervals, task, epochs, lr, solver, noise_std, batch_time):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, _status_badge("idle"), 0, False, True

    method_id = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["method"]
    data = _read_loss_history(method_id)

    if '"clear-btn"' in ctx.triggered[0]["prop_id"] and clear_clicks:
        _write_loss_history({"status": "idle", "history": []}, method_id)
        return False, _status_badge("idle"), 0, False, True

    if '"train-btn"' in ctx.triggered[0]["prop_id"] and n_clicks > 0:
        # Always restart: don't check existing status so stale files never block.
        noise_std = float(noise_std or 0.0)
        batch_time = int(batch_time or 20)
        # Write 'training' BEFORE launching subprocess
        _write_loss_history({"status": "training", "history": [], "current_epoch": 0, "total_epochs": epochs}, method_id)
        subprocess.Popen(
            [_PYTHON, _WORKER, method_id, task, str(epochs), str(lr), solver, str(noise_std), str(batch_time)],
            close_fds=True,
        )
        return True, _status_badge("training"), 0, True, False

    if '"loss-interval"' in ctx.triggered[0]["prop_id"]:
        status = data.get("status", "idle")
        if status == "complete":
            return False, _status_badge("complete", "Done"), 100, False, True
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
    Output({"type": "pred-plot", "method": MATCH}, "figure"),
    Input({"type": "loss-interval", "method": MATCH}, "n_intervals"),
    Input({"type": "clear-btn", "method": MATCH}, "n_clicks"),
    Input({"type": "noise-std", "method": MATCH}, "value"),
    State({"type": "task-select", "method": MATCH}, "data"),
)
def update_plots(n_intervals, clear_clicks, noise_std, task):
    ctx = dash.callback_context
    # In a MATCH callback, we can extract the method_id from the Output component
    # This works even on initial load when ctx.triggered is empty.
    try:
        method_id = ctx.outputs_list[0]['id']['method']
    except (IndexError, KeyError):
        method_id = None
    
    data = _read_loss_history(method_id)
    history = data.get("history", [])

    loss_fig = _empty_fig()
    if history:
        epochs_list = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        loss_fig.add_trace(go.Scatter(
            x=epochs_list, y=train_losses,
            mode="lines", name="Train Loss",
            line=dict(color="#7eb3ff", width=2),
        ))
    _apply_dark_layout(loss_fig, "Training Loss", "Epoch", "Loss")
    if history and len(train_losses) > 0 and min(train_losses) > 0:
        loss_fig.update_layout(yaxis_type="log")

    pred_fig = _empty_fig()
    if task == "spiral":
        true_y_clean = _true_y_spiral_clean.squeeze(1).cpu().numpy()
        pred_y = data.get("pred_y")
        
        # Generate noisy points on-the-fly for interactive visualization
        noise_val = float(noise_std or 0.0)
        # Using a fixed seed for the visualization noise so it doesn't flicker 
        # too much, but still updates when noise_val changes.
        np.random.seed(42) 
        noisy_pts = true_y_clean + np.random.normal(0, noise_val, true_y_clean.shape)

        # Original clean trajectory (dashed)
        pred_fig.add_trace(go.Scatter(
            x=true_y_clean[:, 0], y=true_y_clean[:, 1],
            mode="lines", name="Target (Clean)",
            line=dict(color="#f59e0b", width=1, dash="dash"),
            opacity=0.5
        ))

        # Noisy Training Data (sampled)
        pred_fig.add_trace(go.Scatter(
            x=noisy_pts[::5, 0], y=noisy_pts[::5, 1],
            mode="markers", name=f"Training Data (σ={noise_val})",
            marker=dict(color="#4ade80", size=3, opacity=0.8),
        ))
        
        if pred_y:
            pred_y_np = np.array(pred_y).squeeze(1)
            pred_fig.add_trace(go.Scatter(
                x=pred_y_np[:, 0], y=pred_y_np[:, 1],
                mode="lines", name="Prediction",
                line=dict(color="#7eb3ff", width=2),
            ))
        _apply_dark_layout(pred_fig, "Spiral Trajectory", "x1", "x2", equal_aspect=True)

    elif task == "classifier":
        # Dynamic noise for classifier visualization
        noise_val = float(noise_std or 0.0)
        # Note: we should ideally re-call get_concentric_circles here if we want perfectly interactive dots 
        # but for simplicity we'll just show the pre-loaded ones or re-generate.
        # Re-generating is actually fast enough for 1000 points.
        X_torch, y_torch = get_concentric_circles(noise=noise_val)
        X_np, y_np = X_torch.cpu().numpy(), y_torch.cpu().numpy()
        pred_y = data.get("pred_y")

        # Labels plot
        if pred_y:
            pred_y_np = np.array(pred_y).flatten()
            colors = ["#f59e0b" if prob < 0.0 else "#7eb3ff" for prob in pred_y_np]
            title = f"Predictions (Boundary at 0.0, σ={noise_val})"
        else:
            colors = ["#f59e0b" if label == 0 else "#7eb3ff" for label in y_np.flatten()]
            title = f"True Labels (0 vs 1, σ={noise_val})"
            
        pred_fig.add_trace(go.Scatter(
            x=X_np[:, 0], y=X_np[:, 1],
            marker=dict(color=colors, size=6, opacity=0.8),
            mode="markers", name="Data",
        ))
        _apply_dark_layout(pred_fig, title, "x1", "x2", equal_aspect=True)

    return loss_fig, pred_fig


# ── Compare Models callbacks ──────────────────────────────────────────────────
@callback(
    Output({"type": "weights-select", "method": MATCH}, "options"),
    Output({"type": "weights-select", "method": MATCH}, "value"),
    Output({"type": "delete-status", "method": MATCH}, "children"),
    Input({"type": "task-select", "method": MATCH}, "data"),
    Input({"type": "refresh-models-btn", "method": MATCH}, "n_clicks"),
    Input({"type": "delete-models-btn", "method": MATCH}, "n_clicks"),
)
def update_node_weights_dropdown(task, refresh_clicks, delete_clicks):
    ctx = dash.callback_context
    status_msg = ""
    triggered = [t["prop_id"] for t in ctx.triggered] if ctx.triggered else []

    if any('"delete-models-btn"' in tid for tid in triggered) and delete_clicks:
        deleted = 0
        if _WEIGHTS_DIR.exists():
            for entry in _WEIGHTS_DIR.iterdir():
                try:
                    if entry.is_dir():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()
                    deleted += 1
                except Exception as e:
                    status_msg = f"Error: {e}"
        if not status_msg:
            status_msg = f"Deleted {deleted} model(s)."
        return [], None, status_msg

    if not task:
        return [], None, status_msg

    paths = _list_weights(task)
    opts = [{"label": Path(p).parent.name, "value": p} for p in paths]
    val = opts[0]["value"] if opts else None
    return opts, val, status_msg


@callback(
    Output({"type": "compare-plot", "method": MATCH}, "figure"),
    Input({"type": "weights-select", "method": MATCH}, "value"),
    State({"type": "task-select", "method": MATCH}, "data"),
)
def update_node_compare_plot(weights_path: str | None, task: str | None):
    if not weights_path or not Path(weights_path).exists():
        return _empty_fig("Select a saved run to compare")

    meta = _load_meta(weights_path)
    solver = meta.get("solver", "dopri5")
    fig = _empty_fig()

    try:
        if task == "spiral":
            model = NeuralODE(in_dim=2, num_hidden=64, method=solver).to(device)
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.eval()

            true_y_clean = _true_y_spiral_clean.squeeze(1).cpu().numpy()
            true_y_noisy = _true_y_spiral.squeeze(1).cpu().numpy()

            with torch.no_grad():
                # Predict starting from the clean initial point
                pred = model(_true_y_spiral_clean[0], _t_spiral).cpu().numpy()

            fig.add_trace(go.Scatter(
                x=true_y_clean[:, 0], y=true_y_clean[:, 1],
                mode="lines", name="Target (Clean)",
                line=dict(color="#f59e0b", width=1, dash="dash"),
                opacity=0.5
            ))
            fig.add_trace(go.Scatter(
                x=true_y_noisy[::5, 0], y=true_y_noisy[::5, 1],
                mode="markers", name="Training Data (Noisy)",
                marker=dict(color="#4ade80", size=3, opacity=0.8),
            ))
            fig.add_trace(go.Scatter(
                x=pred[:, 0, 0], y=pred[:, 0, 1],
                mode="lines", name="Saved Model",
                line=dict(color="#7eb3ff", width=2),
            ))
            ep = meta.get("epochs", "?")
            loss = meta.get("final_loss", 0)
            fname = Path(weights_path).name
            _apply_dark_layout(fig, f"Spiral: {fname} | ep={ep} loss={loss:.4f}", "x1", "x2", equal_aspect=True)

        elif task == "classifier":
            model = NeuralODEClassifier(in_dim=2, hidden_dim=16, num_classes=1, method=solver).to(device)
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.eval()

            X_np, y_np = _X_circles.cpu().numpy(), _y_circles.cpu().numpy()

            with torch.no_grad():
                out = model(X).cpu().numpy().flatten()

            colors = ["#f59e0b" if p < 0.0 else "#7eb3ff" for p in out]
            fig.add_trace(go.Scatter(
                x=X_np[:, 0], y=X_np[:, 1],
                mode="markers", name="Predictions",
                marker=dict(color=colors, size=6, opacity=0.8),
            ))
            ep = meta.get("epochs", "?")
            loss = meta.get("final_loss", 0)
            fname = Path(weights_path).name
            _apply_dark_layout(fig, f"Classifier: {fname} | ep={ep} loss={loss:.4f}", "x1", "x2", equal_aspect=True)

    except Exception as e:
        fig = _empty_fig(f"Error loading model: {e}")

    return fig


if __name__ == "__main__":
    app.run(debug=True, port=8052, use_reloader=True)
