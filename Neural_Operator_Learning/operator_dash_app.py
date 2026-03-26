"""Dash interface for GPU-only neural operator learning demos."""
from __future__ import annotations

from datetime import datetime
import json
import shutil
import threading
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, MATCH
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import numpy as np
import torch

from operator_data import build_grid, evaluate_sine_series, make_dataset, solve_poisson_from_coeffs
from operator_models import (
    DeepONet,
    FNO1d,
    predict_deeponet,
    predict_fno,
    require_cuda_device,
    summarize_prediction,
    train_fno,
)
from training import train_deeponet


device = require_cuda_device()
BASE_DIR = Path(__file__).resolve().parent
_history_lock = threading.Lock()
_graph_config = {"responsive": True, "displayModeBar": False}
DEEPNET_WEIGHTS_DIR = BASE_DIR / "weights" / "deeponet"
DEEPNET_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
FNO_WEIGHTS_DIR = BASE_DIR / "weights" / "fno"
FNO_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


METHODS = [
    {
        "id": "deeponet",
        "name": "DeepONet",
        "subtitle": "Branch/trunk architecture for operator queries",
        "description": "DeepONet learns an operator by combining a branch encoding of the input function with a trunk encoding of the query location x.",
        "defaults": {"n_train": 128, "epochs": 400, "lr": 1e-3, "n_points": 64, "n_modes": 6},
        "equation": r"$$-u''(x)=f(x), \quad x\in(0,1), \quad u(0)=u(1)=0$$",
        "architecture": r"$$u_\theta(f)(x)=\sum_{k=1}^p b_k(f)\,t_k(x)+b_0$$",
    },
    {
        "id": "fno",
        "name": "Fourier Neural Operator",
        "subtitle": "Global convolution in Fourier space",
        "description": "FNO updates feature maps on the full discretized function and mixes information globally through truncated Fourier modes.",
        "defaults": {"n_train": 512, "epochs": 400, "lr": 1e-3, "n_points": 64, "n_modes": 6},
        "equation": r"$$-u''(x)=f(x), \quad x\in(0,1), \quad u(0)=u(1)=0$$",
        "architecture": r"$$v_{l+1}(x)=\sigma\left(Wv_l(x)+\mathcal{F}^{-1}(R_\phi\cdot\mathcal{F}(v_l))(x)\right)$$",
    },
]

FORCING_PROFILES = [
    ("mode1", "Additional: sin(pi x)"),
    ("mode2", "Additional: sin(2pi x)"),
    ("mixed_low", "Additional: mixed low modes"),
    ("mixed_high", "Additional: mixed higher modes"),
    ("random_101", "Additional random (seed=101)"),
    ("random_202", "Additional random (seed=202)"),
]


def _history_path(method_id: str) -> Path:
    return BASE_DIR / f"training_history_{method_id}.json"


def _read_history(method_id: str) -> dict:
    path = _history_path(method_id)
    if not path.exists():
        return {"status": "idle", "history": []}
    with _history_lock:
        try:
            with open(path) as stream:
                return json.load(stream)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"status": "idle", "history": []}


def _write_history(method_id: str, data: dict) -> None:
    with _history_lock:
        with open(_history_path(method_id), "w") as stream:
            json.dump(data, stream, indent=2)


def _create_deeponet_run_dir(config: dict) -> Path:
    """Create a timestamped run directory for DeepONet checkpoints."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"deeponet_{stamp}_n{config['n_train']}_p{config['n_points']}_m{config['n_modes']}"
    run_dir = DEEPNET_WEIGHTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _create_fno_run_dir(config: dict) -> Path:
    """Create a timestamped run directory for FNO checkpoints."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"fno_{stamp}_n{config['n_train']}_p{config['n_points']}_m{config['n_modes']}"
    run_dir = FNO_WEIGHTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_deeponet_best(model: DeepONet, run_dir: Path, metadata: dict) -> tuple[str, str]:
    """Overwrite best checkpoint in a run directory and save metadata."""
    weights_path = run_dir / "best_model.pt"
    meta_path = run_dir / "best_model.json"
    torch.save(model.state_dict(), weights_path)
    payload = dict(metadata)
    payload["weights_path"] = str(weights_path)
    payload["metadata_path"] = str(meta_path)
    with open(meta_path, "w") as stream:
        json.dump(payload, stream, indent=2)
    return str(weights_path), str(meta_path)


def _save_fno_model(model: FNO1d, run_dir: Path, metadata: dict) -> tuple[str, str]:
    """Save final FNO checkpoint and metadata."""
    weights_path = run_dir / "final_model.pt"
    meta_path = run_dir / "final_model.json"
    torch.save(model.state_dict(), weights_path)
    payload = dict(metadata)
    payload["weights_path"] = str(weights_path)
    payload["metadata_path"] = str(meta_path)
    with open(meta_path, "w") as stream:
        json.dump(payload, stream, indent=2)
    return str(weights_path), str(meta_path)


def _list_deeponet_models() -> list[dict]:
    """List available saved DeepONet checkpoints (newest first)."""
    metadata_files = sorted(DEEPNET_WEIGHTS_DIR.glob("deeponet_*/best_model.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    options = []
    for meta_path in metadata_files:
        try:
            with open(meta_path) as stream:
                meta = json.load(stream)
            label = (
                f"{meta_path.parent.name} | val_mse={meta.get('best_val_mse', float('nan')):.3e} | "
                f"n={meta.get('n_train', '-')}, p={meta.get('n_points', '-')}, m={meta.get('n_modes', '-')}"
            )
            options.append({"label": label, "value": str(meta_path)})
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    return options


def _list_fno_models() -> list[dict]:
    """List available saved FNO checkpoints (newest first)."""
    metadata_files = sorted(FNO_WEIGHTS_DIR.glob("fno_*/final_model.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    options = []
    for meta_path in metadata_files:
        try:
            with open(meta_path) as stream:
                meta = json.load(stream)
            label = (
                f"{meta_path.parent.name} | "
                f"n={meta.get('n_train', '-')}, p={meta.get('n_points', '-')}, m={meta.get('n_modes', '-')}"
            )
            options.append({"label": label, "value": str(meta_path)})
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    return options


def _forcing_coefficients(profile: str, n_modes: int) -> np.ndarray:
    """Return a deterministic forcing profile in sine-coefficient form."""
    coeffs = np.zeros((1, n_modes), dtype=np.float32)
    if profile == "mode1":
        coeffs[0, 0] = 1.0
    elif profile == "mode2" and n_modes >= 2:
        coeffs[0, 1] = 1.0
    elif profile == "mixed_low":
        for idx, value in enumerate([1.0, -0.35, 0.2]):
            if idx < n_modes:
                coeffs[0, idx] = value
    elif profile == "mixed_high":
        base = [0.0, 0.0, 0.4, -0.6, 0.5, -0.25, 0.15]
        for idx, value in enumerate(base):
            if idx < n_modes:
                coeffs[0, idx] = value
    elif profile == "random_101":
        rng = np.random.default_rng(101)
        coeffs = rng.normal(0.0, 1.0 / (np.arange(1, n_modes + 1) ** 2), size=(1, n_modes)).astype(np.float32)
    elif profile == "random_202":
        rng = np.random.default_rng(202)
        coeffs = rng.normal(0.0, 1.0 / (np.arange(1, n_modes + 1) ** 2), size=(1, n_modes)).astype(np.float32)
    else:
        coeffs[0, 0] = 1.0
    return coeffs


def _build_evaluation_case(profile: str, n_points: int, n_modes: int) -> dict:
    """Build forcing/target arrays for evaluation tab."""
    x = build_grid(n_points)
    coeffs = _forcing_coefficients(profile, n_modes)
    forcing = evaluate_sine_series(coeffs, x)[0]
    target = solve_poisson_from_coeffs(coeffs, x)[0]
    return {"x": x.tolist(), "forcing": forcing.tolist(), "target": target.tolist()}


@torch.no_grad()
def _predict_deeponet_single(model: DeepONet, forcing: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Predict one full field for one forcing profile."""
    model.eval()
    n_points = x.shape[0]
    # To predict the entire solution field in one pass, we repeat the forcing function
    # for each spatial coordinate in x. The branch network sees the same context 
    # while the trunk network evaluates each unique position.
    forcing_batch = forcing.reshape(1, -1).repeat(n_points, 1)
    x_query = x.reshape(-1, 1)
    return model(forcing_batch, x_query).squeeze(-1)


@torch.no_grad()
def _predict_fno_single(model: FNO1d, forcing: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Predict one full field for one forcing profile with FNO."""
    model.eval()
    features = torch.stack([forcing, x], dim=-1).unsqueeze(0)
    return model(features).squeeze(0)


def _empty_fig(title: str, yaxis_title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        title=title,
        paper_bgcolor="#141929",
        plot_bgcolor="#0d1120",
        font=dict(color="#a0b4d8"),
        margin=dict(l=45, r=20, t=45, b=40),
        xaxis_title="x",
        yaxis_title=yaxis_title,
    )
    return fig


def _loss_fig(history: list[dict]) -> go.Figure:
    fig = _empty_fig("Loss vs Epoch", "MSE")
    if history:
        fig.add_trace(
            go.Scatter(
                x=[entry["epoch"] for entry in history],
                y=[entry["train_loss"] for entry in history],
                mode="lines+markers",
                line=dict(color="#7eb3ff", width=3),
                marker=dict(size=6),
                name="Train Loss",
            )
        )
        fig.update_yaxes(type="log")
        fig.update_xaxes(title="Epoch")
    return fig


def _forcing_fig(data: dict | None) -> go.Figure:
    fig = _empty_fig("Input Forcing", "f(x)")
    if data:
        fig.add_trace(
            go.Scatter(
                x=data["x"],
                y=data["forcing"],
                mode="lines",
                line=dict(color="#f59e0b", width=3),
                name="f(x)",
            )
        )
    return fig


def _forcing_overlay_fig(data: dict | None, highlight_idx: int | None = None) -> go.Figure:
    """Plot all training forcings and highlight one selected sample."""
    fig = _empty_fig("Training Forcings (All Samples)", "f(x)")
    if not data:
        return fig

    x_vals = data["x"]
    forcings = data["forcings"]
    n_samples = len(forcings)
    if n_samples == 0:
        return fig

    max_traces = min(n_samples, 200)
    step = max(1, n_samples // max_traces)
    for idx in range(0, n_samples, step):
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=forcings[idx],
                mode="lines",
                line=dict(color="rgba(126,179,255,0.30)", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if highlight_idx is not None and 0 <= highlight_idx < n_samples:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=forcings[highlight_idx],
                mode="lines",
                line=dict(color="#f59e0b", width=3),
                name=f"Sample {highlight_idx}",
            )
        )
    return fig


def _training_points_fig(x_values: list[float] | None) -> go.Figure:
    """Plot 1D sensor locations used for training."""
    fig = _empty_fig("Training Point Locations", "index")
    if x_values:
        y_vals = list(range(len(x_values)))
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_vals,
                mode="markers",
                marker=dict(size=8, color="#22c55e", opacity=0.85),
                name="training points",
            )
        )
        fig.update_yaxes(showticklabels=False)
    return fig


def _solution_fig(data: dict | None) -> go.Figure:
    fig = _empty_fig("Solution Comparison", "u(x)")
    if data:
        fig.add_trace(
            go.Scatter(
                x=data["x"],
                y=data["target"],
                mode="lines",
                line=dict(color="#22c55e", width=3),
                name="Target",
            )
        )
        if "prediction" in data and data["prediction"] is not None:
            fig.add_trace(
                go.Scatter(
                    x=data["x"],
                    y=data["prediction"],
                    mode="lines",
                    line=dict(color="#7eb3ff", width=3, dash="dash"),
                    name="Prediction",
                )
            )
    return fig


def _error_fig(data: dict | None) -> go.Figure:
    fig = _empty_fig("Pointwise Error", "u_pred - u_true")
    if data and "prediction" in data and data["prediction"] is not None:
        target = torch.tensor(data["target"])
        prediction = torch.tensor(data["prediction"])
        fig.add_trace(
            go.Scatter(
                x=data["x"],
                y=(prediction - target).tolist(),
                mode="lines",
                line=dict(color="#ef4444", width=3),
                name="Error",
            )
        )
    return fig


def _plot_card(title: str, graph_id: dict) -> html.Div:
    return html.Div(
        [
            html.Div(title, className="plot-title"),
            dcc.Graph(id=graph_id, config=_graph_config, style={"height": "300px"}),
        ],
        className="plot-card",
    )


def _sidebar(method: dict) -> html.Div:
    defaults = method["defaults"]
    method_id = method["id"]
    return html.Div(
        [
            html.Div("Configuration", className="section-header"),
            dcc.Markdown(
                f"**{method['name']}**\n{method['description']}",
                mathjax=True,
                style={"fontSize": "13px", "color": "#a0b4d8", "marginBottom": "15px"},
            ),
            html.Div("Operator", className="section-header"),
            dcc.Markdown(method["equation"], mathjax=True, style={"fontSize": "14px", "minHeight": "56px"}),
            dcc.Markdown(method["architecture"], mathjax=True, style={"fontSize": "13px", "minHeight": "48px"}),
            html.Hr(style={"borderColor": "#2a3555", "margin": "12px 0"}),
            html.Div("Training Set Size", className="control-label"),
            dcc.Slider(
                id={"type": "n-train", "method": method_id},
                min=64,
                max=1024,
                step=64,
                value=defaults["n_train"],
                marks={i: {"label": str(i), "style": {"color": "#5577aa", "fontSize": "10px"}} for i in [64, 256, 512, 1024]},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Div("Epochs", className="control-label mt-3"),
            dcc.Slider(
                id={"type": "epochs", "method": method_id},
                min=100,
                max=600,
                step=50,
                value=defaults["epochs"],
                marks={i: {"label": str(i), "style": {"color": "#5577aa", "fontSize": "10px"}} for i in [100, 250, 400, 600]},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Div("Learning Rate", className="control-label mt-3"),
            dcc.Input(
                id={"type": "lr", "method": method_id},
                type="number",
                value=defaults["lr"],
                step=0.0001,
                min=0.0001,
                className="w-100",
                style={"marginBottom": "10px"},
            ),
            html.Div("Grid Points", className="control-label"),
            dcc.Slider(
                id={"type": "n-points", "method": method_id},
                min=32,
                max=128,
                step=16,
                value=defaults["n_points"],
                marks={i: {"label": str(i), "style": {"color": "#5577aa", "fontSize": "10px"}} for i in [32, 64, 96, 128]},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Div("Sine Modes", className="control-label mt-3"),
            dcc.Slider(
                id={"type": "n-modes", "method": method_id},
                min=3,
                max=10,
                step=1,
                value=defaults["n_modes"],
                marks={i: {"label": str(i), "style": {"color": "#5577aa", "fontSize": "10px"}} for i in [3, 5, 7, 10]},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Div("Inspect Training Sample", className="control-label mt-3"),
            dcc.Dropdown(
                id={"type": "sample-select", "method": method_id},
                options=[],
                value=0,
                clearable=False,
                style={"marginBottom": "8px"},
            ),
            html.Hr(style={"borderColor": "#2a3555", "margin": "12px 0"}),
            html.Div("Actions", className="section-header"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Train On GPU",
                            id={"type": "train-btn", "method": method_id},
                            n_clicks=0,
                            className="w-100 btn-train",
                            color="primary",
                        ),
                        width=7,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Clear",
                            id={"type": "clear-btn", "method": method_id},
                            n_clicks=0,
                            className="w-100 btn-clear",
                            color="secondary",
                            outline=True,
                        ),
                        width=5,
                    ),
                ],
                className="mb-2",
            ),
            dbc.Progress(
                id={"type": "train-progress", "method": method_id},
                value=0,
                max=100,
                striped=True,
                animated=True,
                className="mb-2",
                style={"height": "6px"},
            ),
            html.Div(
                f"CUDA device: {torch.cuda.get_device_name(device)}",
                className="gpu-note",
            ),
            html.Div(id={"type": "train-status", "method": method_id}, className="status-panel"),
            dcc.Interval(
                id={"type": "poll-interval", "method": method_id},
                interval=1500,
                n_intervals=0,
                disabled=True,
            ),
        ],
        className="operator-sidebar p-3",
    )


def _method_tab(method: dict) -> dbc.Container:
    method_id = method["id"]
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(_sidebar(method), width=3, style={"minWidth": "290px"}),
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(_plot_card("Input Forcing", {"type": "forcing-plot", "method": method_id}), width=6),
                                    dbc.Col(_plot_card("Loss Curve", {"type": "loss-plot", "method": method_id}), width=6),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(_plot_card("Solution Comparison", {"type": "solution-plot", "method": method_id}), width=6),
                                    dbc.Col(_plot_card("Pointwise Error", {"type": "error-plot", "method": method_id}), width=6),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(_plot_card("Training Point Locations", {"type": "train-points-plot", "method": method_id}), width=12),
                                ]
                            ),
                            html.Div(id={"type": "metrics-card", "method": method_id}, className="metrics-card"),
                        ],
                        width=9,
                    ),
                ],
                className="mt-2 g-3",
            )
        ],
        fluid=True,
        className="mt-3",
    )


def _deeponet_evaluation_tab() -> dbc.Container:
    """DeepONet model evaluation tab layout."""
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Div("Model Selection", className="section-header"),
                                dcc.Dropdown(
                                    id="eval-model-select",
                                    options=[],
                                    value=None,
                                    placeholder="Select a saved DeepONet model",
                                    style={"marginBottom": "8px"},
                                ),
                                dbc.Button(
                                    "Refresh Models",
                                    id="eval-refresh-models-btn",
                                    n_clicks=0,
                                    color="secondary",
                                    outline=True,
                                    className="w-100",
                                ),
                                dbc.Button(
                                    "Delete All Models",
                                    id="eval-delete-models-btn",
                                    n_clicks=0,
                                    color="danger",
                                    outline=True,
                                    className="w-100 mt-2",
                                ),
                                html.Div(id="eval-model-status", className="status-panel mt-2"),
                                html.Hr(style={"borderColor": "#2a3555", "margin": "12px 0"}),
                                html.Div("Input Forcing", className="section-header"),
                                dcc.Dropdown(
                                    id="eval-forcing-select",
                                    options=[{"label": label, "value": value} for value, label in FORCING_PROFILES],
                                    value="mode1",
                                    clearable=False,
                                    style={"marginBottom": "10px"},
                                ),
                                dbc.Button(
                                    "Run Evaluation",
                                    id="eval-run-btn",
                                    n_clicks=0,
                                    color="primary",
                                    className="w-100 btn-train",
                                ),
                                html.Div(id="eval-run-status", className="status-panel mt-2"),
                                html.Hr(style={"borderColor": "#2a3555", "margin": "12px 0"}),
                                html.Div(id="eval-metrics-card", className="plot-card p-3"),
                            ],
                            className="operator-sidebar p-3",
                        ),
                        width=3,
                        style={"minWidth": "290px"},
                    ),
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(_plot_card("Selected Forcing", "eval-forcing-plot"), width=6),
                                    dbc.Col(_plot_card("Prediction vs Exact", "eval-solution-plot"), width=6),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(_plot_card("Pointwise Error", "eval-error-plot"), width=12),
                                ]
                            ),
                        ],
                        width=9,
                    ),
                ],
                className="mt-2 g-3",
            )
        ],
        fluid=True,
        className="mt-3",
    )


def _fno_evaluation_tab() -> dbc.Container:
    """FNO model evaluation tab layout."""
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Div("Model Selection", className="section-header"),
                                dcc.Dropdown(
                                    id="fno-eval-model-select",
                                    options=[],
                                    value=None,
                                    placeholder="Select a saved FNO model",
                                    style={"marginBottom": "8px"},
                                ),
                                dbc.Button(
                                    "Refresh Models",
                                    id="fno-eval-refresh-models-btn",
                                    n_clicks=0,
                                    color="secondary",
                                    outline=True,
                                    className="w-100",
                                ),
                                dbc.Button(
                                    "Delete All Models",
                                    id="fno-eval-delete-models-btn",
                                    n_clicks=0,
                                    color="danger",
                                    outline=True,
                                    className="w-100 mt-2",
                                ),
                                html.Div(id="fno-eval-model-status", className="status-panel mt-2"),
                                html.Hr(style={"borderColor": "#2a3555", "margin": "12px 0"}),
                                html.Div("Input Forcing", className="section-header"),
                                dcc.Dropdown(
                                    id="fno-eval-forcing-select",
                                    options=[{"label": label, "value": value} for value, label in FORCING_PROFILES],
                                    value="mode1",
                                    clearable=False,
                                    style={"marginBottom": "10px"},
                                ),
                                dbc.Button(
                                    "Run Evaluation",
                                    id="fno-eval-run-btn",
                                    n_clicks=0,
                                    color="primary",
                                    className="w-100 btn-train",
                                ),
                                html.Div(id="fno-eval-run-status", className="status-panel mt-2"),
                                html.Hr(style={"borderColor": "#2a3555", "margin": "12px 0"}),
                                html.Div(id="fno-eval-metrics-card", className="plot-card p-3"),
                            ],
                            className="operator-sidebar p-3",
                        ),
                        width=3,
                        style={"minWidth": "290px"},
                    ),
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(_plot_card("Selected Forcing", "fno-eval-forcing-plot"), width=6),
                                    dbc.Col(_plot_card("Prediction vs Exact", "fno-eval-solution-plot"), width=6),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(_plot_card("Pointwise Error", "fno-eval-error-plot"), width=12),
                                ]
                            ),
                        ],
                        width=9,
                    ),
                ],
                className="mt-2 g-3",
            )
        ],
        fluid=True,
        className="mt-3",
    )


THEORY_MARKDOWN = r"""
### Neural Operators

Neural operators learn mappings between whole functions, not just between finite-dimensional vectors. In this demo the operator is

$$
f(x)\mapsto u(x), \qquad -u''(x)=f(x), \quad u(0)=u(1)=0.
$$

The forcing is sampled as a random sine expansion, and the exact solution is available analytically in the same basis. That makes the operator-learning task fast enough for an interactive seminar app while still being mathematically clean.

### DeepONet

DeepONet separates the operator into two learned pieces:

- a **branch net** that reads the input function values at sensor points,
- a **trunk net** that reads the query location `x`.

Their latent features are combined to predict `u(x)` at arbitrary points.

### Fourier Neural Operator

FNO works on the full discretized function and alternates:

- local channel mixing in physical space,
- global mixing through truncated Fourier modes.

This makes it a natural architecture when the input and output are both fields on a grid.
"""


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    assets_folder=str(BASE_DIR / "assets"),
)

app.layout = html.Div(
    [
        html.Div(dcc.Markdown("$$ $$", mathjax=True), style={"display": "none"}),
        html.Div(
            [
                html.Span("Neural Operator Learning", className="operator-title"),
                html.Span("DeepONet and Fourier Neural Operators", className="operator-subtitle"),
            ],
            className="operator-navbar d-flex align-items-center",
        ),
        dbc.Container(
            dbc.Tabs(
                id="operator-tabs",
                active_tab="deeponet",
                children=[
                    dbc.Tab(label=method["name"], tab_id=method["id"], children=[_method_tab(method)])
                    for method in METHODS
                ]
                + [
                    dbc.Tab(
                        label="DeepONet Evaluation",
                        tab_id="deeponet-eval",
                        children=[_deeponet_evaluation_tab()],
                    ),
                    dbc.Tab(
                        label="FNO Evaluation",
                        tab_id="fno-eval",
                        children=[_fno_evaluation_tab()],
                    ),
                    dbc.Tab(
                        label="Theory",
                        tab_id="theory",
                        children=[
                            dbc.Container(
                                dbc.Row(
                                    dbc.Col(
                                        html.Div(
                                            dcc.Markdown(
                                                THEORY_MARKDOWN,
                                                mathjax=True,
                                                className="p-4",
                                                style={"color": "#a0b4d8", "fontSize": "15px", "lineHeight": "1.6"},
                                            ),
                                            className="theory-card",
                                        )
                                    )
                                ),
                                fluid=True,
                            )
                        ],
                    )
                ],
            ),
            fluid=True,
            className="px-3 pt-3",
        ),
    ],
    style={"minHeight": "100vh", "backgroundColor": "#0f1117"},
)


def _train_worker(method_id: str, n_train: int, epochs: int, lr: float, n_points: int, n_modes: int) -> None:
    """Background training worker."""
    print(
        "[TRAIN_WORKER_START] "
        f"method={method_id} n_train={n_train} epochs={epochs} lr={lr} n_points={n_points} n_modes={n_modes}",
        flush=True,
    )
    data = {
        "status": "training",
        "history": [],
        "current_epoch": 0,
        "total_epochs": epochs,
        "metrics": None,
        "example": None,
        "best_model_path": None,
        "best_metadata_path": None,
        "config": {
            "n_train": n_train,
            "epochs": epochs,
            "lr": lr,
            "n_points": n_points,
            "n_modes": n_modes,
        },
    }
    _write_history(method_id, data)

    train_data = make_dataset(n_samples=n_train, n_points=n_points, n_modes=n_modes, seed=7)
    test_data = make_dataset(n_samples=32, n_points=n_points, n_modes=n_modes, seed=17)

    try:
        if method_id == "deeponet":
            model = DeepONet(n_sensors=n_points).to(device)
            val_data = make_dataset(n_samples=32, n_points=n_points, n_modes=n_modes, seed=23)
            run_dir = _create_deeponet_run_dir(data["config"])
            best_val_mse = float("inf")

            def val_loss_fn() -> float:
                val_prediction = predict_deeponet(model, val_data, device)
                val_target = val_data["solution"].to(device)
                return float(summarize_prediction(val_prediction, val_target)["mse"])

            def on_epoch(epoch: int, train_loss: float, val_loss: float | None, lr_now: float) -> None:
                nonlocal best_val_mse
                val_mse = float("inf") if val_loss is None else float(val_loss)
                print(
                    f"[DeepONet] epoch={epoch:4d} train_mse={train_loss:.3e} val_mse={val_mse:.3e} lr={lr_now:.3e}",
                    flush=True,
                )

                payload = _read_history(method_id)
                payload["history"].append(
                    {"epoch": epoch, "train_loss": train_loss, "val_mse": val_mse, "lr": lr_now}
                )
                payload["current_epoch"] = epoch

                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    metadata = {
                        "model_type": "deeponet",
                        "saved_at": datetime.now().isoformat(),
                        "best_epoch": epoch,
                        "best_val_mse": val_mse,
                        "n_train": n_train,
                        "epochs": epochs,
                        "lr": lr,
                        "n_points": n_points,
                        "n_modes": n_modes,
                        "device": str(device),
                        "run_dir": str(run_dir),
                    }
                    weights_path, meta_path = _save_deeponet_best(model, run_dir, metadata)
                    payload["best_model_path"] = weights_path
                    payload["best_metadata_path"] = meta_path

                _write_history(method_id, payload)

            train_deeponet(
                model,
                train_data,
                device=device,
                epochs=epochs,
                learning_rate=lr,
                batch_size=1024,
                log_interval=10,
                callback=on_epoch,
                val_loss_fn=val_loss_fn,
                early_stopping_patience=30,
                lr_scheduler_patience=10,
                lr_scheduler_factor=0.5,
                min_lr=1e-6,
            )
            prediction = predict_deeponet(model, test_data, device)
        else:
            model = FNO1d(input_dim=2, width=32, modes=min(12, max(4, n_points // 4)), depth=4).to(device)
            val_data = make_dataset(n_samples=32, n_points=n_points, n_modes=n_modes, seed=23)
            run_dir = _create_fno_run_dir(data["config"])

            def val_loss_fn() -> float:
                val_prediction = predict_fno(model, val_data, device)
                val_target = val_data["solution"].to(device)
                return float(summarize_prediction(val_prediction, val_target)["mse"])

            def on_epoch(epoch: int, train_loss: float, val_loss: float | None, lr_now: float) -> None:
                val_mse = float("nan") if val_loss is None else float(val_loss)
                print(
                    f"[FNO]      epoch={epoch:4d} train_mse={train_loss:.3e} val_mse={val_mse:.3e} lr={lr_now:.3e}",
                    flush=True,
                )
                payload = _read_history(method_id)
                row = {"epoch": epoch, "train_loss": train_loss, "lr": lr_now}
                if val_loss is not None:
                    row["val_mse"] = float(val_loss)
                payload["history"].append(row)
                payload["current_epoch"] = epoch
                _write_history(method_id, payload)

            train_fno(
                model,
                train_data,
                device=device,
                epochs=epochs,
                learning_rate=lr,
                batch_size=32,
                log_interval=10,
                callback=on_epoch,
                val_loss_fn=val_loss_fn,
                early_stopping_patience=30,
                lr_scheduler_patience=10,
                lr_scheduler_factor=0.5,
                min_lr=1e-6,
            )
            fno_meta = {
                "model_type": "fno",
                "saved_at": datetime.now().isoformat(),
                "n_train": n_train,
                "epochs": epochs,
                "lr": lr,
                "n_points": n_points,
                "n_modes": n_modes,
                "device": str(device),
                "run_dir": str(run_dir),
            }
            fno_weights, fno_meta_path = _save_fno_model(model, run_dir, fno_meta)
            payload = _read_history(method_id)
            payload["best_model_path"] = fno_weights
            payload["best_metadata_path"] = fno_meta_path
            _write_history(method_id, payload)
            prediction = predict_fno(model, test_data, device)

        target = test_data["solution"].to(device)
        metrics = summarize_prediction(prediction, target)
        example = {
            "x": test_data["x"][0, :, 0].tolist(),
            "forcing": test_data["forcing"][0].tolist(),
            "target": target[0].detach().cpu().tolist(),
            "prediction": prediction[0].detach().cpu().tolist(),
        }

        payload = _read_history(method_id)
        final_epoch = int(payload.get("current_epoch", epochs))
        payload["status"] = "complete"
        payload["metrics"] = metrics
        payload["example"] = example
        payload["current_epoch"] = final_epoch
        payload["stopped_early"] = final_epoch < epochs
        _write_history(method_id, payload)
    except Exception as exc:
        payload = _read_history(method_id)
        payload["status"] = "error"
        payload["error"] = str(exc)
        _write_history(method_id, payload)


@callback(
    Output({"type": "poll-interval", "method": MATCH}, "disabled"),
    Output({"type": "train-status", "method": MATCH}, "children"),
    Input({"type": "train-btn", "method": MATCH}, "n_clicks"),
    Input({"type": "clear-btn", "method": MATCH}, "n_clicks"),
    State({"type": "n-train", "method": MATCH}, "value"),
    State({"type": "epochs", "method": MATCH}, "value"),
    State({"type": "lr", "method": MATCH}, "value"),
    State({"type": "n-points", "method": MATCH}, "value"),
    State({"type": "n-modes", "method": MATCH}, "value"),
    prevent_initial_call=True,
)
def start_or_clear(
    train_clicks: int,
    clear_clicks: int,
    n_train: int,
    epochs: int,
    lr: float,
    n_points: int,
    n_modes: int,
):
    """Start background training or clear prior state."""
    ctx = dash.callback_context
    triggered = ctx.triggered_id
    if triggered is None:
        raise PreventUpdate

    method_id = triggered["method"]
    if triggered["type"] == "clear-btn":
        path = _history_path(method_id)
        if path.exists():
            path.unlink()
        return True, "Cleared previous results."

    print(
        "[TRAIN_REQUEST] "
        f"method={method_id} n_train={n_train} epochs={epochs} lr={lr} n_points={n_points} n_modes={n_modes}",
        flush=True,
    )

    thread = threading.Thread(
        target=_train_worker,
        args=(method_id, int(n_train), int(epochs), float(lr), int(n_points), int(n_modes)),
        daemon=True,
    )
    thread.start()
    return False, f"Training {method_id} on {torch.cuda.get_device_name(device)}."


@callback(
    Output({"type": "train-progress", "method": MATCH}, "value"),
    Output({"type": "train-status", "method": MATCH}, "children", allow_duplicate=True),
    Output({"type": "loss-plot", "method": MATCH}, "figure"),
    Output({"type": "forcing-plot", "method": MATCH}, "figure"),
    Output({"type": "solution-plot", "method": MATCH}, "figure"),
    Output({"type": "error-plot", "method": MATCH}, "figure"),
    Output({"type": "train-points-plot", "method": MATCH}, "figure"),
    Output({"type": "metrics-card", "method": MATCH}, "children"),
    Output({"type": "sample-select", "method": MATCH}, "options"),
    Output({"type": "sample-select", "method": MATCH}, "value"),
    Output({"type": "poll-interval", "method": MATCH}, "disabled", allow_duplicate=True),
    Input({"type": "poll-interval", "method": MATCH}, "n_intervals"),
    Input({"type": "sample-select", "method": MATCH}, "value"),
    Input({"type": "n-train", "method": MATCH}, "value"),
    Input({"type": "n-points", "method": MATCH}, "value"),
    Input({"type": "n-modes", "method": MATCH}, "value"),
    State({"type": "poll-interval", "method": MATCH}, "id"),
    prevent_initial_call=True,
)
def refresh_training(
    _: int,
    selected_sample: int | None,
    n_train_slider: int,
    n_points_slider: int,
    n_modes_slider: int,
    interval_id: dict,
):
    """Refresh plots and status while training."""
    method_id = interval_id["method"]
    payload = _read_history(method_id)
    history = payload.get("history", [])
    metrics = payload.get("metrics")
    cfg = payload.get("config", {})

    n_train = int(n_train_slider or cfg.get("n_train", 0))
    n_points = int(n_points_slider or cfg.get("n_points", 64))
    n_modes = int(n_modes_slider or cfg.get("n_modes", 6))
    sample_options = [{"label": f"Sample {i}", "value": i} for i in range(n_train)]
    if n_train <= 0:
        selected_sample = 0
    elif selected_sample is None or selected_sample < 0 or selected_sample >= n_train:
        selected_sample = 0

    total_epochs = max(int(payload.get("total_epochs", 1)), 1)
    current_epoch = int(payload.get("current_epoch", 0))
    progress = int(100 * current_epoch / total_epochs)
    status = payload.get("status", "idle")

    if status == "training":
        status_text = f"Training on GPU. Epoch {current_epoch} / {total_epochs}"
        disabled = False
    elif status == "complete":
        early_msg = " (early stopped)" if payload.get("stopped_early") else ""
        status_text = f"Training complete{early_msg}. Relative L2 error = {metrics['relative_l2']:.3e}"
        disabled = True
    elif status == "error":
        status_text = f"Training error: {payload.get('error', 'unknown error')}"
        disabled = True
    else:
        status_text = "Idle."
        disabled = True

    forcing_fig = _forcing_fig(payload.get("example"))
    solution_fig = _solution_fig(payload.get("example"))
    error_fig = _error_fig(payload.get("example"))
    train_points_fig = _training_points_fig(None)

    if n_train > 0:
        train_data = make_dataset(n_samples=n_train, n_points=n_points, n_modes=n_modes, seed=7)
        x_vals = train_data["x"][0, :, 0].detach().cpu().tolist()
        forcing_matrix = train_data["forcing"].detach().cpu().tolist()
        forcing_fig = _forcing_overlay_fig({"x": x_vals, "forcings": forcing_matrix}, highlight_idx=selected_sample)
        train_points_fig = _training_points_fig(x_vals)
        sample_data = {
            "x": train_data["x"][selected_sample, :, 0].detach().cpu().tolist(),
            "target": train_data["solution"][selected_sample].detach().cpu().tolist(),
            "prediction": None,
        }
        solution_fig = _solution_fig(sample_data)
        error_fig = _error_fig(sample_data)

        if status == "complete":
            best_meta_path = payload.get("best_metadata_path")
            if best_meta_path and Path(best_meta_path).exists():
                try:
                    with open(best_meta_path) as stream:
                        best_meta = json.load(stream)
                    forcing_sel = train_data["forcing"][selected_sample].to(device)
                    x_sel = train_data["x"][selected_sample, :, 0].to(device)
                    target_sel = train_data["solution"][selected_sample].to(device)
                    if method_id == "deeponet":
                        model = DeepONet(n_sensors=n_points).to(device)
                        model.load_state_dict(torch.load(best_meta["weights_path"], map_location=device))
                        pred_sel = _predict_deeponet_single(model, forcing_sel, x_sel).detach().cpu()
                    else:
                        model = FNO1d(input_dim=2, width=32, modes=min(12, max(4, n_points // 4)), depth=4).to(device)
                        model.load_state_dict(torch.load(best_meta["weights_path"], map_location=device))
                        pred_sel = _predict_fno_single(model, forcing_sel, x_sel).detach().cpu()
                    sample_data = {
                        "x": x_sel.detach().cpu().tolist(),
                        "target": target_sel.detach().cpu().tolist(),
                        "prediction": pred_sel.tolist(),
                    }
                    solution_fig = _solution_fig(sample_data)
                    error_fig = _error_fig(sample_data)
                    sample_metrics = summarize_prediction(
                        pred_sel.unsqueeze(0),
                        target_sel.detach().cpu().unsqueeze(0),
                    )
                except Exception:
                    sample_metrics = None
            else:
                sample_metrics = None
        else:
            sample_metrics = None
    else:
        sample_metrics = None

    metrics_card = html.Div(
        [
            html.Div("Run Summary", className="section-header"),
            html.Div(f"Status: {status_text}", className="metric-line"),
            html.Div(f"Train samples: {cfg.get('n_train', '-')}", className="metric-line"),
            html.Div(f"Grid points: {cfg.get('n_points', '-')}", className="metric-line"),
            html.Div(f"Modes: {cfg.get('n_modes', '-')}", className="metric-line"),
            html.Div(f"Preview n_train: {n_train}", className="metric-line"),
            html.Div(f"Preview n_points: {n_points}", className="metric-line"),
            html.Div(f"Preview n_modes: {n_modes}", className="metric-line"),
            html.Div(f"Selected sample: {selected_sample}", className="metric-line"),
            html.Div(
                f"Test MSE: {metrics['mse']:.3e}" if metrics else "Test MSE: -",
                className="metric-line",
            ),
            html.Div(
                f"Relative L2: {metrics['relative_l2']:.3e}" if metrics else "Relative L2: -",
                className="metric-line",
            ),
            html.Div(
                f"Selected MSE: {sample_metrics['mse']:.3e}" if sample_metrics else "Selected MSE: -",
                className="metric-line",
            ),
            html.Div(
                f"Selected Relative L2: {sample_metrics['relative_l2']:.3e}" if sample_metrics else "Selected Relative L2: -",
                className="metric-line",
            ),
        ],
        className="plot-card p-3",
    )

    return (
        progress,
        status_text,
        _loss_fig(history),
        forcing_fig,
        solution_fig,
        error_fig,
        train_points_fig,
        metrics_card,
        sample_options,
        selected_sample,
        disabled,
    )


@callback(
    Output({"type": "loss-plot", "method": MATCH}, "figure", allow_duplicate=True),
    Output({"type": "forcing-plot", "method": MATCH}, "figure", allow_duplicate=True),
    Output({"type": "solution-plot", "method": MATCH}, "figure", allow_duplicate=True),
    Output({"type": "error-plot", "method": MATCH}, "figure", allow_duplicate=True),
    Output({"type": "train-points-plot", "method": MATCH}, "figure", allow_duplicate=True),
    Output({"type": "metrics-card", "method": MATCH}, "children", allow_duplicate=True),
    Output({"type": "train-progress", "method": MATCH}, "value", allow_duplicate=True),
    Output({"type": "sample-select", "method": MATCH}, "options", allow_duplicate=True),
    Output({"type": "sample-select", "method": MATCH}, "value", allow_duplicate=True),
    Input({"type": "clear-btn", "method": MATCH}, "n_clicks"),
    prevent_initial_call=True,
)
def clear_figures(_: int):
    """Reset graphs after clearing."""
    return (
        _loss_fig([]),
        _forcing_fig(None),
        _solution_fig(None),
        _error_fig(None),
        _training_points_fig(None),
        html.Div("No run data yet.", className="plot-card p-3"),
        0,
        [],
        0,
    )


@callback(
    Output("eval-model-select", "options"),
    Output("eval-model-select", "value"),
    Output("eval-model-status", "children"),
    Input("eval-refresh-models-btn", "n_clicks"),
    Input("eval-delete-models-btn", "n_clicks"),
    Input({"type": "train-status", "method": "deeponet"}, "children"),
    State("eval-model-select", "value"),
)
def refresh_eval_models(_: int, delete_clicks: int, __: str, current_value: str | None):
    """Refresh DeepONet model list for evaluation tab."""
    ctx = dash.callback_context
    if ctx.triggered_id == "eval-delete-models-btn":
        for run_dir in DEEPNET_WEIGHTS_DIR.glob("deeponet_*"):
            if run_dir.is_dir():
                shutil.rmtree(run_dir, ignore_errors=True)

    options = _list_deeponet_models()
    if not options:
        if ctx.triggered_id == "eval-delete-models-btn":
            return [], None, "Deleted all saved DeepONet models."
        return [], None, "No saved DeepONet models found. Train a DeepONet run first."

    option_values = {opt["value"] for opt in options}
    selected = current_value if current_value in option_values else options[0]["value"]
    return options, selected, f"Loaded {len(options)} saved model(s)."


@callback(
    Output("eval-forcing-plot", "figure"),
    Output("eval-solution-plot", "figure"),
    Output("eval-error-plot", "figure"),
    Output("eval-metrics-card", "children"),
    Output("eval-run-status", "children"),
    Input("eval-run-btn", "n_clicks"),
    State("eval-model-select", "value"),
    State("eval-forcing-select", "value"),
    prevent_initial_call=True,
)
def run_deeponet_evaluation(_: int, selected_metadata_path: str | None, forcing_profile: str):
    """Run selected DeepONet checkpoint on a chosen forcing profile."""
    if not selected_metadata_path:
        raise PreventUpdate

    try:
        with open(selected_metadata_path) as stream:
            metadata = json.load(stream)
        weights_path = Path(metadata["weights_path"])
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights file: {weights_path}")

        n_points = int(metadata.get("n_points", 64))
        n_modes = int(metadata.get("n_modes", 6))
        case = _build_evaluation_case(forcing_profile, n_points, n_modes)

        model = DeepONet(n_sensors=n_points).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        eval_data = {
            "forcing": torch.tensor([case["forcing"]], dtype=torch.float32),
            "x": torch.tensor(case["x"], dtype=torch.float32).reshape(1, -1, 1),
            "solution": torch.tensor([case["target"]], dtype=torch.float32),
        }
        prediction = predict_deeponet(model, eval_data, device).detach().cpu()
        target = eval_data["solution"]
        metrics = summarize_prediction(prediction, target)

        example = {
            "x": case["x"],
            "forcing": case["forcing"],
            "target": target[0].tolist(),
            "prediction": prediction[0].tolist(),
        }
        metrics_card = html.Div(
            [
                html.Div("Evaluation Summary", className="section-header"),
                html.Div(f"Model: {Path(selected_metadata_path).parent.name}", className="metric-line"),
                html.Div(f"Profile: {forcing_profile}", className="metric-line"),
                html.Div(f"MSE: {metrics['mse']:.3e}", className="metric-line"),
                html.Div(f"Relative L2: {metrics['relative_l2']:.3e}", className="metric-line"),
                html.Div(f"Grid points: {n_points}", className="metric-line"),
                html.Div(f"Sine modes: {n_modes}", className="metric-line"),
            ],
            className="plot-card p-3",
        )
        return (
            _forcing_fig(example),
            _solution_fig(example),
            _error_fig(example),
            metrics_card,
            "Evaluation complete.",
        )
    except Exception as exc:
        return (
            _forcing_fig(None),
            _solution_fig(None),
            _error_fig(None),
            html.Div("Evaluation failed.", className="plot-card p-3"),
            f"Evaluation error: {exc}",
        )


@callback(
    Output("fno-eval-model-select", "options"),
    Output("fno-eval-model-select", "value"),
    Output("fno-eval-model-status", "children"),
    Input("fno-eval-refresh-models-btn", "n_clicks"),
    Input("fno-eval-delete-models-btn", "n_clicks"),
    Input({"type": "train-status", "method": "fno"}, "children"),
    State("fno-eval-model-select", "value"),
)
def refresh_fno_eval_models(_: int, __: int, ___: str, current_value: str | None):
    """Refresh FNO model list for evaluation tab."""
    ctx = dash.callback_context
    if ctx.triggered_id == "fno-eval-delete-models-btn":
        for run_dir in FNO_WEIGHTS_DIR.glob("fno_*"):
            if run_dir.is_dir():
                shutil.rmtree(run_dir, ignore_errors=True)

    options = _list_fno_models()
    if not options:
        if ctx.triggered_id == "fno-eval-delete-models-btn":
            return [], None, "Deleted all saved FNO models."
        return [], None, "No saved FNO models found. Train an FNO run first."

    option_values = {opt["value"] for opt in options}
    selected = current_value if current_value in option_values else options[0]["value"]
    return options, selected, f"Loaded {len(options)} saved model(s)."


@callback(
    Output("fno-eval-forcing-plot", "figure"),
    Output("fno-eval-solution-plot", "figure"),
    Output("fno-eval-error-plot", "figure"),
    Output("fno-eval-metrics-card", "children"),
    Output("fno-eval-run-status", "children"),
    Input("fno-eval-run-btn", "n_clicks"),
    State("fno-eval-model-select", "value"),
    State("fno-eval-forcing-select", "value"),
    prevent_initial_call=True,
)
def run_fno_evaluation(_: int, selected_metadata_path: str | None, forcing_profile: str):
    """Run selected FNO checkpoint on a chosen forcing profile."""
    if not selected_metadata_path:
        raise PreventUpdate

    try:
        with open(selected_metadata_path) as stream:
            metadata = json.load(stream)
        weights_path = Path(metadata["weights_path"])
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights file: {weights_path}")

        n_points = int(metadata.get("n_points", 64))
        n_modes = int(metadata.get("n_modes", 6))
        case = _build_evaluation_case(forcing_profile, n_points, n_modes)

        model = FNO1d(input_dim=2, width=32, modes=min(12, max(4, n_points // 4)), depth=4).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        forcing = torch.tensor(case["forcing"], dtype=torch.float32).to(device)
        x_grid = torch.tensor(case["x"], dtype=torch.float32).to(device)
        target = torch.tensor(case["target"], dtype=torch.float32)
        prediction = _predict_fno_single(model, forcing, x_grid).detach().cpu()
        metrics = summarize_prediction(prediction.unsqueeze(0), target.unsqueeze(0))

        example = {
            "x": case["x"],
            "forcing": case["forcing"],
            "target": target.tolist(),
            "prediction": prediction.tolist(),
        }
        metrics_card = html.Div(
            [
                html.Div("Evaluation Summary", className="section-header"),
                html.Div(f"Model: {Path(selected_metadata_path).parent.name}", className="metric-line"),
                html.Div(f"Profile: {forcing_profile}", className="metric-line"),
                html.Div(f"MSE: {metrics['mse']:.3e}", className="metric-line"),
                html.Div(f"Relative L2: {metrics['relative_l2']:.3e}", className="metric-line"),
                html.Div(f"Grid points: {n_points}", className="metric-line"),
                html.Div(f"Sine modes: {n_modes}", className="metric-line"),
            ],
            className="plot-card p-3",
        )
        return (
            _forcing_fig(example),
            _solution_fig(example),
            _error_fig(example),
            metrics_card,
            "Evaluation complete.",
        )
    except Exception as exc:
        return (
            _forcing_fig(None),
            _solution_fig(None),
            _error_fig(None),
            html.Div("Evaluation failed.", className="plot-card p-3"),
            f"Evaluation error: {exc}",
        )


if __name__ == "__main__":
    app.run(debug=True, port=8053, dev_tools_hot_reload=True)
