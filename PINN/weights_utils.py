"""Model weights persistence: save and load PINN state with metadata."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from pinn_core import PINN

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def _ensure_weights_dir() -> Path:
    """Create weights directory if it does not exist."""
    WEIGHTS_DIR.mkdir(exist_ok=True)
    return WEIGHTS_DIR


def save_model(
    model: PINN,
    equation_type: str,
    n_x: int,
    n_t: int,
    epochs: int,
    num_hidden: int | None = None,
    dim_hidden: int | None = None,
    strategy: str = "standard",
    learning_type: str = "standard",
    metadata: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Save model state_dict and metadata.

    Args:
        model: The trained PINN.
        equation_type: 'heat' or 'burgers'.
        n_x: Number of spatial collocation points.
        n_t: Number of temporal collocation points.
        epochs: Number of training epochs.
        metadata: Optional extra metadata to store.

    Returns:
        Tuple of (weights_path, metadata_path).
    """
    # Extract first 3 significant digits of the loss for the filename
    loss_val = metadata.get("best_val_loss", 0.0) if metadata else 0.0
    # Use scientific notation to get significant digits regardless of magnitude
    loss_str = f"{loss_val:.2e}".split("e")[0].replace(".", "")[:3]
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{equation_type}_{strategy}_{learning_type}_{ts}_loss_{loss_str}"

    # Create a dedicated subfolder for this training run
    run_dir = WEIGHTS_DIR / base
    run_dir.mkdir(exist_ok=True)

    pt_path = run_dir / f"{base}.pt"
    json_path = run_dir / f"{base}.json"

    torch.save(model.state_dict(), pt_path)

    meta: dict[str, Any] = {
        "equation_type": equation_type,
        "n_x": n_x,
        "n_t": n_t,
        "epochs": epochs,
    }
    if num_hidden is not None:
        meta["num_hidden"] = num_hidden
    if dim_hidden is not None:
        meta["dim_hidden"] = dim_hidden
    meta["strategy"] = strategy
    meta["learning_type"] = learning_type
    if metadata:
        meta.update(metadata)

    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    return str(pt_path), str(json_path)


def create_run_dir(
    equation_type: str,
    strategy: str = "standard",
    learning_type: str = "standard",
) -> Path:
    """Create a fixed run directory at the start of training (no loss in name yet)."""
    _ensure_weights_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{equation_type}_{strategy}_{learning_type}_{ts}"
    run_dir = WEIGHTS_DIR / base
    run_dir.mkdir(exist_ok=True)
    return run_dir


def update_model(
    run_dir: Path,
    model: "PINN",
    metadata: dict[str, Any],
) -> tuple[str, str]:
    """Overwrite the best model weights and metadata inside an existing run directory."""
    base = run_dir.name
    pt_path = run_dir / f"{base}.pt"
    json_path = run_dir / f"{base}.json"
    torch.save(model.state_dict(), pt_path)
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return str(pt_path), str(json_path)


def load_metadata(weights_path: str) -> dict[str, Any]:
    """Load metadata JSON for a weights file."""
    p = Path(weights_path)
    json_path = p.with_suffix(".json")
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        return json.load(f)


def list_weights(equation_type: str | None = None) -> list[str]:
    """List available .pt files in weights directory, optionally filtered by equation type."""
    _ensure_weights_dir()
    # Search recursively to include per-training subfolders
    all_pts = list(WEIGHTS_DIR.rglob("*.pt"))
    # Filter for files that exist and optionally match equation type
    valid_paths = [p for p in all_pts if p.is_file()]
    if equation_type:
        prefix = f"{equation_type}_"
        valid_paths = [p for p in valid_paths if p.name.startswith(prefix)]
    
    # Sort by modification time, newest first
    valid_paths.sort(key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True)
    return [str(p) for p in valid_paths]
