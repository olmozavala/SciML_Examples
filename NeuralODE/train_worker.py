"""Background training worker.

Called by neural_ode_dash_app.py via subprocess.Popen. Runs completely
independently so the Flask server thread is never blocked.

Usage:
    python train_worker.py <method_id> <task> <epochs> <lr> <solver>
"""
import json
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from neural_ode_core import NeuralODE, NeuralODEClassifier, device
from spiral_example import get_spiral_data
from classifier_example import get_concentric_circles

# ── File-based IPC ────────────────────────────────────────────────────────────
_lock = threading.Lock()


def _path(method_id: str) -> Path:
    return ROOT / f"loss_history_{method_id}.json"


def _write(data: dict, method_id: str) -> None:
    with _lock:
        with open(_path(method_id), "w") as f:
            json.dump(data, f, indent=2)


def _read(method_id: str) -> dict:
    p = _path(method_id)
    if not p.exists():
        return {"status": "idle", "history": []}
    with _lock:
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"status": "idle", "history": []}


# ── Training ──────────────────────────────────────────────────────────────────
def run_training(method_id: str, task: str, epochs: int, lr: float, solver: str) -> None:
    if task == "spiral":
        t, true_y = get_spiral_data()
        model = NeuralODE(in_dim=2, num_hidden=64, method=solver).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        loss_fn = nn.L1Loss()
        batch_time = 20      # standard segment length
        n_pts = len(t)
    elif task == "classifier":
        X, y = get_concentric_circles()
        model = NeuralODEClassifier(in_dim=2, hidden_dim=16, num_classes=1, method=solver).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown task: {task}")

    def on_epoch(ep: int, loss_val: float) -> None:
        data = _read(method_id)
        if data.get("status") != "training":
            return
        history = data.get("history", [])
        history.append({"epoch": ep, "train_loss": loss_val})
        data["history"] = history
        data["current_epoch"] = ep
        data["total_epochs"] = epochs
        if task == "spiral" and ep % 10 == 0:
            with torch.no_grad():
                pred = model(true_y[0], t).cpu().numpy()
                data["pred_y"] = pred.tolist()
        if task == "classifier" and ep % 10 == 0:
            with torch.no_grad():
                out = model(X).cpu().numpy()
                data["pred_y"] = out.tolist()
        _write(data, method_id)

    _write({"status": "training", "history": [], "current_epoch": 0, "total_epochs": epochs}, method_id)
    print(f"Starting training on device: {device}", flush=True)

    best_loss = float('inf')
    early_stop_counter = 0
    patience = 20

    try:
        for epoch in range(epochs):
            if _read(method_id).get("status") != "training":
                break
            optimizer.zero_grad()
            if task == "spiral":
                s = torch.from_numpy(
                    np.random.choice(np.arange(n_pts - batch_time, dtype=np.int64), 50, replace=False)
                ).to(device)
                batch_y0 = true_y[s].squeeze(1)
                batch_t = t[:batch_time]
                batch_true = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0).squeeze(2)
                pred_y = odeint(model.func, batch_y0, batch_t, method=model.method)
                loss = loss_fn(pred_y, batch_true)
            else:
                pred_y = model(X)
                loss = loss_fn(pred_y, y)
            loss.backward()
            optimizer.step()
            
            # Step scheduler and handle early stopping
            scheduler.step(loss.item())
            if loss.item() < best_loss:
                best_loss = loss.item()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            if early_stop_counter >= patience:
                print(f"[worker:{task}] Early stopping at epoch {epoch}", flush=True)
                break

            if epoch % 10 == 0 or epoch == epochs - 1:
                on_epoch(epoch, loss.item())
                curr_lr = optimizer.param_groups[0]['lr']
                print(f"[worker:{task}] epoch {epoch}/{epochs}  loss={loss.item():.4f}  lr={curr_lr:.6f}", flush=True)

        data = _read(method_id)
        data["status"] = "complete"

        # ── Save model weights ──────────────────────────────────────────────
        try:
            weights_dir = ROOT / "weights"
            weights_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{task}_{solver}_{ts}"
            run_dir = weights_dir / base
            run_dir.mkdir(exist_ok=True)
            pt_path = run_dir / f"{base}.pt"
            json_path = run_dir / f"{base}.json"
            torch.save(model.state_dict(), pt_path)
            history = data.get("history", [])
            final_loss = history[-1]["train_loss"] if history else 0.0
            meta = {"task": task, "solver": solver, "epochs": epochs, "lr": lr,
                    "final_loss": final_loss}
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=2)
            data["weights_path"] = str(pt_path)
        except Exception as we:
            print(f"Warning: could not save weights: {we}")

        _write(data, method_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        data = _read(method_id)
        data["status"] = "error"
        data["error"] = str(e)
        _write(data, method_id)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: train_worker.py <method_id> <task> <epochs> <lr> <solver>")
        sys.exit(1)
    _, method_id, task, epochs_s, lr_s, solver = sys.argv
    run_training(method_id, task, int(epochs_s), float(lr_s), solver)
