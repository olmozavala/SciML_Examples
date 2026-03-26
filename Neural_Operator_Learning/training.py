"""Training utilities for neural operator workflows."""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def _epoch_log(epoch: int, epochs: int, log_interval: int) -> bool:
    return epoch == 1 or epoch % log_interval == 0 or epoch == epochs


def train_deeponet(
    model: nn.Module,
    train_data: dict[str, torch.Tensor],
    device: torch.device,
    epochs: int = 250,
    learning_rate: float = 1e-3,
    batch_size: int = 1024,
    log_interval: int = 25,
    callback: Callable[[int, float, float | None, float], None] | None = None,
    val_loss_fn: Callable[[], float] | None = None,
    early_stopping_patience: int = 0,
    lr_scheduler_patience: int = 0,
    lr_scheduler_factor: float = 0.5,
    min_lr: float = 1e-6,
) -> list[dict[str, float]]:
    """Train DeepONet on pointwise forcing-location samples."""
    forcing = train_data["forcing"].to(device)
    x_grid = train_data["x"].to(device)
    solution = train_data["solution"].to(device)
    n_samples, n_points = forcing.shape

    forcing_pairs = forcing.unsqueeze(1).repeat(1, n_points, 1).reshape(n_samples * n_points, n_points)
    x_pairs = x_grid.reshape(n_samples * n_points, 1)
    y_pairs = solution.reshape(n_samples * n_points, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    history: list[dict[str, float]] = []
    n_total = forcing_pairs.shape[0]

    scheduler = None
    if val_loss_fn is not None and lr_scheduler_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            min_lr=min_lr,
        )
    best_metric = float("inf")
    no_improve_count = 0

    model.train()
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(n_total, device=device)
        epoch_loss = 0.0

        for start in range(0, n_total, batch_size):
            idx = permutation[start : start + batch_size]
            forcing_batch = forcing_pairs[idx]
            x_batch = x_pairs[idx]
            y_batch = y_pairs[idx]

            optimizer.zero_grad(set_to_none=True)
            prediction = model(forcing_batch, x_batch)
            loss = loss_fn(prediction, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item()) * forcing_batch.size(0)

        avg_loss = epoch_loss / max(n_total, 1)
        val_loss = None
        monitor_metric = avg_loss
        if val_loss_fn is not None:
            val_loss = float(val_loss_fn())
            monitor_metric = val_loss
        if scheduler is not None:
            scheduler.step(monitor_metric)
        if monitor_metric < best_metric - 1e-12:
            best_metric = monitor_metric
            no_improve_count = 0
        else:
            no_improve_count += 1
        current_lr = float(optimizer.param_groups[0]["lr"])

        if _epoch_log(epoch, epochs, log_interval):
            record = {"epoch": float(epoch), "train_loss": avg_loss, "lr": current_lr}
            if val_loss is not None:
                record["val_loss"] = val_loss
            history.append(record)
            if callback is not None:
                callback(epoch, avg_loss, val_loss, current_lr)

        if early_stopping_patience > 0 and no_improve_count >= early_stopping_patience:
            break

    return history

