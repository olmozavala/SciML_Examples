"""Shared GPU-only neural operator models and training utilities."""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from operator_data import relative_l2_error


def require_cuda_device() -> torch.device:
    """Return the CUDA device or fail fast if no GPU is available."""
    if not torch.cuda.is_available():
        raise RuntimeError("This module requires a CUDA-capable GPU. No CUDA device was detected.")
    return torch.device("cuda")


class MLP(nn.Module):
    """Small MLP used in DeepONet branch/trunk networks."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class DeepONet(nn.Module):
    """Branch/trunk operator architecture."""

    def __init__(self, n_sensors: int, width: int = 64, latent_dim: int = 64) -> None:
        super().__init__()
        self.branch_net = MLP(n_sensors, width, latent_dim, depth=3)
        self.trunk_net = MLP(1, width, latent_dim, depth=3)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, forcing_samples: torch.Tensor, x_query: torch.Tensor) -> torch.Tensor:
        branch_features = self.branch_net(forcing_samples)
        trunk_features = self.trunk_net(x_query)
        return (branch_features * trunk_features).sum(dim=-1, keepdim=True) + self.bias


class SpectralConv1d(nn.Module):
    """1D Fourier layer keeping only the lowest modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        scale = 1.0 / max(in_channels * out_channels, 1)
        weight = scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        self.weight = nn.Parameter(weight)
        self.out_channels = out_channels
        self.modes = modes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, n_points = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        active_modes = min(self.modes, x_ft.size(-1))
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x_ft.size(-1),
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :active_modes] = torch.einsum(
            "bim,iom->bom",
            x_ft[:, :, :active_modes],
            self.weight[:, :, :active_modes],
        )
        return torch.fft.irfft(out_ft, n=n_points, dim=-1)


class FNOBlock1d(nn.Module):
    """Fourier layer plus local mixing."""

    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.local = nn.Conv1d(width, width, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral(x) + self.local(x))


class FNO1d(nn.Module):
    """Minimal 1D Fourier Neural Operator."""

    def __init__(self, input_dim: int = 2, width: int = 32, modes: int = 12, depth: int = 4) -> None:
        super().__init__()
        self.lift = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(depth)])
        self.project = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.lift(inputs).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)
        return self.project(x).squeeze(-1)


def _epoch_log(epoch: int, epochs: int, log_interval: int) -> bool:
    return epoch == 1 or epoch % log_interval == 0 or epoch == epochs


def train_deeponet(
    model: DeepONet,
    train_data: dict[str, torch.Tensor],
    device: torch.device,
    epochs: int = 250,
    learning_rate: float = 1e-3,
    batch_size: int = 1024,
    log_interval: int = 25,
    callback: Callable[[int, float], None] | None = None,
) -> list[dict[str, float]]:
    """Train DeepONet on pointwise pairs."""
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
        if _epoch_log(epoch, epochs, log_interval):
            record = {"epoch": float(epoch), "train_loss": avg_loss}
            history.append(record)
            if callback is not None:
                callback(epoch, avg_loss)

    return history


@torch.no_grad()
def predict_deeponet(
    model: DeepONet,
    data: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Predict full solution fields with DeepONet."""
    model.eval()
    forcing = data["forcing"].to(device)
    x_grid = data["x"].to(device)
    predictions = []
    for sample_id in range(forcing.shape[0]):
        forcing_sample = forcing[sample_id].unsqueeze(0).repeat(x_grid.shape[1], 1)
        x_query = x_grid[sample_id]
        pred = model(forcing_sample, x_query).squeeze(-1)
        predictions.append(pred)
    return torch.stack(predictions, dim=0)


def train_fno(
    model: FNO1d,
    train_data: dict[str, torch.Tensor],
    device: torch.device,
    epochs: int = 300,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    log_interval: int = 25,
    callback: Callable[[int, float], None] | None = None,
) -> list[dict[str, float]]:
    """Train an FNO on full function batches."""
    forcing = train_data["forcing"].to(device)
    x_grid = train_data["x"].to(device)
    solution = train_data["solution"].to(device)
    inputs = torch.cat([forcing.unsqueeze(-1), x_grid], dim=-1)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    loss_fn = nn.MSELoss()
    history: list[dict[str, float]] = []
    n_total = inputs.shape[0]

    model.train()
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(n_total, device=device)
        epoch_loss = 0.0

        for start in range(0, n_total, batch_size):
            idx = permutation[start : start + batch_size]
            input_batch = inputs[idx]
            target_batch = solution[idx]

            optimizer.zero_grad(set_to_none=True)
            prediction = model(input_batch)
            loss = loss_fn(prediction, target_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item()) * input_batch.size(0)

        avg_loss = epoch_loss / max(n_total, 1)
        if _epoch_log(epoch, epochs, log_interval):
            record = {"epoch": float(epoch), "train_loss": avg_loss}
            history.append(record)
            if callback is not None:
                callback(epoch, avg_loss)

    return history


@torch.no_grad()
def predict_fno(
    model: FNO1d,
    data: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Predict full solution fields with an FNO."""
    model.eval()
    inputs = torch.cat([data["forcing"].unsqueeze(-1), data["x"]], dim=-1).to(device)
    return model(inputs)


def summarize_prediction(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute scalar evaluation metrics."""
    mse = nn.functional.mse_loss(prediction, target)
    rel_error = relative_l2_error(prediction, target)
    return {
        "mse": float(mse.detach().item()),
        "relative_l2": rel_error,
    }
