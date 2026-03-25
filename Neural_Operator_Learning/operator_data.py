"""Shared dataset generation for simple neural operator examples."""
from __future__ import annotations

import math

import numpy as np
import torch


def set_seed(seed: int = 7) -> None:
    """Set numpy and torch RNG seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_grid(n_points: int) -> np.ndarray:
    """Uniform grid on [0, 1]."""
    return np.linspace(0.0, 1.0, n_points, dtype=np.float32)


def sample_sine_coefficients(
    n_samples: int,
    n_modes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw random sine-series coefficients with decaying amplitude."""
    mode_ids = np.arange(1, n_modes + 1, dtype=np.float32)
    scales = 1.0 / mode_ids**2
    coeffs = rng.normal(loc=0.0, scale=scales, size=(n_samples, n_modes))
    return coeffs.astype(np.float32)


def evaluate_sine_series(coeffs: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Evaluate sum_k coeff_k * sin(k*pi*x) on the grid."""
    mode_ids = np.arange(1, coeffs.shape[1] + 1, dtype=np.float32)
    basis = np.sin(np.pi * np.outer(mode_ids, grid)).astype(np.float32)
    return coeffs @ basis


def solve_poisson_from_coeffs(coeffs: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Exact Poisson solution for the sine-series forcing basis.

    If f(x) = sum_k a_k sin(k*pi*x), then
    u(x) = sum_k a_k / (k*pi)^2 * sin(k*pi*x)
    """
    mode_ids = np.arange(1, coeffs.shape[1] + 1, dtype=np.float32)
    solution_coeffs = coeffs / ((np.pi * mode_ids) ** 2)
    return evaluate_sine_series(solution_coeffs.astype(np.float32), grid)


def make_dataset(
    n_samples: int,
    n_points: int = 64,
    n_modes: int = 6,
    seed: int = 7,
) -> dict[str, torch.Tensor]:
    """Create forcing/solution pairs on a fixed grid."""
    rng = np.random.default_rng(seed)
    grid = build_grid(n_points)
    coeffs = sample_sine_coefficients(n_samples, n_modes, rng)
    forcing = evaluate_sine_series(coeffs, grid)
    solution = solve_poisson_from_coeffs(coeffs, grid)

    x_grid = np.repeat(grid[None, :], n_samples, axis=0)
    return {
        "x": torch.tensor(x_grid[..., None], dtype=torch.float32),
        "forcing": torch.tensor(forcing, dtype=torch.float32),
        "solution": torch.tensor(solution, dtype=torch.float32),
    }


def relative_l2_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Return relative L2 error."""
    numerator = torch.linalg.norm(prediction - target)
    denominator = torch.linalg.norm(target).clamp_min(1e-12)
    return float((numerator / denominator).item())


def describe_dataset(data: dict[str, torch.Tensor]) -> str:
    """Short human-readable dataset summary."""
    n_samples, n_points = data["forcing"].shape
    return f"{n_samples} samples on a {n_points}-point grid"
