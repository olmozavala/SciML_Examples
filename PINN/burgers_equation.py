"""Burgers equation PINN: IC, BC, and compute_loss.

Code based on https://github.com/madagra/basic-pinn
"""
import numpy as np
import torch

from pinn_core import PINN, U_call, dfdx, dfdt

device = "cuda" if torch.cuda.is_available() else "cpu"
NU = 0.01 / np.pi

X_DOMAIN = [-1.0, 1.0]
T_DOMAIN = [0.0, 1.0]
NUM_HIDDEN = 2
DIM_HIDDEN = 15


def initial_condition(x: torch.Tensor) -> torch.Tensor:
    """Initial condition: u(x,0) = -sin(pi*x)."""
    return -torch.sin(np.pi * x).reshape(-1, 1)


def boundary_condition(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Homogeneous Dirichlet BC: u = 0 at boundaries."""
    return torch.zeros(x.size(), device=x.device)


def _sample_boundary_points(
    t: torch.Tensor,
    x_min: float,
    x_max: float,
) -> torch.Tensor:
    """Sample collocation points at domain boundaries x=x_min and x=x_max."""
    n = t.numel()
    r = torch.rand(n, device=t.device)
    x_boundary = torch.where(
        r > 0.5,
        torch.full((n,), x_max, device=t.device, dtype=t.dtype),
        torch.full((n,), x_min, device=t.device, dtype=t.dtype),
    ).reshape(-1, 1)
    return x_boundary


def compute_loss(
    U_nn: PINN,
    x: torch.Tensor,
    t: torch.Tensor,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Burgers equation loss: PDE residual + IC + BC.

    PDE: du/dt + u*du/dx - nu*d2u/dx2 = 0 with nu = 0.01/pi
    """
    x_min, x_max = X_DOMAIN[0], X_DOMAIN[1]
    # PDE residual: du/dt + u*du/dx - nu*d2u/dx2
    u_val = U_call(U_nn, x, t)
    res_val = dfdt(U_nn, x, t, order=1) + u_val * dfdx(U_nn, x, t, order=1) - NU * dfdx(U_nn, x, t, order=2)
    interior_loss = res_val.pow(2).mean()

    # BC loss: u(x_min, t) = 0 and u(x_max, t) = 0
    x_boundary = _sample_boundary_points(t, x_min, x_max)
    bnd_val = U_call(U_nn, x_boundary, t) - boundary_condition(x_boundary, t)
    boundary_loss = bnd_val.pow(2).mean()

    # IC loss: u(x, 0) = -sin(pi*x)
    f_initial = initial_condition(x)
    t_initial = torch.zeros_like(x)
    ic_val = U_call(U_nn, x, t_initial) - f_initial
    initial_loss = ic_val.pow(2).mean()

    total_loss = interior_loss + initial_loss + boundary_loss

    if return_components:
        return total_loss, {
            "pde": interior_loss,
            "ic": initial_loss,
            "bc": boundary_loss,
        }
    return total_loss
