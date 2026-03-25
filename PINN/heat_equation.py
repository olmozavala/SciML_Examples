"""Heat equation PINN: IC, BC, and compute_loss.

Code based on https://github.com/madagra/basic-pinn
"""
import numpy as np
import torch

from pinn_core import PINN, U_call, dfdx, dfdt

device = "cuda" if torch.cuda.is_available() else "cpu"

X_DOMAIN = [0.0, 10.0]
T_DOMAIN = [0.0, 1.0]
ALPHA = 1.1
NUM_HIDDEN = 3
DIM_HIDDEN = 15


def initial_condition(x: torch.Tensor) -> torch.Tensor:
    """Initial condition: sum of two Gaussians centered at x=4 and x=8."""
    res = (
        torch.exp(-(x - 4) ** 2).reshape(-1, 1)
        + 2 * torch.exp(-(x - 8) ** 2).reshape(-1, 1)
    )
    return res


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
    """Compute heat equation loss: PDE residual + IC + BC.

    PDE: du/dt - alpha * d2u/dx2 = 0 with alpha = 1.1
    """
    x_min, x_max = X_DOMAIN[0], X_DOMAIN[1]
    # PDE residual: du/dt - alpha * d2u/dx2
    res_val = dfdt(U_nn, x, t, order=1) - ALPHA * dfdx(U_nn, x, t, order=2)
    interior_loss = res_val.pow(2).mean()

    # BC loss: u(x_min, t) = 0 and u(x_max, t) = 0
    x_boundary = _sample_boundary_points(t, x_min, x_max)
    bnd_val = U_call(U_nn, x_boundary, t) - boundary_condition(x_boundary, t)
    boundary_loss = bnd_val.pow(2).mean()

    # IC loss: u(x, 0) = initial_condition(x)
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
