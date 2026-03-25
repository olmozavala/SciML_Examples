"""Reference solutions for Heat and Burgers equations.

Used for ground-truth comparison in the Dash Comparison tab.
"""
import numpy as np


def heat_exact_solution(
    x: np.ndarray,
    t: np.ndarray,
    alpha: float = 1.1,
) -> np.ndarray:
    """Exact solution of the heat equation with IC u(x,0) = exp(-(x-4)^2) + 2*exp(-(x-8)^2).

    For a Gaussian IC exp(-(x-a)^2), the solution is
    (1/sqrt(1+4*alpha*t)) * exp(-(x-a)^2/(1+4*alpha*t)).

    Args:
        x: Spatial coordinates, shape (N,) or (Nx, Nt).
        t: Time coordinates, shape (N,) or (Nx, Nt). Use np.broadcast with x.
        alpha: Thermal diffusivity (default 1.1).

    Returns:
        Solution u(x,t), same shape as the broadcast of x and t.
    """
    x_arr = np.asarray(x, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    denom = 1.0 + 4 * alpha * t_arr
    u1 = (1.0 / np.sqrt(denom)) * np.exp(-((x_arr - 4) ** 2) / denom)
    u2 = (2.0 / np.sqrt(denom)) * np.exp(-((x_arr - 8) ** 2) / denom)
    return u1 + u2


def burgers_reference_solution(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    nu: float = 0.01 / np.pi,
    n_steps: int = 5000,
) -> np.ndarray:
    """Reference solution for Burgers equation via high-resolution finite difference.

    PDE: u_t + u*u_x = nu*u_xx
    IC: u(x,0) = -sin(pi*x)
    BC: u(-1,t) = u(1,t) = 0

    Uses upwind scheme for advection and implicit Euler for diffusion.

    Args:
        x_grid: 1D array of spatial points (uniform).
        t_grid: 1D array of time points (uniform).
        nu: Viscosity (default 0.01/pi).
        n_steps: Number of time steps for internal integration (fine resolution).

    Returns:
        Array of shape (len(x_grid), len(t_grid)) with u(x,t).
    """
    x = np.asarray(x_grid, dtype=float).flatten()
    t = np.asarray(t_grid, dtype=float).flatten()
    n_x = len(x)
    n_t = len(t)
    dx = (x[-1] - x[0]) / (n_x - 1) if n_x > 1 else 1.0
    t_max = t[-1]
    dt = t_max / n_steps

    u_history = np.zeros((n_x, n_t))
    u_history[:, 0] = -np.sin(np.pi * x)
    u = u_history[:, 0].copy()
    t_idx = 1

    for step in range(1, n_steps + 1):
        t_curr = step * dt

        # Upwind for u*u_x
        u_x = np.zeros_like(u)
        u_x[1:-1] = np.where(
            u[1:-1] >= 0,
            (u[1:-1] - u[:-2]) / dx,
            (u[2:] - u[1:-1]) / dx,
        )
        u_x[0] = u_x[1]
        u_x[-1] = u_x[-2]

        # Implicit diffusion
        D = nu * dt / (dx**2)
        diag = 1 + 2 * D
        off = -D
        A = (
            np.diag(diag * np.ones(n_x))
            + np.diag(off * np.ones(n_x - 1), 1)
            + np.diag(off * np.ones(n_x - 1), -1)
        )
        A[0, :] = 0
        A[0, 0] = 1
        A[-1, :] = 0
        A[-1, -1] = 1

        rhs = u - dt * u * u_x
        rhs[0] = 0
        rhs[-1] = 0
        u = np.linalg.solve(A, rhs)

        while t_idx < n_t and t[t_idx] <= t_curr:
            u_history[:, t_idx] = u
            t_idx += 1
        if t_idx >= n_t:
            break

    for i in range(t_idx, n_t):
        u_history[:, i] = u

    return u_history
