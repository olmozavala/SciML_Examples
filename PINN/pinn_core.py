"""Shared PINN components: network, autodiff helpers, and training.

Code based on https://github.com/madagra/basic-pinn
"""
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class PINN(nn.Module):
    """Physics-Informed Neural Network: 2 inputs (x, t), 1 output u(x,t)."""

    def __init__(
        self,
        num_hidden: int,
        dim_hidden: int,
        act: nn.Module = None,
    ) -> None:
        super().__init__()
        if act is None:
            act = nn.Tanh()
        self.layer_in = nn.Linear(2, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 1)
        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_stack = torch.cat([x, t], dim=1)
        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        return self.layer_out(out)


def U_call(U_nn: PINN, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate the PINN at (x, t)."""
    return U_nn(x, t)


def df(
    output: torch.Tensor,
    input_tensor: torch.Tensor,
    order: int = 1,
) -> torch.Tensor:
    """Compute derivative of output w.r.t. input_tensor using autograd."""
    df_value = output
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input_tensor,
            grad_outputs=torch.ones_like(input_tensor),
            create_graph=True,
            retain_graph=True,
        )[0]
    return df_value


def dfdt(
    U_nn: PINN,
    x: torch.Tensor,
    t: torch.Tensor,
    order: int = 1,
) -> torch.Tensor:
    """Compute partial derivative of u w.r.t. t."""
    f_value = U_call(U_nn, x, t)
    return df(f_value, t, order=order)


def dfdx(
    U_nn: PINN,
    x: torch.Tensor,
    t: torch.Tensor,
    order: int = 1,
) -> torch.Tensor:
    """Compute partial derivative of u w.r.t. x."""
    f_value = U_call(U_nn, x, t)
    return df(f_value, x, order=order)


def train_model(
    U_nn: PINN,
    loss_fn: Callable,
    learning_rate: float = 0.01,
    max_epochs: int = 1_000,
    on_epoch_callback: Optional[Callable[[int, float], None]] = None,
    callback_interval: int = 10,
    patience: int = 0,
    val_loss_fn: Optional[Callable[["PINN"], float]] = None,
    adaptive_weights: bool = False,
) -> "PINN":
    """Train the PINN with optional early stopping and adaptive loss weighting.

    Args:
        adaptive_weights: If True, uses gradient-based learning rate annealing to balance
                         PDE residual vs boundary/initial losses.
    """
    optimizer = torch.optim.Adam(U_nn.parameters(), lr=learning_rate)

    best_val = float("inf")
    no_improve_count = 0

    # Weights for adaptive weighting (initially 1.0)
    weights = {"pde": 1.0, "ic": 1.0, "bc": 1.0, "conservation": 1.0}
    alpha_anneal = 0.1  # Smoothing factor for weight updates

    for epoch in range(max_epochs):
        try:
            optimizer.zero_grad()

            # Handle both single-tensor and (tensor, dict) return types
            res = loss_fn(U_nn)
            if isinstance(res, tuple):
                loss, components = res
            else:
                loss, components = res, None

            if adaptive_weights and components:
                # Update weights every 10 epochs (Annealing algorithm)
                if epoch % 10 == 0 and epoch > 0:
                    # 1. Compute gradients for each component
                    grads = {}
                    for head in ["pde", "ic", "bc", "conservation"]:
                        if head in components:
                            U_nn.zero_grad()
                            components[head].backward(retain_graph=True)
                            grad_norms = []
                            for p in U_nn.parameters():
                                if p.grad is not None:
                                    grad_norms.append(p.grad.abs().mean())
                            if grad_norms:
                                grads[head] = torch.stack(grad_norms).mean()

                    # 2. Update weights: weight = (1-alpha)*weight + alpha * (mean_pde_grad / mean_head_grad)
                    if "pde" in grads:
                        mean_grad_pde = grads["pde"]
                        for head in ["ic", "bc", "conservation"]:
                            if head in grads and grads[head] > 0:
                                hat_lambda = mean_grad_pde / grads[head]
                                weights[head] = (1 - alpha_anneal) * weights[head] + alpha_anneal * hat_lambda

                # 3. Apply weights to computed loss
                weighted_loss = components["pde"]
                for head in ["ic", "bc", "conservation"]:
                    if head in components:
                        weighted_loss += weights[head] * components[head]
                loss = weighted_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val = float(loss.detach())

            is_callback_step = epoch % callback_interval == 0 or epoch == max_epochs - 1

            if on_epoch_callback and is_callback_step:
                on_epoch_callback(epoch, loss_val)

            if patience > 0 and val_loss_fn is not None and is_callback_step:
                val_loss = val_loss_fn(U_nn)
                if val_loss < best_val:
                    best_val = val_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        break

        except KeyboardInterrupt:
            break
    return U_nn



def check_gradient(
    U_nn: PINN,
    x: torch.Tensor,
    t: torch.Tensor,
) -> bool:
    """Verify autograd gradients against finite differences."""
    eps = 1e-4
    dfdx_fd = (U_call(U_nn, x + eps, t) - U_call(U_nn, x - eps, t)) / (2 * eps)
    dfdx_autodiff = dfdx(U_nn, x, t, order=1)
    is_matching_x = torch.allclose(dfdx_fd.T, dfdx_autodiff.T, atol=1e-2, rtol=1e-2)
    dfdt_fd = (U_call(U_nn, x, t + eps) - U_call(U_nn, x, t - eps)) / (2 * eps)
    dfdt_autodiff = dfdt(U_nn, x, t, order=1)
    is_matching_t = torch.allclose(dfdt_fd.T, dfdt_autodiff.T, atol=1e-2, rtol=1e-2)
    eps = 1e-2
    d2fdx2_fd = (
        U_call(U_nn, x + eps, t)
        - 2 * U_call(U_nn, x, t)
        + U_call(U_nn, x - eps, t)
    ) / (eps**2)
    d2fdx2_autodiff = dfdx(U_nn, x, t, order=2)
    is_matching_x2 = torch.allclose(d2fdx2_fd.T, d2fdx2_autodiff.T, atol=1e-2, rtol=1e-2)
    d2fdt2_fd = (
        U_call(U_nn, x, t + eps)
        - 2 * U_call(U_nn, x, t)
        + U_call(U_nn, x, t - eps)
    ) / (eps**2)
    d2fdt2_autodiff = dfdt(U_nn, x, t, order=2)
    is_matching_t2 = torch.allclose(d2fdt2_fd.T, d2fdt2_autodiff.T, atol=1e-2, rtol=1e-2)
    return is_matching_x and is_matching_t and is_matching_x2 and is_matching_t2


def compute_conservation_loss(
    U_nn: PINN,
    equation: str,
    x_domain: list[float],
    t: torch.Tensor,
) -> torch.Tensor:
    """Compute global conservation loss: d/dt Integral(u dx) = Flux_in - Flux_out.

    Enforces that the total 'mass' evolves according to the boundary fluxes.
    """
    # 1. Sample a batch of t and compute integral over x at each t
    # We use a simple midpoint rule or mean for the integral
    nx_integral = 50
    x_int = torch.linspace(x_domain[0], x_domain[1], nx_integral, device=device).reshape(-1, 1)
    
    loss_vals = []
    alpha = 1.1 if equation == "heat" else 0.0
    nu = (0.01 / np.pi) if equation == "burgers" else 0.0

    # We check conservation at the provided t points
    for ti in t.unique()[:10]: # Check a few time slices
        t_slice = torch.full_like(x_int, ti.item(), requires_grad=True)
        u_slice = U_nn(x_int, t_slice)
        
        # Total mass at time ti
        total_mass = u_slice.mean() * (x_domain[1] - x_domain[0])
        
        # Time derivative of mass (estimated by autograd)
        # We need the gradient of the *integral* w.r.t. t
        # (This is a simplified proxy for the full flux-balance law)
        # For cPINN, we can also just enforce that I(t) matches I(0) + Flux_accumulated
        
        # Simpler: Enforce Flux Balance directly if possible, 
        # but Flux depends on u_x at boundaries.
        xm = torch.tensor([[x_domain[0]], [x_domain[1]]], device=device, requires_grad=True)
        tm = torch.full((2, 1), ti.item(), device=device, requires_grad=True)
        um = U_nn(xm, tm)
        uxm = df(um, xm, order=1)
        
        if equation == "heat":
            # dM/dt = alpha * (ux(R) - ux(L))
            flux_balance = alpha * (uxm[1] - uxm[0])
            # We can't easily get dM/dt without another derivative, 
            # so we enforce u_t integral = flux_balance
            ut_integral = df(u_slice, t_slice, order=1).mean() * (x_domain[1] - x_domain[0])
            loss_vals.append((ut_integral - flux_balance).pow(2))
        else:
            # Burgers: dM/dt = nu*(ux(R)-ux(L)) - 0.5*(u(R)^2 - u(L)^2)
            flux_balance = nu * (uxm[1] - uxm[0]) - 0.5 * (um[1]**2 - um[0]**2)
            ut_integral = df(u_slice, t_slice, order=1).mean() * (x_domain[1] - x_domain[0])
            loss_vals.append((ut_integral - flux_balance).pow(2))

    if not loss_vals:
        return torch.tensor(0.0, device=device)
    return torch.stack(loss_vals).mean()


def compute_vpinn_loss(
    U_nn: PINN,
    equation: str,
    x: torch.Tensor,
    t: torch.Tensor,
    test_fns: list[Callable],
) -> torch.Tensor:
    """Compute weak-form (variational) residual loss.

    Uses a Petrov-Galerkin approach: integral(Res * v) = 0 for all test functions v.
    """
    # 1. Evaluate U and its first derivatives on the grid
    u = U_nn(x, t)
    ux = df(u, x, order=1)
    ut = df(u, t, order=1)

    weak_residuals = []
    
    # Constants
    alpha = 1.1 if equation == "heat" else 0.0
    nu = (0.01 / np.pi) if equation == "burgers" else 0.0

    for v_fn in test_fns:
        # 2. Evaluate test function and its derivatives
        v = v_fn(x, t)
        vx = df(v, x, order=1)
        
        # 3. Compute weak-form integrand
        if equation == "heat":
            # ut*v + alpha*ux*vx (after IBP)
            integrand = ut * v + alpha * ux * vx
        else:
            # Burgers: (ut + u*ux)*v + nu*ux*vx (after IBP)
            integrand = (ut + u * ux) * v + nu * ux * vx
        
        # 4. Monte-Carlo or trapezoidal integration (mean)
        weak_residuals.append(integrand.mean())

    # Total weak loss is the sum of squared test residuals
    return torch.stack(weak_residuals).pow(2).mean()


def get_training_points_plotly(
    t_domain: list[float],
    x_domain: list[float],
    x_np: np.ndarray,
    t_np: np.ndarray,
    add_boundary: bool = True,
) -> list[dict]:
    """Return Plotly scatter traces for interior (red) and boundary (green) points."""
    data_interior = []
    for i in range(len(t_np)):
        for j in range(len(x_np)):
            data_interior.append({"x": float(x_np[j]), "t": float(t_np[i])})
    traces = [
        {
            "x": [d["x"] for d in data_interior],
            "y": [d["t"] for d in data_interior],
            "mode": "markers",
            "name": "Interior",
            "marker": {"size": 3, "color": "red"},
        }
    ]
    if add_boundary:
        data_bnd = []
        for i in range(len(t_np)):
            data_bnd.append({"x": x_domain[0], "t": float(t_np[i])})
            data_bnd.append({"x": x_domain[1], "t": float(t_np[i])})
        for j in range(len(x_np)):
            data_bnd.append({"x": float(x_np[j]), "t": t_domain[0]})
        traces.append(
            {
                "x": [d["x"] for d in data_bnd],
                "y": [d["t"] for d in data_bnd],
                "mode": "markers",
                "name": "Boundary",
                "marker": {"size": 3, "color": "green"},
            }
        )
    return traces


def legendre_poly(x: torch.Tensor, n: int) -> torch.Tensor:
    """Evaluate n-th order Legendre polynomial on domain [-1, 1]."""
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return x

    p0 = torch.ones_like(x)
    p1 = x
    for i in range(2, n + 1):
        pi = ((2 * i - 1) * x * p1 - (i - 1) * p0) / i
        p0, p1 = p1, pi
    return p1


def get_vpinn_test_functions(
    n_test_x: int,
    n_test_t: int,
    x_range: list[float],
    t_range: list[float],
) -> list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Generate a library of test functions (tensor product of polynomials).

    Maps x to [-1, 1] and t to [-1, 1] before applying Legendre polynomials.
    Ensures test functions vanish at boundaries for weak form.
    """
    test_fns = []

    def map_to_unit(val, vmin, vmax):
        return 2 * (val - vmin) / (vmax - vmin) - 1

    for i in range(n_test_x):
        for j in range(n_test_t):

            def test_fn(x, t, poly_i=i, poly_j=j):
                # Map to [-1, 1]
                u_x = map_to_unit(x, x_range[0], x_range[1])
                u_t = map_to_unit(t, t_range[0], t_range[1])
                # Vanishing at boundaries: (1-x^2)(1-t^2) * P_i(x) * P_j(t)
                weight = (1 - u_x**2) * (1 - u_t**2)
                return weight * legendre_poly(u_x, poly_i) * legendre_poly(u_t, poly_j)

            test_fns.append(test_fn)

    return test_fns


def get_adaptive_points(
    U_nn: PINN,
    residual_fn: Callable[[PINN, torch.Tensor, torch.Tensor], torch.Tensor],
    x_domain: list[float],
    t_domain: list[float],
    n_points: int,
    pool_size: int = 10_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual-based adaptive sampling.

    Samples a large pool of points, evaluates the PDE residual, and returns the
    n_points with the highest absolute residual.
    """
    # 1. Sample a large pool of random points
    x_pool = torch.rand(pool_size, 1, device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
    t_pool = torch.rand(pool_size, 1, device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]
    x_pool.requires_grad = True
    t_pool.requires_grad = True

    # 2. Evaluate residual (U_nn is on device already)
    with torch.set_grad_enabled(True):
        res = residual_fn(U_nn, x_pool, t_pool)
        abs_res = res.detach().abs().squeeze()

    # 3. Pick the top N points (clamp so k never exceeds pool size)
    n_points = min(n_points, abs_res.shape[0])
    _, idx = torch.topk(abs_res, n_points)
    return x_pool[idx].detach(), t_pool[idx].detach()
