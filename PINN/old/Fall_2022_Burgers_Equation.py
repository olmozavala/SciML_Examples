## Code based from the work made here https://github.com/madagra/basic-pinn/blob/main/burgers_equation_1d.py
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import random
import torch
from torch import nn, tensor
import time as t

device = "cuda" if torch.cuda.is_available() else "cpu"
##

def initial_condition(x) -> torch.Tensor:
    res = - torch.sin(np.pi * x).reshape(-1, 1)
    return res

def boundary_condition(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    boundary_value = torch.zeros(x.size()).to(device)  # Here this is the only thing we need
    # In case we had something more complicated
    # boundary_value[x == -1] = tensor([0]) # Value at x == -1
    # boundary_value[x == 1] = tensor([0]) # Value at x == 1
    return boundary_value

class PINN(nn.Module):
    """Simple neural network accepting two features as input and returning a single output

    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh()):
        super().__init__()

        self.layer_in = nn.Linear(2, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 1)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act

    def forward(self, x, t):
        x_stack = torch.cat([x, t], dim=1)
        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        return self.layer_out(out)

def U_call(U_nn: PINN, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""
    return U_nn(x, t)

def df(output: torch.Tensor, input: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    df_value = output
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input,
            grad_outputs=torch.ones_like(input),
            create_graph=True,
            retain_graph=True,
        )[0]

    return df_value

def dfdt(U_nn: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1):
    """Derivative with respect to the time variable of arbitrary order"""
    f_value = U_call(U_nn, x, t)
    return df(f_value, t, order=order)

def dfdx(U_nn: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1):
    """Derivative with respect to the spatial variable of arbitrary order"""
    f_value = U_call(U_nn, x, t)
    return df(f_value, x, order=order)

def compute_loss( U_nn: PINN, x: torch.Tensor = None, t: torch.Tensor = None ) -> torch.float:
    """Compute the full loss function as interior loss + boundary loss

    This custom loss function is fully defined with differentiable tensors therefore
    the .backward() method can be applied to it
    """
    # PDE residual
    interior_loss = dfdt(U_nn, x, t, order=1) + U_call(U_nn, x, t) * dfdx(U_nn, x, t, order=1) - (0.01/np.pi)*dfdx(U_nn, x, t, order=2)

    # Boundary conditions at the domain extrema
    x_boundary = tensor([0 if x > 0.5 else 1 for x in torch.rand(t.size())]).reshape(-1,1).to(device)
    boundary_loss = U_call(U_nn, x_boundary, t) - boundary_condition(x_boundary, t)

    # initial condition loss (this should be only when t = 0)
    f_initial = initial_condition(x)
    t_initial = torch.zeros_like(x)
    initial_loss_f = U_call(U_nn, x, t_initial) - f_initial

    # obtain the final MSE loss by averaging each loss term and summing them up
    final_loss = \
        interior_loss.pow(2).mean() + \
        initial_loss_f.pow(2).mean() + \
        boundary_loss.pow(2).mean()

    return final_loss

def train_model(U_nn: PINN,
                loss_fn: Callable,
                learning_rate: int = 0.01,
                max_epochs: int = 1_000, ) -> PINN:
    optimizer = torch.optim.Adam(U_nn.parameters(), lr=learning_rate)

    for epoch in range(max_epochs):
        try:
            loss: torch.Tensor = loss_fn(U_nn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 1000 == 0:
                print(f"Epoch: {epoch} - Loss: {float(loss):>7f}")

        except KeyboardInterrupt:
            break
    return U_nn

def check_gradient(U_nn: PINN, x: torch.Tensor, t: torch.Tensor) -> bool:
    eps = 1e-4

    dfdx_fd = (U_call(U_nn, x + eps, t) - U_call(U_nn, x - eps, t)) / (2 * eps)
    dfdx_autodiff = dfdx(U_nn, x, t, order=1)
    is_matching_x = torch.allclose(dfdx_fd.T, dfdx_autodiff.T, atol=1e-2, rtol=1e-2)

    dfdt_fd = (U_call(U_nn, x, t + eps) - U_call(U_nn, x, t - eps)) / (2 * eps)
    dfdt_autodiff = dfdt(U_nn, x, t, order=1)
    is_matching_t = torch.allclose(dfdt_fd.T, dfdt_autodiff.T, atol=1e-2, rtol=1e-2)

    eps = 1e-2

    d2fdx2_fd = (U_call(U_nn, x + eps, t) - 2 * U_call(U_nn, x, t) + U_call(U_nn, x - eps, t)) / (
                eps ** 2)
    d2fdx2_autodiff = dfdx(U_nn, x, t, order=2)
    is_matching_x2 = torch.allclose(d2fdx2_fd.T, d2fdx2_autodiff.T, atol=1e-2, rtol=1e-2)

    d2fdt2_fd = (U_call(U_nn, x, t + eps) - 2 * U_call(U_nn, x, t) + U_call(U_nn, x, t - eps)) / (
                eps ** 2)
    d2fdt2_autodiff = dfdt(U_nn, x, t, order=2)
    is_matching_t2 = torch.allclose(d2fdt2_fd.T, d2fdt2_autodiff.T, atol=1e-2, rtol=1e-2)

    return is_matching_x and is_matching_t and is_matching_x2 and is_matching_t2

def plot_locations(ax, t_domain, x_domain, x, t, title="", add_boundary=True, s=5):
    n_t = len(t)
    n_x = len(x)
    eps = .01
    # ax.imshow(np.ones((n_x, n_t)), extent=(x_domain[0]-eps, x_domain[1]+eps, t_domain[0]-eps, t_domain[1]+eps))
    data = []
    data_bnd = []
    for i in range(n_t):
        for j in range(n_x):
            data.append([t[i], x[j]])

    # Adding boundary points
    if add_boundary:
        for i in range(n_t):
            data_bnd.append([t[i], x_domain[0]])
            data_bnd.append([t[i], x_domain[1]])
            data_bnd.append([t_domain[0], x[j]])

        for j in range(n_x):
            data_bnd.append([t[i], x_domain[0]])
            data_bnd.append([t[i], x_domain[1]])
            data_bnd.append([t_domain[0], x[j]])

    data_np = np.array(data)
    ax.scatter(data_np[:,1], data_np[:,0], s=s, c="red")
    ax.set_xlabel("X")
    ax.set_ylabel("Time")
    if add_boundary:
        data_bnd_np = np.array(data_bnd)
        ax.scatter(data_bnd_np[:,1], data_bnd_np[:,0], s=s, c="green")
    ax.set_title(title)
    # for i in range(n_t):
    #     for j in range(n_x):
    #         ax.text(t[i], x[j], "x", color="red")
    #         ax.text(t[i], x_domain[0], "x", color="green")
    #         ax.text(t[i], x_domain[1], "x", color="green")
    #         ax.text(t_domain[0], x[j], "x", color="green")
    # ax.set_title(f"Training locations. Total: {x_np.size*t_np.size + x_np.size*2 + t_np.size}")
    return ax

if __name__ == "__main__":
    from functools import partial

    x_domain = [-1., 1.];
    n_x = 100
    t_domain = [0.0, 1.];
    n_t = 100

    ## Training points
    x_np = np.linspace(x_domain[0], x_domain[1], n_x, dtype=np.float32)
    t_np = np.linspace(t_domain[0], t_domain[1], n_t, dtype=np.float32)
    # x_np = 2*random(n_x).astype(np.float32) - 1
    # t_np = random(n_t).astype(np.float32)

    fig, ax = plt.subplots(1,1, figsize=(6, 6))
    ax = plot_locations(ax, t_domain, x_domain, x_np, t_np, title=f"Training points {n_x*n_t} --> {n_x*n_t*3}")
    plt.show()
    print("Done!")

    ##
    x_raw = torch.tensor(x_np, requires_grad=True).to(device)
    t_raw = torch.tensor(t_np, requires_grad=True).to(device)
    grids = torch.meshgrid(x_raw, t_raw, indexing="ij")

    x_train = grids[0].flatten().reshape(-1, 1).to(device)
    t_train = grids[1].flatten().reshape(-1, 1).to(device)

    ## Initial status
    U_nn = PINN(2, 15).to(device)
    assert check_gradient(U_nn, x_train, t_train)

    ## train the PINN
    loss_fn = partial(compute_loss, x=x_train, t=t_train)  # The object loss_fn will be called with x and t always
    U_nn_trained = train_model(
        U_nn, loss_fn=loss_fn, learning_rate=0.025, max_epochs=5_000
    )
    compute_loss(U_nn_trained, x=x_train, t=t_train)
    print("Done training!")

    ## --------- Evaluation ---------------
    n_x_test = 100
    n_t_test = 100
    x_domain_test = [-1., 1.];
    t_domain_test = [0.0, 1.];
    # Uniformly distributed
    x_np = np.linspace(x_domain_test[0], x_domain_test[1], n_x_test, dtype=np.float32)
    t_np = np.linspace(t_domain_test[0], t_domain_test[1], n_t_test, dtype=np.float32)
    grids = torch.meshgrid(torch.tensor(x_np), torch.tensor(t_np), indexing="ij")
    # Flatten and Send to CUDA
    x_test = grids[0].flatten().reshape(-1, 1).to(device)
    t_test = grids[1].flatten().reshape(-1, 1).to(device)
    print("Done!")

    ##
    u2d = U_nn_trained(x_test, t_test)
    u2d = u2d.to("cpu").detach().numpy()
    u2d = u2d.reshape(n_x_test, n_t_test)
    fig, axs = plt.subplots(2,1, figsize=(8,6), gridspec_kw={'height_ratios': [1, 5]})
    axs[0].plot(x_np, initial_condition(torch.tensor(x_np)).to("cpu").detach().numpy())
    axs[0].set_title("Initial conditions")
    axs[1].imshow(np.flip(np.transpose(u2d)), extent=(min(x_domain_test), max(x_domain_test),
                               min(t_domain_test), max(t_domain_test)))
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Time")
    axs[1].set_title("Trained model")
    plt.show()
    print("Done!")

    ## --------- Intermediate values
    for i in range(0,99,10):
        fig, axs = plt.subplots(2,1, figsize=(6,8), gridspec_kw={'height_ratios': [1, 5]})
        axs[0].plot(x_np, u2d[:,i])
        axs[0].set_ylim([-1.3,1.3])
        axs[0].set_title(F"Time {t_np[i]:0.2f}")
        axs[1].imshow(np.flipud(np.transpose(u2d)), extent=(min(x_domain), max(x_domain),
                               min(t_domain), max(t_domain)), cmap="inferno")

        axs[1].set_aspect(5)
        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Time")
        axs[1].set_title("Trained model")
        axs[1].plot([-1,1],[t_np[i],t_np[i]], c="red")
        plt.show()
        t.sleep(.2)
    print("Done!")


    ## Evaluating outside the trained domain
    ## --------- Evaluation ---------------
    n_x_test = 100
    n_t_test = 100
    x_domain_test = [-1., 1.];
    t_domain_test = [1., 2.];
    # Uniformly distributed
    x_np = np.linspace(x_domain_test[0], x_domain_test[1], n_x_test, dtype=np.float32)
    t_np = np.linspace(t_domain_test[0], t_domain_test[1], n_t_test, dtype=np.float32)
    grids = torch.meshgrid(torch.tensor(x_np), torch.tensor(t_np), indexing="ij")
    # Flatten and Send to CUDA
    x_test = grids[0].flatten().reshape(-1, 1).to(device)
    t_test = grids[1].flatten().reshape(-1, 1).to(device)

    print("Done!")

    ##
    u2d_outside = U_nn_trained(x_test, t_test)
    u2d_outside = u2d_outside.to("cpu").detach().numpy()
    u2d_outside = u2d_outside.reshape(n_x_test, n_t_test)
    u2d_final = np.concatenate([u2d, u2d_outside], axis=1)
    fig, axs = plt.subplots(2,1, figsize=(8,6), gridspec_kw={'height_ratios': [1, 5]})
    axs[0].plot(x_np, initial_condition(torch.tensor(x_np)).to("cpu").detach().numpy())
    axs[0].set_title("Initial conditions")
    axs[1].imshow(np.flip(np.transpose(u2d_final)), extent=(min(x_domain_test), max(x_domain_test),
                               min(t_domain), max(t_domain_test)))
    axs[1].plot([-1,1], [1,1], c="red")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Time")
    axs[1].set_title("Trained model")
    plt.show()
    print("Done!")
##

