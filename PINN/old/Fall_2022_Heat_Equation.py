## Code based from the work made here https://github.com/madagra/basic-pinn/blob/main/burgers_equation_1d.py
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import random
import torch
import time as t
from torch import nn, tensor

device = "cuda" if torch.cuda.is_available() else "cpu"
##

def initial_condition(x: torch.Tensor) -> torch.Tensor:
    """Compute the initial condition for the heat equation.
    
    The initial condition is defined as the sum of two Gaussian functions:
    - One Gaussian centered at x=4 with amplitude 1
    - One Gaussian centered at x=8 with amplitude 2
    
    Args:
        x: Spatial coordinates as a tensor of shape (N, 1) or (N,)
        
    Returns:
        Initial temperature values at each spatial point, shape (N, 1)
    """
    res = torch.exp(-(x - 4)**2).reshape(-1, 1) + 2*torch.exp(-(x - 8)**2).reshape(-1, 1)
    return res

def boundary_condition(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute the boundary condition values for the heat equation.
    
    The boundary condition enforces zero temperature at the domain boundaries.
    This implements homogeneous Dirichlet boundary conditions.
    
    Args:
        x: Spatial coordinates at the boundary, shape (N, 1) or (N,)
        t: Time coordinates, shape (N, 1) or (N,)
        
    Returns:
        Boundary temperature values (all zeros), shape matching x
    """
    boundary_value = torch.zeros(x.size()).to(device)  # Here this is the only thing we need
    return boundary_value

class PINN(nn.Module):
    """Simple neural network accepting two features as input and returning a single output

    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh()):
        """Initialize the Physics-Informed Neural Network.
        
        Creates a fully connected neural network with:
        - Input layer: 2 features (x, t)
        - Hidden layers: num_hidden layers with dim_hidden neurons each
        - Output layer: 1 value (u(x,t))
        
        Args:
            num_hidden: Total number of hidden layers in the network
            dim_hidden: Number of neurons in each hidden layer
            act: Activation function to use (default: Tanh)
        """
        super().__init__()

        self.layer_in = nn.Linear(2, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 1)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network.
        
        Takes spatial (x) and temporal (t) coordinates and returns the
        predicted solution value u(x, t).
        
        Args:
            x: Spatial coordinates, shape (N, 1)
            t: Time coordinates, shape (N, 1)
            
        Returns:
            Predicted solution values u(x, t), shape (N, 1)
        """
        x_stack = torch.cat([x, t], dim=1)
        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        return self.layer_out(out)

def U_call(U_nn: PINN, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model.
    
    Wrapper function that calls the neural network to evaluate the solution
    u(x, t) at given spatial and temporal coordinates.
    
    Args:
        U_nn: The trained PINN model
        x: Spatial coordinates, shape (N, 1)
        t: Time coordinates, shape (N, 1)
        
    Returns:
        Approximate solution values u(x, t), shape (N, 1)
    """
    return U_nn(x, t)

def df(output: torch.Tensor, input: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine.
    
    Computes derivatives of arbitrary order by recursively applying automatic
    differentiation. This is essential for computing PDE residuals in PINNs.
    
    Args:
        output: The output tensor from the neural network (requires_grad=True)
        input: The input tensor with respect to which derivatives are computed
        order: Order of the derivative (1 for first derivative, 2 for second, etc.)
        
    Returns:
        Derivative tensor of the specified order, shape matching input
    """
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

def dfdt(U_nn: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Compute the derivative with respect to the time variable of arbitrary order.
    
    Computes the partial derivative ∂^order u/∂t^order using automatic differentiation.
    This is needed for the time derivative term in the heat equation.
    
    Args:
        U_nn: The PINN model
        x: Spatial coordinates, shape (N, 1)
        t: Time coordinates, shape (N, 1) with requires_grad=True
        order: Order of the time derivative (default: 1)
        
    Returns:
        Time derivative tensor, shape (N, 1)
    """
    f_value = U_call(U_nn, x, t)
    return df(f_value, t, order=order)

def dfdx(U_nn: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Compute the derivative with respect to the spatial variable of arbitrary order.
    
    Computes the partial derivative ∂^order u/∂x^order using automatic differentiation.
    This is needed for the spatial derivative terms in the heat equation.
    
    Args:
        U_nn: The PINN model
        x: Spatial coordinates, shape (N, 1) with requires_grad=True
        t: Time coordinates, shape (N, 1)
        order: Order of the spatial derivative (default: 1, use 2 for second derivative)
        
    Returns:
        Spatial derivative tensor, shape (N, 1)
    """
    f_value = U_call(U_nn, x, t)
    return df(f_value, x, order=order)

def compute_loss( U_nn: PINN, x: torch.Tensor = None, t: torch.Tensor = None ) -> torch.float:
    """Compute the full loss function as interior loss + boundary loss.

    The total loss consists of three components:
    1. Interior loss: PDE residual (∂u/∂t - α*∂²u/∂x²) evaluated at interior points
    2. Initial condition loss: Difference between predicted and true initial condition
    3. Boundary condition loss: Difference between predicted and true boundary values
    
    This custom loss function is fully defined with differentiable tensors, therefore
    the .backward() method can be applied to it for gradient-based optimization.
    
    Args:
        U_nn: The PINN model to evaluate
        x: Spatial coordinates for training points, shape (N, 1)
        t: Time coordinates for training points, shape (N, 1)
        
    Returns:
        Total loss value (scalar tensor) as the sum of MSE losses
    """
    # PDE residual (PINN) -----------------------------------
    interior_loss = dfdt(U_nn, x, t, order=1) - 1.1*dfdx(U_nn, x, t, order=2)

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
                learning_rate: float = 0.01,
                max_epochs: int = 1_000, ) -> PINN:
    """Train the Physics-Informed Neural Network.
    
    Trains the PINN using the Adam optimizer to minimize the combined loss function
    (PDE residual + initial condition + boundary condition). The training can be
    interrupted with Ctrl+C.
    
    Args:
        U_nn: The PINN model to train
        loss_fn: Loss function that takes U_nn as input and returns a scalar loss
        learning_rate: Learning rate for the Adam optimizer (default: 0.01)
        max_epochs: Maximum number of training epochs (default: 1000)
        
    Returns:
        The trained PINN model
    """
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
    """Verify that automatic differentiation gradients match finite difference approximations.
    
    This function performs a gradient check by comparing:
    - First derivatives (∂u/∂x, ∂u/∂t) computed via autograd vs finite differences
    - Second derivatives (∂²u/∂x², ∂²u/∂t²) computed via autograd vs finite differences
    
    This is useful for debugging to ensure the PINN implementation is correct.
    
    Args:
        U_nn: The PINN model to check
        x: Spatial coordinates for testing, shape (N, 1)
        t: Time coordinates for testing, shape (N, 1)
        
    Returns:
        True if all gradient checks pass (within tolerance), False otherwise
    """
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

def plot_locations(ax, t_domain: list, x_domain: list, x: np.ndarray, t: np.ndarray, 
                   title: str = "", add_boundary: bool = True, s: int = 5) -> object:
    """Plot the spatial-temporal locations of training points.
    
    Visualizes where training points are sampled in the (x, t) domain.
    Interior points are shown in red, boundary points (if enabled) in green.
    
    Args:
        ax: Matplotlib axes object to plot on
        t_domain: Time domain bounds [t_min, t_max]
        x_domain: Spatial domain bounds [x_min, x_max]
        x: Spatial coordinates of training points, shape (N,)
        t: Time coordinates of training points, shape (M,)
        title: Title for the plot (default: "")
        add_boundary: Whether to show boundary points in green (default: True)
        s: Marker size for scatter plot (default: 5)
        
    Returns:
        The matplotlib axes object with the plot
    """
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

    return ax

if __name__ == "__main__":
    from functools import partial

    x_domain = [0., 10.];
    n_x = 40
    t_domain = [0.0, 1.];
    n_t = 40

    ## Training points
    x_np = np.linspace(x_domain[0], x_domain[1], n_x, dtype=np.float32)
    t_np = np.linspace(t_domain[0], t_domain[1], n_t, dtype=np.float32)
    # x_np = random(n_x).astype(np.float32)*10
    # t_np = random(n_t).astype(np.float32)

    fig, axs = plt.subplots(1,2, figsize=(10, 5))
    x_del = np.arange(x_domain[0], x_domain[1], .1)
    axs[0].plot(x_del, initial_condition(torch.tensor(x_del)).detach().numpy())
    axs[0].set_title("Initial conditions")
    axs[1] = plot_locations(axs[1], t_domain, x_domain, x_np, t_np, title=f"Training points {n_x*n_t} --> {n_x*n_t*3}")
    plt.show()
    print("Done!")

    ##
    x_raw = torch.tensor(x_np, requires_grad=True).to(device)
    t_raw = torch.tensor(t_np, requires_grad=True).to(device)
    grids = torch.meshgrid(x_raw, t_raw, indexing="ij")

    x_train = grids[0].flatten().reshape(-1, 1).to(device)
    t_train = grids[1].flatten().reshape(-1, 1).to(device)

    ## ############## Network creation #############
    U_nn = PINN(3, 15).to(device)
    assert check_gradient(U_nn, x_train, t_train)

    ## train the PINN
    loss_fn = partial(compute_loss, x=x_train, t=t_train)  # The object loss_fn will be called with x and t always
    U_nn_trained = train_model(
        U_nn, loss_fn=loss_fn, learning_rate=0.025, max_epochs=2_000
    )
    compute_loss(U_nn_trained, x=x_train, t=t_train)
    print("Done training!")

    ## --------- Evaluation ---------------
    n_x_test = 100
    n_t_test = 100
    # Uniformly distributed
    x_np = np.linspace(x_domain[0], x_domain[1], n_x_test, dtype=np.float32)
    t_np = np.linspace(t_domain[0], t_domain[1], n_t_test, dtype=np.float32)
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
    axs[0].set_title("Initial conditions u(x,0)")
    axs[1].imshow(np.flipud(np.transpose(u2d)), extent=(min(x_domain), max(x_domain),
                               min(t_domain), max(t_domain)), cmap="inferno")

    axs[1].set_aspect(5)
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Time")
    axs[1].set_title("Trained model")
    plt.show()
    print("Done!")

    ## --------- Intermediate values
    for i in range(0,99,10):
        print(i)
        fig, axs = plt.subplots(2,1, figsize=(8,6), gridspec_kw={'height_ratios': [1, 5]})
        axs[0].plot(x_np, u2d[:,i])
        axs[0].set_ylim([0,2])
        axs[0].set_title(F"Time {t_np[i]:0.2f}")
        axs[1].imshow(np.flipud(np.transpose(u2d)), extent=(min(x_domain), max(x_domain),
                               min(t_domain), max(t_domain)), cmap="inferno")

        axs[1].set_aspect(5)
        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Time")
        axs[1].set_title("Trained model")
        axs[1].plot([0,10],[t_np[i],t_np[i]], c="red")
        plt.show()
        t.sleep(.3)
    print("Done!")

    ## --------- Evaluation outside domain---------------
    n_x_test = 100
    n_t_test = 100
    x_domain_test = x_domain;
    t_domain_test = [1., 2.];
    # Uniformly distributed
    x_np = np.linspace(x_domain_test[0], x_domain_test[1], n_x_test, dtype=np.float32)
    t_np = np.linspace(t_domain_test[0], t_domain_test[1], n_t_test, dtype=np.float32)
    grids = torch.meshgrid(torch.tensor(x_np), torch.tensor(t_np), indexing="ij")
    # Flatten and Send to CUDA
    x_test = grids[0].flatten().reshape(-1, 1).to(device)
    t_test = grids[1].flatten().reshape(-1, 1).to(device)

    # print("Plotting!")
    # fig, ax = plt.subplots(1,1, figsize=(4, 4))
    # ax = plot_locations(ax, t_domain_test, x_domain_test, x_test, t_test, title=f"Testing locations {n_x_test*n_t_test}",
    #                     add_boundary=False, s=1)
    # plt.show()
    print("Done!")

    ##
    u2d_outside = U_nn_trained(x_test, t_test)
    u2d_outside = u2d_outside.to("cpu").detach().numpy()
    u2d_outside = u2d_outside.reshape(n_x_test, n_t_test)
    u2d_final = np.concatenate([u2d, u2d_outside], axis=1)
    fig, axs = plt.subplots(2,1, figsize=(8,8), gridspec_kw={'height_ratios': [1, 5]})
    axs[0].plot(x_np, initial_condition(torch.tensor(x_np)).to("cpu").detach().numpy())
    axs[0].set_title("Initial conditions")
    axs[1].imshow(np.flipud(np.transpose(u2d_final)), extent=(min(x_domain_test), max(x_domain_test),
                                                        min(t_domain), max(t_domain_test)), cmap="inferno" )
    axs[1].plot([0,10], [1,1], c="red")
    axs[1].set_aspect(5)
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Time")
    axs[1].set_title("Trained model")
    plt.show()
    print("Done!")


    ## --------- Intermediate values
    for i in range(0,99,10):
        print(i)
        fig, axs = plt.subplots(2,1, figsize=(8,8), gridspec_kw={'height_ratios': [1, 5]})
        axs[0].plot(x_np, u2d_final[:,i+100])
        axs[0].set_ylim([-.1,2])
        axs[0].grid()
        axs[0].set_title(F"Time {t_np[i]:0.2f}")
        axs[1].imshow(np.flipud(np.transpose(u2d_final)), extent=(min(x_domain), max(x_domain),
                               min(t_domain), max(t_domain_test)), cmap="inferno")

        axs[1].set_aspect(5)
        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Time")
        axs[1].set_title("Trained model")
        axs[1].plot([0,10],[t_np[i],t_np[i]], c="red")
        plt.show()
    print("Done!")
##

