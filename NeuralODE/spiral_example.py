import torch
from torchdiffeq import odeint

device = "cuda" if torch.cuda.is_available() else "cpu"

class SpiralDynamics(torch.nn.Module):
    def forward(self, t, y):
        # A simple linear oscillator creating a spiral
        true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]], device=y.device)
        return torch.mm(y, true_A)

def get_spiral_data(n_points=1000, t_max=25.0):
    """
    Generate a 2D spiral trajectory.
    Returns:
    t: [n_points]
    true_y: [n_points, 1, 2]
    """
    true_y0 = torch.tensor([[2., 0.]], device=device)
    t = torch.linspace(0., t_max, n_points, device=device)
    with torch.no_grad():
        true_y = odeint(SpiralDynamics(), true_y0, t, method='dopri5')
    return t, true_y
