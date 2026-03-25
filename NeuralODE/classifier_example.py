import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_concentric_circles(n_samples=1000, noise=0.05):
    """
    Generate a 2D dataset of concentric circles.
    Returns:
    X: [n_samples, 2]
    y: [n_samples, 1]
    """
    from sklearn.datasets import make_circles
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    return torch.tensor(X, dtype=torch.float32, device=device), \
           torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)
