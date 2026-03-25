import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class ODEFunc(nn.Module):
    def __init__(self, in_dim=2, num_hidden=50):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.ELU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ELU(),
            nn.Linear(num_hidden, in_dim)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, val=0)
                
    def forward(self, t, y):
        return self.net(y)


class NeuralODE(nn.Module):
    def __init__(self, in_dim=2, num_hidden=64, method='dopri5'):
        super(NeuralODE, self).__init__()
        self.func = ODEFunc(in_dim, num_hidden)
        self.method = method
        
    def forward(self, y0, t):
        out = odeint(self.func, y0, t, method=self.method)
        return out

class NeuralODEClassifier(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=16, num_classes=1, method='dopri5'):
        super(NeuralODEClassifier, self).__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.ode_block = NeuralODE(in_dim=hidden_dim, num_hidden=32, method=method)
        self.decoder = nn.Linear(hidden_dim, num_classes)
        self.t = torch.tensor([0.0, 1.0], device=device) # Integration time [0, 1]

    def forward(self, x):
        h0 = self.encoder(x)
        # Solve ODE
        hT = self.ode_block(h0, self.t)[-1]
        out = self.decoder(hT)
        return out
