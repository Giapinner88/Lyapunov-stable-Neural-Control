import torch
import torch.nn as nn
import torch.nn.functional as F

class Controller(nn.Module):
    def __init__(self, x_dim=2, u_dim=1, u_max=2.0, hidden_layers=[32, 32]):
        super().__init__()
        self.u_max = u_max
        layers = []
        input_dim = x_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, u_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # KHỬ FLOAT64
        u_max_t = torch.tensor(self.u_max, dtype=torch.float32, device=x.device)
        return u_max_t * self.net(x)

class LyapunovNetwork(nn.Module):
    def __init__(self, x_dim=2, hidden_layers=[64, 64], eps=0.01):
        super().__init__()
        self.eps = eps
        layers = []
        input_dim = x_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)
        self.L = nn.Parameter(torch.eye(x_dim))

    def forward(self, x):
        v_raw = self.net(x)
        zeros = torch.zeros_like(x)
        v_zero = self.net(zeros)
        v_nn = F.relu(v_raw - v_zero)
        
        P = torch.matmul(self.L.t(), self.L)
        v_quad = torch.sum(x * torch.matmul(x, P), dim=1, keepdim=True)
        
        # KHỬ FLOAT64 và INT
        eps_t = torch.tensor(self.eps, dtype=torch.float32, device=x.device)
        v_eps = eps_t * torch.sum(x * x, dim=1, keepdim=True) # x * x thay cho x**2
        
        return v_nn + v_quad + v_eps