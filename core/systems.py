import torch
import torch.nn as nn
import math

class InvertedPendulum(nn.Module):
    def __init__(self, dt=0.05, m=0.15, l=0.5, mu=0.05, g=9.81):
        super().__init__()
        self.dt = dt
        self.m = m
        self.l = l
        self.mu = mu
        self.g = g
        self.x_dim = 2
        self.u_dim = 1
        self.x_bounds = torch.tensor([[-math.pi, math.pi], [-8.0, 8.0]], dtype=torch.float32)
        self.u_bounds = torch.tensor([[-2.0, 2.0]], dtype=torch.float32)

    def forward(self, x, u):
        theta = x[:, 0:1]
        dot_theta = x[:, 1:2]
        
        # BỌC MỌI HẰNG SỐ VÀO TENSOR FLOAT32
        dev = x.device
        c_g_l = torch.tensor(self.g / self.l, dtype=torch.float32, device=dev)
        c_mu = torch.tensor(self.mu / (self.m * self.l * self.l), dtype=torch.float32, device=dev)
        c_u = torch.tensor(1.0 / (self.m * self.l * self.l), dtype=torch.float32, device=dev)
        t_dt = torch.tensor(self.dt, dtype=torch.float32, device=dev)
        
        ddot_theta = c_g_l * torch.sin(theta) - c_mu * dot_theta + c_u * u
        dot_theta_next = dot_theta + ddot_theta * t_dt
        theta_next = theta + dot_theta_next * t_dt 
        
        return torch.cat([theta_next, dot_theta_next], dim=1)