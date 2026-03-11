import torch
import torch.nn as nn
import math

class InvertedPendulum(nn.Module):
    def __init__(self, dt=0.05, m=0.15, l=0.5, mu=0.05, g=9.81):
        """
        Khởi tạo các tham số vật lý của hệ thống.
        """
        super().__init__()
        self.dt = dt
        self.m = m
        self.l = l
        self.mu = mu
        self.g = g
        
        # Không gian trạng thái và điều khiển
        self.x_dim = 2
        self.u_dim = 1
        
        # Giới hạn vật lý (Dùng cho Verifier sau này)
        # Ví dụ: theta thuộc [-pi, pi], dot_theta thuộc [-8, 8], u thuộc [-2, 2]
        self.x_bounds = torch.tensor([[-math.pi, math.pi], [-8.0, 8.0]])
        self.u_bounds = torch.tensor([[-2.0, 2.0]])

    def forward(self, x, u):
        """
        Tính toán trạng thái tiếp theo x_{t+1} = f(x_t, u_t)
        Input:
            x: Tensor kích thước (batch_size, 2)
            u: Tensor kích thước (batch_size, 1)
        Output:
            x_next: Tensor kích thước (batch_size, 2)
        """
        # Tách trạng thái
        theta = x[:, 0:1]
        dot_theta = x[:, 1:2]
        
        # Phương trình gia tốc góc
        ddot_theta = (self.g / self.l) * torch.sin(theta) \
                     - (self.mu / (self.m * self.l**2)) * dot_theta \
                     + (1.0 / (self.m * self.l**2)) * u
                     
        # Rời rạc hóa Euler
        dot_theta_next = dot_theta + ddot_theta * self.dt
        theta_next = theta + dot_theta_next * self.dt # Cập nhật vị trí dựa trên vận tốc mới
        
        # Chuẩn hóa góc theta về đoạn [-pi, pi] (quan trọng để tránh tràn số)
        theta_next = (theta_next + math.pi) % (2 * math.pi) - math.pi
        
        return torch.cat([theta_next, dot_theta_next], dim=1)