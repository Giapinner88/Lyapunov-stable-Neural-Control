import torch
import torch.nn as nn
import numpy as np
import scipy.linalg as la

# ==========================================
# 1. CORE INTEGRATOR
# ==========================================
def rk4_step(continuous_dynamics_fn, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    k1 = continuous_dynamics_fn(x, u)
    k2 = continuous_dynamics_fn(x + 0.5 * dt * k1, u)
    k3 = continuous_dynamics_fn(x + 0.5 * dt * k2, u)
    k4 = continuous_dynamics_fn(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ==========================================
# 2. BASE DYNAMICS (Đã tích hợp nn.Module)
# ==========================================
class BaseDynamics(nn.Module):
    def __init__(self, nx: int, nu: int, dt: float):
        super().__init__()
        self.nx = nx
        self.nu = nu
        self.dt = dt

    def continuous_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Phải được ghi đè bởi hệ vật lý cụ thể")

    def step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return rk4_step(self.continuous_dynamics, x, u, self.dt)
        
    def get_lqr_baseline(self):
        raise NotImplementedError()


# ==========================================
# 3. INVERTED PENDULUM DYNAMICS
# ==========================================
class PendulumDynamics(BaseDynamics):
    def __init__(self, m=0.15, l=0.5, b=0.1, g=9.81, dt=0.02):
        super().__init__(nx=2, nu=1, dt=dt)
        
        # Đăng ký các hằng số vật lý như là parameters/buffers 
        # để chúng tự động chuyển sang GPU khi gọi .to(device)
        self.register_buffer('m', torch.tensor(m))
        self.register_buffer('l', torch.tensor(l))
        self.register_buffer('b', torch.tensor(b))
        self.register_buffer('g', torch.tensor(g))

    def continuous_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        theta = x[:, 0:1]
        theta_dot = x[:, 1:2]

        I = self.m * (self.l ** 2)
        theta_ddot = (self.m * self.g * self.l * torch.sin(theta) - self.b * theta_dot + u) / I
        
        return torch.cat([theta_dot, theta_ddot], dim=1)

    def get_lqr_baseline(self):
        """
        Tuyến tính hóa hệ thống quanh x=[0,0] và u=0.
        Tính Jacobian (Đạo hàm riêng) để lập ma trận A và B.
        """
        m = self.m.item()
        l = self.l.item()
        b = self.b.item()
        g = self.g.item()
        I = m * (l ** 2)

        # Ma trận A = df/dx
        A = np.array([
            [0.0, 1.0],
            [(m * g * l) / I, -b / I]
        ])

        # Ma trận B = df/du
        B = np.array([
            [0.0],
            [1.0 / I]
        ])

        # Ma trận trọng số Q (Phạt trạng thái) và R (Phạt lực u)
        # Hệ con lắc ưu tiên giữ góc đứng thẳng (index 0) nên đặt trọng số Q lớn hơn
        Q = np.diag([10.0, 1.0])
        R = np.array([[1.0]])

        # Giải phương trình Riccati liên tục (CARE)
        S = la.solve_continuous_are(A, B, Q, R)
        
        # Tính ma trận Gain K = R^-1 * B^T * S
        K = np.linalg.inv(R) @ (B.T @ S)

        # Trả về dạng PyTorch Tensor
        return torch.tensor(K, dtype=torch.float32), torch.tensor(S, dtype=torch.float32)