import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg as la


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
        x_dot = self.continuous_dynamics(x, u)
        return x + x_dot * self.dt
        
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


class CartpoleDynamics(BaseDynamics):
    """
    Cartpole dynamics around upright equilibrium.
    State: [x, x_dot, theta, theta_dot], action: force.
    """

    def __init__(
        self,
        gravity: float = 9.8,
        masscart: float = 1.0,
        masspole: float = 0.1,
        length: float = 1.0,
        dt: float = 0.05,
        damping: float = 0.0,
        max_force: float = 30.0,
        position_integration: str = "midpoint",
    ):
        super().__init__(nx=4, nu=1, dt=dt)
        self.register_buffer("gravity", torch.tensor(gravity))
        self.register_buffer("masscart", torch.tensor(masscart))
        self.register_buffer("masspole", torch.tensor(masspole))
        self.register_buffer("length", torch.tensor(length))
        self.register_buffer("damping", torch.tensor(damping))
        self.register_buffer("max_force", torch.tensor(max_force))
        integration_mode = str(position_integration).lower()
        if integration_mode not in {"euler", "midpoint", "semi_implicit"}:
            raise ValueError("position_integration must be one of: euler, midpoint, semi_implicit")
        self.position_integration = integration_mode

    def continuous_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        cart_pos = x[:, 0:1]
        cart_vel = x[:, 1:2]
        theta = x[:, 2:3]
        theta_dot = x[:, 3:4]

        action = u[:, 0:1]
        max_force = self.max_force
        force = action
        force = (-max_force) + F.relu(force - (-max_force))
        force = max_force - F.relu(max_force - force)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        total_mass = self.masscart + self.masspole
        temp = self.masscart + self.masspole * sin_theta * sin_theta

        theta_ddot = (
            -force * cos_theta
            - self.masspole * self.length * theta_dot * theta_dot * cos_theta * sin_theta
            + total_mass * self.gravity * sin_theta
            - self.damping * theta_dot
        ) / (self.length * temp)

        x_ddot = (
            force
            + self.masspole * sin_theta * (self.length * theta_dot * theta_dot - self.gravity * cos_theta)
        ) / temp

        return torch.cat([cart_vel, x_ddot, theta_dot, theta_ddot], dim=1)

    def step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x_dot = self.continuous_dynamics(x, u)
        if self.position_integration == "euler":
            return x + self.dt * x_dot

        cart_pos = x[:, 0:1]
        cart_vel = x[:, 1:2]
        theta = x[:, 2:3]
        theta_dot = x[:, 3:4]

        x_ddot = x_dot[:, 1:2]
        theta_ddot = x_dot[:, 3:4]

        cart_vel_next = cart_vel + self.dt * x_ddot
        theta_dot_next = theta_dot + self.dt * theta_ddot

        if self.position_integration == "semi_implicit":
            cart_pos_next = cart_pos + self.dt * cart_vel_next
            theta_next = theta + self.dt * theta_dot_next
        else:
            cart_pos_next = cart_pos + 0.5 * self.dt * (cart_vel + cart_vel_next)
            theta_next = theta + 0.5 * self.dt * (theta_dot + theta_dot_next)

        return torch.cat([cart_pos_next, cart_vel_next, theta_next, theta_dot_next], dim=1)

    def get_lqr_baseline(self):
        device = self.masscart.device if hasattr(self, 'masscart') else self.m.device
        x0 = torch.zeros((1, self.nx), dtype=torch.float32, device=device, requires_grad=True)
        u0 = torch.zeros((1, self.nu), dtype=torch.float32, device=device, requires_grad=True)

        f = self.continuous_dynamics(x0, u0)

        A_rows, B_rows = [], []
        for i in range(self.nx):
            grad_x, grad_u = torch.autograd.grad(f[0, i], [x0, u0], retain_graph=True)
            A_rows.append(grad_x[0].detach().cpu().numpy())
            B_rows.append(grad_u[0].detach().cpu().numpy())

        A_c = np.stack(A_rows, axis=0)
        B_c = np.stack(B_rows, axis=0)

        # ---------------------------------------------------------
        # CHUYỂN ĐỔI SANG KHÔNG GIAN RỜI RẠC (EULER DISCRETIZATION)
        # x_{k+1} = x_k + dt * (A_c * x_k + B_c * u_k)
        #         = (I + A_c * dt) * x_k + (B_c * dt) * u_k
        # ---------------------------------------------------------
        I = np.eye(self.nx)
        A_d = I + A_c * self.dt
        B_d = B_c * self.dt

        # Trọng số Q, R (CartPole ví dụ)
        Q = np.diag([5.0, 1.0, 20.0, 2.0])
        R = np.array([[0.1]])

        # Giải phương trình Riccati Rời rạc (DARE)
        S = la.solve_discrete_are(A_d, B_d, Q, R)
        
        # Ma trận Gain cho hệ rời rạc: K = (R + B_d^T S B_d)^-1 B_d^T S A_d
        # scipy linalg có thể tính tự động, nhưng tốt nhất nên tính tay hoặc dùng hàm K của scipy.
        # Lưu ý: la.solve_discrete_are chỉ trả về S (P trong tài liệu chuẩn).
        temp = np.linalg.inv(R + B_d.T @ S @ B_d)
        K = temp @ (B_d.T @ S @ A_d)

        return torch.tensor(K, dtype=torch.float32), torch.tensor(S, dtype=torch.float32)