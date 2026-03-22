import torch
import abc
import math

# ==========================================
# 1. CORE INTEGRATOR (Bản chất của việc rời rạc hóa)
# ==========================================
def rk4_step(continuous_dynamics_fn, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Thuật toán Runge-Kutta bậc 4 (RK4) thuần PyTorch.
    Tại sao không dùng Euler? Euler (x_next = x + f(x)*dt) gây tích lũy sai số tuyến tính, 
    khiến mạng Lyapunov học trên một hệ động lực học bị sai lệch. 
    RK4 lấy trung bình trọng số của 4 điểm gradient trong khoảng dt, giảm sai số xuống bậc O(dt^4).
    Đặc biệt: Vì chỉ dùng toán tử +, *, toàn bộ đồ thị tính toán (computation graph) 
    được bảo toàn để PGD có thể backpropagate qua hàm này.
    """
    k1 = continuous_dynamics_fn(x, u)
    k2 = continuous_dynamics_fn(x + 0.5 * dt * k1, u)
    k3 = continuous_dynamics_fn(x + 0.5 * dt * k2, u)
    k4 = continuous_dynamics_fn(x + dt * k3, u)
    
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next


# ==========================================
# 2. BASE DYNAMICS (Khuôn mẫu đa hình)
# ==========================================
class BaseDynamics(abc.ABC):
    def __init__(self, nx: int, nu: int, dt: float):
        self.nx = nx  # Số chiều không gian trạng thái
        self.nu = nu  # Số chiều tín hiệu điều khiển
        self.dt = dt  # Bước thời gian rời rạc

    @abc.abstractmethod
    def continuous_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Định nghĩa phương trình vi phân: dx/dt = f(x, u)
        Phải được ghi đè bởi hệ vật lý cụ thể.
        """
        pass

    def step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Hàm giao tiếp chính với vòng lặp CEGIS. 
        Biến hệ liên tục thành hệ rời rạc x_{t+1} = f_discrete(x_t, u_t)
        """
        return rk4_step(self.continuous_dynamics, x, u, self.dt)
        
    @abc.abstractmethod
    def get_lqr_baseline(self):
        """
        Trả về ma trận K (Gain) và S (Solution of Riccati) của bộ điều khiển LQR.
        Dùng để khởi tạo V_ref(x) = x^T S x và policy baseline trong bài báo.
        """
        pass


# ==========================================
# 3. INVERTED PENDULUM DYNAMICS
# ==========================================
class PendulumDynamics(BaseDynamics):
    def __init__(self, m=0.15, l=0.5, b=0.1, g=9.81, dt=0.02):
        super().__init__(nx=2, nu=1, dt=dt)
        self.m = m
        self.l = l
        self.b = b
        self.g = g

    def continuous_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Trạng thái x = [theta, theta_dot]
        theta: Góc lệch so với vị trí thẳng đứng (cân bằng trên cùng).
        Phương trình: 
        dx1/dt = x2
        dx2/dt = (m*g*l*sin(x1) - b*x2 + u) / (m * l^2)
        """
        # Tách các biến trạng thái từ batch. Shape của x là (Batch, 2)
        theta = x[:, 0:1]
        theta_dot = x[:, 1:2]

        # Tránh việc dùng torch.cat nếu không cần thiết để tối ưu tốc độ, 
        # nhưng ở đây torch.cat là cách tường minh nhất để giữ batch_size
        I = self.m * (self.l ** 2)
        
        theta_ddot = (self.m * self.g * self.l * torch.sin(theta) - self.b * theta_dot + u) / I
        
        # Đạo hàm của trạng thái: dx/dt
        dx = torch.cat([theta_dot, theta_ddot], dim=1)
        return dx

    def get_lqr_baseline(self):
        # Tạm thời trả về None, chúng ta sẽ dùng thư viện control hoặc scipy 
        # để tính ma trận LQR tuyến tính hóa ở bước sau.
        pass