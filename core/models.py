import torch
import torch.nn as nn

# ==========================================
# 1. NEURAL CONTROLLER
# ==========================================
class NeuralController(nn.Module):
    def __init__(self, nx: int, nu: int, hidden_sizes: list = [64, 64], u_bound: float = 1.0):
        super().__init__()
        self.u_bound = u_bound
        
        layers = []
        in_size = nx
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.Tanh())  # Tanh giúp gradient mượt hơn ReLU trong hệ động lực học
            in_size = h
            
        layers.append(nn.Linear(in_size, nu))
        # Giới hạn lực điều khiển đầu ra trong khoảng [-u_bound, u_bound]
        layers.append(nn.Tanh()) 
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.u_bound


# ==========================================
# 2. NEURAL LYAPUNOV FUNCTION
# ==========================================
class NeuralLyapunov(nn.Module):
    def __init__(self, nx: int, hidden_sizes: list = [64, 64], eps: float = 0.01):
        super().__init__()
        
        layers = []
        in_size = nx
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.Tanh())
            in_size = h
        
        # Đầu ra của phi_V không cần bị giới hạn số chiều, nhưng thường để bằng nx
        layers.append(nn.Linear(in_size, nx)) 
        self.phi_V = nn.Sequential(*layers)
        
        # Ma trận R (có thể học được) và hằng số eps cho cụm Positive Definite
        self.R = nn.Parameter(torch.randn(nx, nx) * 0.1)
        self.eps = eps
        
        # Lưu trước ma trận đơn vị để không phải tạo lại mỗi lần forward
        self.register_buffer("eye", torch.eye(nx))
        self.register_buffer("origin", torch.zeros(1, nx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Công thức: V(x) = ||phi_V(x) - phi_V(0)||_1 + x^T * (eps * I + R^T * R) * x
        """
        # 1. Tính toán phần phi_V(0) (yêu cầu không có gradient backprop qua số 0)
        zero = self.origin.to(device=x.device, dtype=x.dtype)
        with torch.no_grad():
            phi_0 = self.phi_V(zero)   # shape (1, nx)

        # 2. Tính ||phi_V(x) - phi_V(0)||_1 (Tổng giá trị tuyệt đối dọc theo các chiều state)
        phi_x = self.phi_V(x)          # shape (B, nx)
        term1 = torch.sum(torch.abs(phi_x - phi_0), dim=1, keepdim=True)
        
        # 3. Tính x^T * P * x, trong đó P = eps*I + R^T * R
        # Phép tính này phải thực hiện theo batch ma trận.
        P = self.eps * self.eye + torch.matmul(self.R.T, self.R)
        
        # torch.einsum('bi,ij,bj->b'...) tính x^T P x cho toàn bộ batch rất nhanh và gọn
        term2 = torch.einsum('bi,ij,bj->b', x, P, x).unsqueeze(1)
        
        return term1 + term2