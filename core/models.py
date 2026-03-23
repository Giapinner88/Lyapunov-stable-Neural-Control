import torch
import torch.nn as nn

# ==========================================
# 1. NEURAL CONTROLLER
# ==========================================
class NeuralController(nn.Module):
    def __init__(self, nx: int, nu: int, hidden_sizes: list = [64, 64], u_bound: float = 6.0):
        super().__init__()
        self.u_bound = u_bound
        self.nx = nx
        
        # [BẢN VÁ]: Khởi tạo giới hạn vật lý để chuẩn hóa (Góc 3.14, Vận tốc 8.0)
        self.register_buffer("state_limits", torch.tensor([3.1415, 8.0]))
        self.register_buffer("origin", torch.zeros(nx), persistent=False)
        
        layers = []
        in_size = nx
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.Tanh())  
            in_size = h
            
        layers.append(nn.Linear(in_size, nu))
        layers.append(nn.Tanh()) 
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ép x vật lý về x_norm nằm trong khoảng [-1, 1] để Tanh không bị bão hòa
        x_norm = x / self.state_limits
        u_raw = self.net(x_norm)

        # Cưỡng bức điều kiện cân bằng chính xác: u(0) = 0.
        zero_norm = self.origin.unsqueeze(0) / self.state_limits
        u_origin = self.net(zero_norm).squeeze(0)

        return (u_raw - u_origin) * self.u_bound


# ==========================================
# 2. NEURAL LYAPUNOV FUNCTION
# ==========================================
class NeuralLyapunov(nn.Module):
    def __init__(self, nx: int, hidden_sizes: list = [64, 64], eps: float = 0.01):
        super().__init__()
        if eps <= 0.0:
            raise ValueError("eps phải > 0 để đảm bảo V(x) dương xác định nghiêm ngặt")
        
        # [BẢN VÁ]: Giới hạn chuẩn hóa
        self.register_buffer("state_limits", torch.tensor([3.1415, 8.0]))
        
        layers = []
        in_size = nx
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.Tanh())
            in_size = h
        
        layers.append(nn.Linear(in_size, nx)) 
        self.phi_V = nn.Sequential(*layers)
        
        # FIX: Khởi tạo R lớn hơn (từ 0.1 → 0.5) để P = R^T*R kích thước hợp lý
        self.R = nn.Parameter(torch.randn(nx, nx) * 0.5)
        self.eps = eps
        
        self.register_buffer("eye", torch.eye(nx))
        self.register_buffer("origin", torch.zeros(nx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Tính phi_0
        zero_norm = self.origin.unsqueeze(0) / self.state_limits
        with torch.no_grad():
            phi_0 = self.phi_V(zero_norm).squeeze(0)

        # 2. Tính ||phi_V(x) - phi_V(0)||_1 trên x đã chuẩn hóa
        x_norm = x / self.state_limits
        phi_x = self.phi_V(x_norm)
        term1 = torch.sum(torch.abs(phi_x - phi_0), dim=1, keepdim=True)
        
        # 3. Tính x^T * P * x 
        # (LƯU Ý: Tuyệt đối dùng x GỐC ở đây để bảo toàn kích thước vật lý của mỏ neo LQR)
        P = self.eps * self.eye + torch.matmul(self.R.T, self.R)
        Px = torch.matmul(x, P)
        term2 = torch.sum(Px * x, dim=1, keepdim=True)
        
        # 4. Hàm Lyapunov không dùng offset cứng:
        # term1 >= 0 và term2 > 0 với mọi x != 0 vì P = eps*I + R^T R là SPD (eps > 0).
        # Do đó V(0) = 0 và V(x) > 0 với mọi x != 0.
        V = term1 + term2
        return V

    def load_state_dict(self, state_dict, strict: bool = True):
        # Backward compatibility: old checkpoints stored origin as shape [1, nx].
        origin_key = "origin"
        if origin_key in state_dict:
            origin_tensor = state_dict[origin_key]
            if origin_tensor.ndim == 2 and origin_tensor.shape[0] == 1:
                state_dict = state_dict.copy()
                state_dict[origin_key] = origin_tensor.squeeze(0)
        return super().load_state_dict(state_dict, strict=strict)