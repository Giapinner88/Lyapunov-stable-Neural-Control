import torch
import torch.nn as nn

# ==========================================
# 1. NEURAL CONTROLLER
# ==========================================
class NeuralController(nn.Module):
    def __init__(self, nx: int, nu: int, hidden_sizes: list = [64, 64], u_bound: float = 6.0):
        super().__init__()
        self.u_bound = u_bound
        
        # [BẢN VÁ]: Khởi tạo giới hạn vật lý để chuẩn hóa (Góc 3.14, Vận tốc 8.0)
        self.register_buffer("state_limits", torch.tensor([3.1415, 8.0]))
        
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
        return self.net(x_norm) * self.u_bound


# ==========================================
# 2. NEURAL LYAPUNOV FUNCTION
# ==========================================
class NeuralLyapunov(nn.Module):
    def __init__(self, nx: int, hidden_sizes: list = [64, 64], eps: float = 0.01):
        super().__init__()
        
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
        self.register_buffer("origin", torch.zeros(1, nx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Tính phi_0
        zero_norm = self.origin / self.state_limits
        with torch.no_grad():
            phi_0 = self.phi_V(zero_norm)

        # 2. Tính ||phi_V(x) - phi_V(0)||_1 trên x đã chuẩn hóa
        x_norm = x / self.state_limits
        phi_x = self.phi_V(x_norm)          
        term1 = torch.sum(torch.abs(phi_x - phi_0), dim=1, keepdim=True)
        
        # 3. Tính x^T * P * x 
        # (LƯU Ý: Tuyệt đối dùng x GỐC ở đây để bảo toàn kích thước vật lý của mỏ neo LQR)
        P = self.eps * self.eye + torch.matmul(self.R.T, self.R)
        term2 = torch.einsum('bi,ij,bj->b', x, P, x).unsqueeze(1)
        
        # 4. BỨC TỎNG GIỚI HẠN V CÓ KÍCH THƯỚC HỢP LÝ:
        # Thêm max thấp nhất là 0.01 để đảm bảo V > 0 luôn
        # và có kích thước thích hợp để gradient PGD có thể hoạt động hiệu quả
        V = term1 + term2
        V = torch.relu(V) + 0.01
        return V