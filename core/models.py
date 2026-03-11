import torch
import torch.nn as nn
import torch.nn.functional as F

class Controller(nn.Module):
    def __init__(self, x_dim=2, u_dim=1, u_max=2.0, hidden_layers=[32, 32]):
        super().__init__()
        self.u_max = u_max
        
        # Xây dựng cấu trúc MLP linh hoạt
        layers = []
        input_dim = x_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        
        # Lớp đầu ra sử dụng Tanh
        layers.append(nn.Linear(input_dim, u_dim))
        layers.append(nn.Tanh())
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Input: x (batch_size, x_dim)
        Output: u (batch_size, u_dim) trong giới hạn [-u_max, u_max]
        """
        # Ánh xạ đầu ra vào khoảng vật lý cho phép
        return self.u_max * self.net(x)


class LyapunovNetwork(nn.Module):
    def __init__(self, x_dim=2, hidden_layers=[64, 64], eps=0.01):
        super().__init__()
        self.eps = eps
        
        # Mạng nơ-ron phi tuyến cho V
        layers = []
        input_dim = x_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        
        layers.append(nn.Linear(input_dim, 1)) # Đầu ra vô hướng (năng lượng)
        self.net = nn.Sequential(*layers)
        
        # Thành phần bậc hai (Quadratic) x^T P x
        # Khởi tạo ma trận L (để P = L^T L luôn xác định dương)
        self.L = nn.Parameter(torch.eye(x_dim))

    def forward(self, x):
        """
        Tính giá trị năng lượng V(x)
        """
        # 1. Giá trị gốc của mạng
        v_raw = self.net(x)
        
        # 2. Giá trị tại điểm cân bằng x=0
        zeros = torch.zeros_like(x)
        v_zero = self.net(zeros)
        
        # 3. Tính thành phần mạng (đã neo V(0) = 0 và >= 0)
        v_nn = F.relu(v_raw - v_zero)
        
        # 4. Tính thành phần bậc hai x^T (L^T L) x
        P = torch.matmul(self.L.t(), self.L)
        # x: (batch, x_dim), P: (x_dim, x_dim) -> x^T P x: (batch, 1)
        v_quad = torch.sum(x * torch.matmul(x, P), dim=1, keepdim=True)
        
        # 5. Thành phần chuẩn hóa nhỏ epsilon * ||x||^2 để tránh V quá phẳng
        v_eps = self.eps * torch.sum(x**2, dim=1, keepdim=True)
        
        return v_nn + v_quad + v_eps