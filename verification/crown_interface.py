import torch
import torch.nn as nn

class VerificationWrapper(nn.Module):
    def __init__(self, system, controller, lyapunov, kappa=0.1):
        """
        Gộp cả hệ vật lý và mạng Nơ-ron thành một đồ thị tính toán duy nhất.
        """
        super().__init__()
        self.system = system
        self.controller = controller
        self.lyapunov = lyapunov
        self.kappa = kappa

    def forward(self, x):
        """
        Đầu vào: x
        Đầu ra: [V(x), F(x)]
        """
        # 1. Tính toán điều khiển và trạng thái tiếp theo
        u = self.controller(x)
        x_next = self.system(x, u)
        
        # 2. Tính năng lượng
        v_curr = self.lyapunov(x)
        v_next = self.lyapunov(x_next)
        
        # 3. Tính độ vi phạm
        F_x = v_next - (1 - self.kappa) * v_curr
        
        # Trả về 2 kênh đầu ra để CROWN áp đặt điều kiện logic
        return torch.cat([v_curr, F_x], dim=1)

def generate_vnnlib_spec(filepath, x_bounds, rho):
    """
    Sinh file đặc tả .vnnlib cho bộ giải CROWN.
    Điều kiện phản ví dụ: x thuộc x_bounds AND V(x) <= rho AND F(x) > 0.
    """
    x_dim = len(x_bounds)
    with open(filepath, 'w') as f:
        # Khai báo biến
        for i in range(x_dim):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("(declare-const Y_0 Real)\n") # Y_0 là V(x)
        f.write("(declare-const Y_1 Real)\n\n") # Y_1 là F(x)

        # Định nghĩa Input Bounds (x in B)
        for i in range(x_dim):
            f.write(f"(assert (<= X_{i} {x_bounds[i][1]}))\n")
            f.write(f"(assert (>= X_{i} {x_bounds[i][0]}))\n")

        # Định nghĩa Output Specifications (Phản ví dụ)
        f.write(f"\n; Y_0 = V(x) <= rho\n")
        f.write(f"(assert (<= Y_0 {rho}))\n")
        
        f.write(f"\n; Y_1 = F(x) > 0 (Vi phạm)\n")
        f.write(f"(assert (> Y_1 0.0))\n")