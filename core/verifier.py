import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

class SystemViolationGraph(nn.Module):
    def __init__(self, controller, lyapunov, dynamics, rho=0.01):
        """
        Bản chất: Đóng gói toàn bộ hệ động lực vào một hàm forward duy nhất.
        Đầu vào là x_t, đầu ra là lượng vi phạm điều kiện Lyapunov.
        """
        super().__init__()
        self.controller = controller
        self.lyapunov = lyapunov
        self.dynamics = dynamics
        self.rho = rho # Hệ số ép tốc độ hội tụ tối thiểu

    def forward(self, x_t):
        # 1. Năng lượng hiện tại
        v_t = self.lyapunov(x_t)
        
        # 2. Hành động điều khiển (đã bị kẹp bởi Tanh trong model)
        u_t = self.controller(x_t)
        
        # 3. Trạng thái tiếp theo (RK4)
        # Lưu ý: Hàm rk4_step trong core/dynamics.py của bạn phải hoàn toàn 
        # là các phép toán tensor, không chứa rẽ nhánh logic.
        x_next = self.dynamics.rk4_step(x_t, u_t)
        
        # 4. Năng lượng tiếp theo
        v_next = self.lyapunov(x_next)
        
        # 5. Hàm vi phạm (Violation)
        # Theo lý thuyết: V(x_{t+1}) - V(x_t) <= -rho * V(x_t)
        # Chuyển vế: V(x_{t+1}) - V(x_t) + rho * V(x_t) <= 0
        # Nếu violation > 0, tức là hệ thống đang vi phạm tính ổn định.
        violation = v_next - v_t + self.rho * v_t
        return violation

    def verify_and_mine_ce(controller, lyapunov, dynamics, device, 
                       grid_size=(50, 50), 
                       state_limits=(3.14, 8.0)):
        """
        Chia không gian pha thành grid_size[0] x grid_size[1] hộp nhỏ.
        Dùng CROWN xác minh từng hộp. Trả về các điểm vi phạm để Learner học lại.
        """
        model = SystemViolationGraph(controller, lyapunov, dynamics).to(device)
        model.eval() # Bắt buộc phải ở chế độ eval() khi dùng CROWN

        # Khởi tạo auto_LiRPA BoundedModule
        dummy_input = torch.zeros(1, 2, device=device)
        bounded_model = BoundedModule(model, dummy_input, bound_opts={'relu': 'adaptive'})

        # Tạo lưới tọa độ (Tâm của các hộp con)
        theta_centers = torch.linspace(-state_limits[0], state_limits[0], grid_size[0])
        dot_theta_centers = torch.linspace(-state_limits[1], state_limits[1], grid_size[1])
        
        grid_theta, grid_dot = torch.meshgrid(theta_centers, dot_theta_centers, indexing='ij')
        x_centers = torch.stack([grid_theta.flatten(), grid_dot.flatten()], dim=1).to(device)
        
        # Bán kính của mỗi hộp con (eps)
        eps_theta = (state_limits[0] * 2) / (grid_size[0] - 1) / 2.0
        eps_dot = (state_limits[1] * 2) / (grid_size[1] - 1) / 2.0
        eps_tensor = torch.tensor([eps_theta, eps_dot], device=device)

        # Cấu trúc BoundedTensor: Biểu diễn một tập hợp các hộp bao quanh x_centers
        ptb = PerturbationLpNorm(norm=torch.inf, eps=eps_tensor)
        bounded_x = BoundedTensor(x_centers, ptb)

        # Chạy CROWN (Lan truyền ngược qua RK4 và 2 mạng nơ-ron)
        # Lấy ra ranh giới trên (ub)
        with torch.no_grad():
            lb, ub = bounded_model.compute_bounds(x=(bounded_x,), method='CROWN')

        # Lọc ra những hộp có khả năng vi phạm (ub > 0)
        # Tuy nhiên, ta chỉ quan tâm đến vi phạm nếu điểm đó nằm ngoài vùng LQR mỏ neo 
        # (vì vùng LQR quanh 0 đã được chứng minh giải tích là ổn định).
        
        v_values = lyapunov(x_centers).squeeze()
        
        violation_mask = (ub.squeeze() > 0) & (v_values > 0.05) # Bỏ qua vùng tâm
        
        counterexamples = x_centers[violation_mask]
        
        print(f"[Verifier] Quét {len(x_centers)} vùng. CROWN báo động {len(counterexamples)} vùng rủi ro.")
        
        return counterexamples