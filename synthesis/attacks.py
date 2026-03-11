# synthesis/attacks.py
import torch
import torch.nn as nn

class PGDAttacker:
    def __init__(self, system, controller, lyapunov, config):
        self.system = system
        self.controller = controller
        self.lyapunov = lyapunov
        self.bounds = config.state_bounds  # Tensor shape [x_dim, 2]
        self.steps = config.pgd_steps      # e.g., 10 to 50
        self.alpha = config.pgd_step_size  # e.g., 0.01
        self.kappa = config.kappa          # e.g., 0.1
        self.penalty_weight = 10.0         # lambda parameter
        
        # [BỔ SUNG CỐT LÕI]: Bộ đệm lưu trữ phản ví dụ hình thức từ CROWN
        self.formal_ce_buffer = []

    def sample_random(self, batch_size):
        """Khám phá không gian ngẫu nhiên (Exploration)."""
        min_bounds = self.bounds[:, 0]
        max_bounds = self.bounds[:, 1]
        x = torch.rand(batch_size, self.bounds.shape[0]) * (max_bounds - min_bounds) + min_bounds
        return x.detach()

    def find_counter_examples(self, batch_size, rho):
        """Thực thi PGD tìm kiếm lân cận tối ưu (Exploitation)."""
        x = self.sample_random(batch_size)
        min_bounds = self.bounds[:, 0]
        max_bounds = self.bounds[:, 1]

        for k in range(self.steps):
            x.requires_grad_(True)

            u = self.controller(x)
            x_next = self.system(x, u) # SỬA LỖI: Gọi trực tiếp __call__ của nn.Module thay vì .step()

            v_current = self.lyapunov(x).squeeze(-1)
            v_next = self.lyapunov(x_next).squeeze(-1)

            F_x = v_next - (1.0 - self.kappa) * v_current
            penalty = self.penalty_weight * torch.relu(v_current - rho)

            attack_loss = (F_x - penalty).mean()
            attack_loss.backward()

            with torch.no_grad():
                grad_x = x.grad
                x_new = x + self.alpha * grad_x.sign() # Gradient Ascent để cực đại hóa vi phạm
                x = torch.clamp(x_new, min=min_bounds, max=max_bounds)
                
            x = x.detach() # Giải phóng đồ thị tính toán

        return x

    def add_formal_counter_examples(self, x_formal):
        """Tiếp nhận điểm vi phạm từ CROWN."""
        if x_formal is not None:
            # Lưu trữ dưới dạng tensor không có gradient
            self.formal_ce_buffer.append(x_formal.detach().cpu())

    def get_training_batch(self, batch_size, rho):
        """
        Trộn 3 nguồn dữ liệu theo nguyên lý CEGIS:
        1. Điểm tấn công PGD (Lớn nhất)
        2. Mẫu ngẫu nhiên (Giữ phân phối cơ sở)
        3. Phản ví dụ hình thức (Điểm mù bắt buộc phải khắc phục)
        """
        x_adv = self.find_counter_examples(batch_size, rho)
        x_rand = self.sample_random(batch_size // 5)
        
        tensors = [x_adv, x_rand]
        
        if len(self.formal_ce_buffer) > 0:
            # Đưa toàn bộ phản ví dụ hình thức đã tìm được vào mỗi batch huấn luyện
            x_formal = torch.cat(self.formal_ce_buffer, dim=0).to(x_adv.device)
            tensors.append(x_formal)
            
        return torch.cat(tensors, dim=0)