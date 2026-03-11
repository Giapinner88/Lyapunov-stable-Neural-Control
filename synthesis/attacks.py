import torch
import torch.nn.functional as F

class PGDAttacker:
    def __init__(self, system, controller, lyapunov, config):
        """
        Khởi tạo Kẻ tấn công (Attacker) dựa trên phương pháp Hạ Gradient Chiếu.
        """
        self.system = system
        self.controller = controller
        self.lyapunov = lyapunov
        
        # Đã sửa lỗi truy cập Dictionary thay vì Object Attributes
        self.bounds = config.get('state_bounds')      # Tensor shape [x_dim, 2]
        self.steps = config.get('pgd_steps', 15)
        self.alpha = config.get('pgd_step_size', 0.05)
        self.kappa = config.get('kappa', 0.1)
        self.penalty_weight = config.get('lambda_margin', 100.0)

    def sample_random(self, batch_size):
        """
        Sinh ngẫu nhiên các điểm trạng thái tuân theo phân phối đều (Uniform)
        nằm gọn trong ranh giới vật lý B.
        """
        x_dim = self.bounds.shape[0]
        # rand_tensor có giá trị [0, 1)
        rand_tensor = torch.rand((batch_size, x_dim))
        
        min_bounds = self.bounds[:, 0]
        max_bounds = self.bounds[:, 1]
        
        # Phép nội suy tuyến tính: x = min + rand * (max - min)
        x_sampled = min_bounds + rand_tensor * (max_bounds - min_bounds)
        return x_sampled

    def find_counter_examples(self, batch_size, rho):
        """
        Thực thi thuật toán PGD để săn lùng các phản ví dụ (Adversarial Examples).
        Mục tiêu: Tối đa hóa hàm vi phạm F(x), nhưng bị phạt nặng nếu x văng ra khỏi rho.
        """
        # Đưa các mạng về chế độ đánh giá để gradient không làm rối loạn Dropout/BatchNorm
        self.system.eval()
        self.controller.eval()
        self.lyapunov.eval()

        # 1. Khởi tạo điểm neo ngẫu nhiên
        x = self.sample_random(batch_size)

        min_bounds = self.bounds[:, 0]
        max_bounds = self.bounds[:, 1]

        # 2. Vòng lặp Leo đồi Gradient (Gradient Ascent Loop)
        for k in range(self.steps):
            x.requires_grad_(True)

            # Tính toán động lực học
            u = self.controller(x)
            x_next = self.system(x, u)

            v_curr = self.lyapunov(x)
            v_next = self.lyapunov(x_next)

            # Hàm vi phạm (Lyapunov Derivative)
            F_x = v_next - (1 - self.kappa) * v_curr

            # Hàm mục tiêu tấn công (Attack Objective)
            penalty = self.penalty_weight * F.relu(v_curr - rho)
            # Dùng torch.mean để chuẩn hóa loss trên toàn bộ batch
            attack_loss = torch.mean(F_x - penalty)

            # Tính đạo hàm của attack_loss theo tensor đầu vào x
            grad_x = torch.autograd.grad(attack_loss, x)[0]
            
            # Cập nhật trạng thái: Cộng gradient để TỐI ĐA HÓA (Maximize) lỗi
            with torch.no_grad():
                # Kỹ thuật Fast Gradient Sign: Dùng hàm sign() để nhảy các bước bằng nhau, tránh gradient bị suy giảm
                x_new = x + self.alpha * torch.sign(grad_x)
                
                # Phép chiếu (Projection): Ép x_new nằm gọn trong hộp giới hạn B
                x = torch.max(torch.min(x_new, max_bounds), min_bounds)
        
        # Đưa mạng về lại chế độ huấn luyện cho Trainer
        self.system.train()
        self.controller.train()
        self.lyapunov.train()

        # Cắt đứt đồ thị tính toán để tránh rò rỉ bộ nhớ (Memory Leak)
        return x.detach()