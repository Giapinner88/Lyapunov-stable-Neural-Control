# synthesis/trainer.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging

class CEGISTrainer:
    def __init__(self, system, controller, lyapunov, attacker, config):
        """
        Khởi tạo khối Synthesis (Huấn luyện mạng Nơ-ron) trong vòng lặp CEGIS.
        """
        self.system = system
        self.controller = controller
        self.lyapunov = lyapunov
        self.attacker = attacker
        
        # Hỗ trợ linh hoạt: config có thể là dictionary hoặc object (dataclass/Namespace)
        if isinstance(config, dict):
            self.rho = config.get('rho', 0.1)
            self.kappa = config.get('kappa', 0.1)
            self.epochs = config.get('epochs', 1000)
            self.batch_size = config.get('batch_size', 5000)
            self.lambda_margin = config.get('lambda_margin', 100.0)
        else:
            self.rho = getattr(config, 'rho', 0.1)
            self.kappa = getattr(config, 'kappa', 0.1)
            self.epochs = getattr(config, 'epochs', 1000)
            self.batch_size = getattr(config, 'batch_size', 5000)
            self.lambda_margin = getattr(config, 'lambda_margin', 100.0)
        
        # Sử dụng Adam Optimizer tách biệt cho 2 mạng để tránh can thiệp chéo gradient
        self.opt_ctrl = optim.Adam(self.controller.parameters(), lr=1e-3)
        self.opt_lyap = optim.Adam(self.lyapunov.parameters(), lr=1e-3)

    def compute_loss(self, x):
        """
        Tính toán hàm suy hao nới lỏng (Relaxed Loss) theo công thức của Yang et al. (2024).
        """
        # 1. Động lực học tiến (Forward Dynamics)
        u = self.controller(x)
        x_next = self.system(x, u)
        
        v_curr = self.lyapunov(x)
        v_next = self.lyapunov(x_next)
        
        # 2. Sai số giảm Lyapunov (Lyapunov Derivative Violation)
        # Điều kiện lý tưởng: V(x_next) - (1 - kappa) * V(x) <= 0
        F_x = v_next - (1 - self.kappa) * v_curr
        
        # 3. Hàm suy hao nới lỏng (Relaxed Loss)
        # Phạt vi phạm F(x) > 0, nhưng nới lỏng (không phạt) nếu điểm đó nằm ngoài vùng ROA (V(x) > rho)
        margin = self.lambda_margin * F.relu(v_curr - self.rho)
        loss_lyap = torch.mean(F.relu(F_x - margin))
        
        # 4. Hàm suy hao hiệu suất (Performance Loss)
        # Khuyến khích năng lượng V(x) trơn tru và tối thiểu hóa nỗ lực điều khiển u
        loss_perf = 0.01 * torch.mean(v_curr) + 0.01 * torch.mean(u**2)
        
        total_loss = loss_lyap + loss_perf
        return total_loss, F_x, v_curr

    def train(self):
        """
        Thực thi quá trình tối ưu hóa trọng số. Hàm này sẽ được gọi lặp đi lặp lại 
        bởi bộ điều phối (Orchestrator) trong cấu trúc CEGIS.
        """
        for epoch in range(self.epochs):
            self.controller.train()
            self.lyapunov.train()
            
            # --- GIAI ĐOẠN 1: TÌM PHẢN VÍ DỤ (SYNTHESIS / ATTACK) ---
            # Giao phó toàn bộ logic tổ hợp tập dữ liệu cho PGDAttacker. 
            # Dữ liệu tại đây đã tự động hợp nhất: Điểm PGD, Điểm ngẫu nhiên, 
            # và các Phản ví dụ Hình thức (Formal CE) từ CROWN.
            x_train = self.attacker.get_training_batch(self.batch_size, self.rho)
            
            # --- GIAI ĐOẠN 2: CẬP NHẬT TRỌNG SỐ (LEARNING) ---
            self.opt_ctrl.zero_grad()
            self.opt_lyap.zero_grad()
            
            loss, F_x, v_curr = self.compute_loss(x_train)
            
            loss.backward()
            self.opt_ctrl.step()
            self.opt_lyap.step()
            
            # --- KIỂM CHỨNG THỰC NGHIỆM TẠI CHỖ (EMPIRICAL LOGGING) ---
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                # Chỉ đánh giá mức độ vi phạm bên trong ROA hiện tại (V(x) < rho)
                inside_roa_mask = v_curr < self.rho
                
                if inside_roa_mask.any():
                    max_violation = torch.max(F_x[inside_roa_mask]).item()
                    safe_ratio = (F_x[inside_roa_mask] <= 0).float().mean().item() * 100
                else:
                    max_violation = 0.0
                    safe_ratio = 100.0
                    
                logging.info(
                    f"Epoch {epoch:04d}/{self.epochs} | Loss: {loss.item():.4f} | "
                    f"Max F(x) in ROA: {max_violation:.4f} | "
                    f"Empirical Safe Ratio: {safe_ratio:.1f}%"
                )