import torch
import torch.nn.functional as F
import torch.optim as optim

class CEGISTrainer:
    def __init__(self, system, controller, lyapunov, attacker, config):
        """
        Khởi tạo vòng lặp huấn luyện CEGIS.
        """
        self.system = system
        self.controller = controller
        self.lyapunov = lyapunov
        self.attacker = attacker
        
        # Siêu tham số
        self.rho = config.get('rho', 0.1)
        self.kappa = config.get('kappa', 0.1)
        self.epochs = config.get('epochs', 1000)
        self.batch_size = config.get('batch_size', 5000)
        self.lambda_margin = config.get('lambda_margin', 100.0) # Trọng số nới lỏng
        
        # Tách biệt Optimizer cho hai mạng để tránh xung đột gradient
        self.opt_ctrl = optim.Adam(self.controller.parameters(), lr=1e-3)
        self.opt_lyap = optim.Adam(self.lyapunov.parameters(), lr=1e-3)

    def compute_loss(self, x):
        """
        Tính toán hàm suy hao tổng hợp.
        """
        # 1. Động lực học tiến (Forward Dynamics)
        u = self.controller(x)
        x_next = self.system(x, u)
        
        v_curr = self.lyapunov(x)
        v_next = self.lyapunov(x_next)
        
        # 2. Sai số giảm Lyapunov (Lyapunov Derivative Violation)
        F_x = v_next - (1 - self.kappa) * v_curr
        
        # 3. Hàm suy hao nới lỏng (Relaxed Loss)
        margin = self.lambda_margin * F.relu(v_curr - self.rho)
        loss_lyap = torch.mean(F.relu(F_x - margin))
        
        # 4. Hàm suy hao hiệu suất (Performance Loss)
        # Khuyến khích V(x) nhỏ gọn quanh gốc tọa độ và tiết kiệm tín hiệu u
        loss_perf = 0.01 * torch.mean(v_curr) + 0.01 * torch.mean(u**2)
        
        total_loss = loss_lyap + loss_perf
        return total_loss, F_x, v_curr

    def train(self):
        """
        Thực thi vòng lặp CEGIS.
        """
        for epoch in range(self.epochs):
            self.controller.train()
            self.lyapunov.train()
            
            # --- GIAI ĐOẠN 1: TÌM PHẢN VÍ DỤ (SYNTHESIS/ATTACK) ---
            # Sử dụng PGD để săn lùng các trạng thái vi phạm nghiêm trọng nhất
            x_adv = self.attacker.find_counter_examples(self.batch_size, self.rho)
            
            # Trộn thêm nhiễu ngẫu nhiên để duy trì tính tổng quát (Exploration)
            # Giả định attacker có hàm random_sample
            x_rand = self.attacker.sample_random(self.batch_size // 5) 
            x_train = torch.cat([x_adv, x_rand], dim=0)
            
            # --- GIAI ĐOẠN 2: CẬP NHẬT TRỌNG SỐ (LEARNING) ---
            self.opt_ctrl.zero_grad()
            self.opt_lyap.zero_grad()
            
            loss, F_x, v_curr = self.compute_loss(x_train)
            
            loss.backward()
            self.opt_ctrl.step()
            self.opt_lyap.step()
            
            # --- KIỂM CHỨNG & LOGGING TẠI CHỖ ---
            if epoch % 10 == 0:
                # Chỉ đánh giá mức độ vi phạm bên trong ROA hiện tại (v_curr < rho)
                inside_roa_mask = v_curr < self.rho
                if inside_roa_mask.any():
                    max_violation = torch.max(F_x[inside_roa_mask]).item()
                    safe_ratio = (F_x[inside_roa_mask] <= 0).float().mean().item() * 100
                else:
                    max_violation = 0.0
                    safe_ratio = 100.0
                    
                print(f"Epoch {epoch:04d} | Loss: {loss.item():.4f} | "
                      f"Max F(x) in ROA: {max_violation:.4f} | "
                      f"Safe Points: {safe_ratio:.1f}%")