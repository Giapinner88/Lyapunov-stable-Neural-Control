import torch


class CounterexampleBank:
    """
    Ngân hàng lưu phản ví dụ x_bad để learner luôn học trên cả lỗi mới lẫn lỗi cũ.
    """

    def __init__(self, capacity: int = 50000, storage_device: str = "cpu"):
        self.capacity = int(capacity)
        self.storage_device = storage_device
        self._x = None

    @property
    def size(self) -> int:
        if self._x is None:
            return 0
        return int(self._x.shape[0])

    def add(self, x_bad: torch.Tensor) -> None:
        x_bad = x_bad.detach().to(self.storage_device)
        if x_bad.numel() == 0:
            return
        if self._x is None:
            self._x = x_bad[-self.capacity :]
            return

        self._x = torch.cat([self._x, x_bad], dim=0)
        if self._x.shape[0] > self.capacity:
            self._x = self._x[-self.capacity :]

    def sample(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.size == 0:
            raise RuntimeError("CounterexampleBank is empty.")
        n = int(n)
        if n <= 0:
            return self._x[:0].to(device=device, dtype=dtype)

        idx = torch.randint(0, self.size, (n,), device=self._x.device)
        return self._x[idx].to(device=device, dtype=dtype)


class PGDAttacker:
    """
    Kẻ tấn công: Tìm kiếm các trạng thái x làm vi phạm điều kiện giảm của hàm Lyapunov.
    Điều kiện lý tưởng: V(x_next) - V(x) <= -alpha * V(x)
    Mục tiêu của PGD: Cực đại hóa mức độ vi phạm (Violation) = V(x_next) - (1 - alpha) * V(x)
    """
    def __init__(
        self,
        dynamics,
        controller,
        lyapunov,
        num_steps=10,
        step_size=0.1,
        num_restarts=8,
        boundary_ratio=0.3,
        local_ratio=0.2,
        noise_scale=0.01,
    ):
        self.dynamics = dynamics
        self.controller = controller
        self.lyapunov = lyapunov
        self.num_steps = num_steps
        self.step_size = step_size
        self.num_restarts = max(1, int(num_restarts))
        self.boundary_ratio = float(boundary_ratio)
        self.local_ratio = float(local_ratio)
        self.noise_scale = float(noise_scale)

    def _sample_restart_seed(self, x_init: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor) -> torch.Tensor:
        """
        Trộn 3 chiến lược khởi tạo để giảm nguy cơ kẹt cực đại cục bộ:
        uniform toàn miền, gần biên, và nhiễu quanh x_init.
        """
        batch_size, nx = x_init.shape
        device = x_init.device
        dtype = x_init.dtype

        n_boundary = int(batch_size * self.boundary_ratio)
        n_local = int(batch_size * self.local_ratio)
        n_uniform = max(0, batch_size - n_boundary - n_local)

        seeds = []
        span = x_max - x_min

        if n_uniform > 0:
            rand_u = torch.rand((n_uniform, nx), device=device, dtype=dtype)
            seeds.append(x_min + rand_u * span)

        if n_boundary > 0:
            side = torch.randint(0, 2, (n_boundary, nx), device=device)
            eps = torch.rand((n_boundary, nx), device=device, dtype=dtype) * 0.05
            near_min = x_min + eps * span
            near_max = x_max - eps * span
            seeds.append(torch.where(side == 0, near_min, near_max))

        if n_local > 0:
            pick_idx = torch.randint(0, batch_size, (n_local,), device=device)
            base = x_init[pick_idx]
            local_noise = torch.randn_like(base) * (0.1 * span)
            seeds.append(torch.clamp(base + local_noise, min=x_min, max=x_max))

        if not seeds:
            return x_init.clone().detach()

        x_seed = torch.cat(seeds, dim=0)
        if x_seed.shape[0] < batch_size:
            pad_idx = torch.randint(0, x_seed.shape[0], (batch_size - x_seed.shape[0],), device=device)
            x_seed = torch.cat([x_seed, x_seed[pad_idx]], dim=0)
        elif x_seed.shape[0] > batch_size:
            perm = torch.randperm(x_seed.shape[0], device=device)[:batch_size]
            x_seed = x_seed[perm]
        return x_seed.detach()

    def _run_single_pgd(self, x_start: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor, alpha_lyap: float) -> torch.Tensor:
        x = x_start.clone().detach().requires_grad_(True)

        for step_idx in range(self.num_steps):
            u = self.controller(x)
            x_next = self.dynamics.step(x, u)

            V_curr = self.lyapunov(x)
            V_next = self.lyapunov(x_next)

            # Đồng bộ với loss Lyapunov của learner: maximize V_next - (1 - alpha) * V_curr
            violation = V_next - (1.0 - alpha_lyap) * V_curr
            loss_attack = -torch.sum(violation)

            loss_attack.backward()

            with torch.no_grad():
                # TÍNH TOÁN BƯỚC CHÂN TỈ LỆ VỚI KHÔNG GIAN
                span = x_max - x_min
                # step_size (vd: 0.01) giờ là 1% của toàn bộ dải vật lý
                scaled_step = span * self.step_size 
                
                x_adv = x + scaled_step * x.grad.sign()
                
                if self.noise_scale > 0.0:
                    anneal = 1.0 - (step_idx / max(1, self.num_steps - 1))
                    x_adv = x_adv + self.noise_scale * anneal * span * torch.randn_like(x_adv)
                
                x_adv = torch.clamp(x_adv, min=x_min, max=x_max)

            x = x_adv.clone().detach().requires_grad_(True)

        return x.detach()

    def attack(self, x_init: torch.Tensor, x_bounds: tuple, alpha_lyap: float = 0.01) -> torch.Tensor:
        """
        x_init: Các điểm hạt giống ban đầu (Batch, nx)
        x_bounds: Giới hạn không gian trạng thái, dạng (x_min, x_max)
        """
        x_min, x_max = x_bounds
        x_min = torch.as_tensor(x_min, device=x_init.device, dtype=x_init.dtype).view(1, -1)
        x_max = torch.as_tensor(x_max, device=x_init.device, dtype=x_init.dtype).view(1, -1)

        best_x = None
        best_score = None

        for restart_idx in range(self.num_restarts):
            if restart_idx == 0:
                x_start = torch.clamp(x_init.clone().detach(), min=x_min, max=x_max)
            else:
                x_start = self._sample_restart_seed(x_init, x_min, x_max)

            x_candidate = self._run_single_pgd(x_start, x_min, x_max, alpha_lyap)

            with torch.no_grad():
                u = self.controller(x_candidate)
                x_next = self.dynamics.step(x_candidate, u)
                V_curr = self.lyapunov(x_candidate)
                V_next = self.lyapunov(x_next)
                score = V_next - (1.0 - alpha_lyap) * V_curr

                if best_score is None:
                    best_score = score
                    best_x = x_candidate
                else:
                    better = score > best_score
                    best_x = torch.where(better, x_candidate, best_x)
                    best_score = torch.where(better, score, best_score)

        return best_x.detach()

class CEGISLoop:
    """
    Trái tim của hệ thống: Vòng lặp đan xen giữa Learner và Attacker
    """
    def __init__(
        self,
        dynamics,
        controller,
        lyapunov,
        attacker,
        optimizer,
        bank_capacity: int = 50000,
        bank_storage_device: str = "cpu",
        replay_new_ratio: float = 0.25,
    ):
        self.dynamics = dynamics
        self.controller = controller
        self.lyapunov = lyapunov
        self.attacker = attacker
        self.optimizer = optimizer
        self.counterexample_bank = CounterexampleBank(bank_capacity, bank_storage_device)
        self.replay_new_ratio = float(replay_new_ratio)

    def _build_training_batch(self, x_new_bad: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Tạo batch huấn luyện trộn lỗi mới + lỗi cũ trong ngân hàng.
        """
        batch_size = int(batch_size)
        new_count = max(1, int(batch_size * self.replay_new_ratio))
        old_count = max(0, batch_size - new_count)

        device = x_new_bad.device
        dtype = x_new_bad.dtype

        idx_new = torch.randint(0, x_new_bad.shape[0], (new_count,), device=device)
        batch_new = x_new_bad[idx_new]

        if old_count > 0 and self.counterexample_bank.size > 0:
            batch_old = self.counterexample_bank.sample(old_count, device=device, dtype=dtype)
            x_batch = torch.cat([batch_new, batch_old], dim=0)
        else:
            if old_count > 0:
                idx_extra = torch.randint(0, x_new_bad.shape[0], (old_count,), device=device)
                x_batch = torch.cat([batch_new, x_new_bad[idx_extra]], dim=0)
            else:
                x_batch = batch_new

        perm = torch.randperm(x_batch.shape[0], device=device)
        return x_batch[perm]

    def cegis_step(
        self,
        x_seed: torch.Tensor,
        x_bounds: tuple,
        K: torch.Tensor,  # <--- Thêm K
        S: torch.Tensor,  # <--- Thêm S
        alpha_lyap: float = 0.01,
        train_batch_size: int | None = None,
    ):
        """
        1) Attacker tìm x_bad và nạp vào ngân hàng.
        2) Learner bốc batch trộn từ ngân hàng để huấn luyện.
        """
        x_bad = self.attacker.attack(x_seed, x_bounds, alpha_lyap=alpha_lyap)
        self.counterexample_bank.add(x_bad)

        if train_batch_size is None:
            train_batch_size = x_seed.shape[0]

        x_train = self._build_training_batch(x_bad, train_batch_size)
        
        # DEBUG: Kiểm tra mức độ violation từ attacker
        with torch.no_grad():
            u_debug = self.controller(x_bad)
            x_next_debug = self.dynamics.step(x_bad, u_debug)
            V_curr_debug = self.lyapunov(x_bad)
            V_next_debug = self.lyapunov(x_next_debug)
            violation_debug = V_next_debug - (1.0 - alpha_lyap) * V_curr_debug
            max_violation = torch.max(violation_debug).item()
            mean_violation = torch.mean(violation_debug).item()
        
        loss = self.learner_step(x_train, alpha_lyap=alpha_lyap, K=K, S=S)  # <--- Truyền K, S vào đây

        return {
            "loss": loss,
            "bank_size": self.counterexample_bank.size,
            "num_new_bad": int(x_bad.shape[0]),
            "train_batch_size": int(x_train.shape[0]),
            "max_violation": max_violation,
            "mean_violation": mean_violation,
        }

    def learner_step(self, x_samples: torch.Tensor, alpha_lyap: float, K: torch.Tensor, S: torch.Tensor):
        self.optimizer.zero_grad()
        
        # --- 1. LOSS VI PHẠM TỪ PGD (Mở rộng ranh giới) ---
        u = self.controller(x_samples)
        x_next = self.dynamics.step(x_samples, u)
        V_curr = self.lyapunov(x_samples)
        V_next = self.lyapunov(x_next)
        lyap_decrease_loss = torch.mean(torch.relu(V_next - (1.0 - alpha_lyap) * V_curr))
        
        # --- 2. LOSS MỎ NEO LQR (Giữ hình dáng vật lý tại trung tâm) ---
        device = x_samples.device
        # Sinh một batch nhỏ quanh gốc tọa độ (bán kính 0.1)
        x_small = (torch.rand((128, self.dynamics.nx), device=device) * 2.0 - 1.0) * 0.1
        
        u_nn = self.controller(x_small)
        u_lqr = -torch.matmul(x_small, K.T)
        u_lqr = torch.clamp(u_lqr, min=-self.controller.u_bound, max=self.controller.u_bound)
        loss_u = torch.nn.functional.mse_loss(u_nn, u_lqr)
        
        V_nn = self.lyapunov(x_small)
        V_lqr = torch.einsum("bi,ij,bj->b", x_small, S, x_small).unsqueeze(1)
        loss_v = torch.nn.functional.mse_loss(V_nn, V_lqr)
        
        # Gộp loss: Diệt điểm mù (trọng số 1.0) + Giữ mỏ neo (trọng số 0.01 - GẢM từ 0.1)
        # Giảm anchor weight để learner không bị distract, tập trung vào violation
        loss = lyap_decrease_loss + 0.01 * (loss_u + loss_v)
        
        loss.backward()
        self.optimizer.step()
        
        # FIX: Trả về loss THỰC TẾ được backprop, không phải từng bộ phận mà bị ignore
        return loss.item()