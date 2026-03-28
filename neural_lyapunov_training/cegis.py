import torch

try:
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
except Exception:
    BoundedModule = None
    BoundedTensor = None
    PerturbationLpNorm = None


def lyapunov_decrease_expression(lyapunov, x_next: torch.Tensor, x: torch.Tensor, alpha_lyap: float) -> torch.Tensor:
    if hasattr(lyapunov, "algebraic_decrease"):
        return lyapunov.algebraic_decrease(x_next, x, alpha_lyap)
    v_curr = lyapunov(x)
    v_next = lyapunov(x_next)
    return v_next - (1.0 - alpha_lyap) * v_curr


class CounterexampleBank:
    """
    Ngân hàng lưu phản ví dụ x_bad để learner luôn học trên cả lỗi mới lẫn lỗi cũ.
    """

    def __init__(self, capacity: int = 50000, storage_device: str = "cpu", mode: str = "fifo"):
        self.capacity = int(capacity)
        self.storage_device = storage_device
        self.mode = str(mode).lower()
        if self.mode not in {"fifo", "reservoir"}:
            raise ValueError("CounterexampleBank mode must be 'fifo' or 'reservoir'.")
        self._x = None
        self._seen = 0

    @property
    def size(self) -> int:
        if self._x is None:
            return 0
        return int(self._x.shape[0])

    def add(self, x_bad: torch.Tensor) -> None:
        x_bad = x_bad.detach().to(self.storage_device)
        if x_bad.numel() == 0:
            return
        self._seen += int(x_bad.shape[0])

        if self.mode == "fifo":
            if self._x is None:
                self._x = x_bad[-self.capacity :]
                return

            self._x = torch.cat([self._x, x_bad], dim=0)
            if self._x.shape[0] > self.capacity:
                self._x = self._x[-self.capacity :]
            return

        # Reservoir mode keeps a uniform sample from all historical counterexamples.
        if self._x is None:
            self._x = x_bad[:0].clone()

        total_seen_before_batch = self._seen - int(x_bad.shape[0])
        for idx in range(x_bad.shape[0]):
            sample = x_bad[idx : idx + 1]
            current_seen = total_seen_before_batch + idx + 1
            if self._x.shape[0] < self.capacity:
                self._x = torch.cat([self._x, sample], dim=0)
                continue

            replace_idx = int(torch.randint(0, current_seen, (1,), device=self._x.device).item())
            if replace_idx < self.capacity:
                self._x[replace_idx : replace_idx + 1] = sample

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

            # Mục tiêu: Cực đại hóa violation
            violation = lyapunov_decrease_expression(self.lyapunov, x_next, x, alpha_lyap)
            
            # SỬA LỖI TẠI ĐÂY: Dùng autograd để lấy trực tiếp đạo hàm của violation theo x.
            # Tránh dùng .backward() để không làm bẩn/tích lũy gradient của các mạng nơ-ron.
            grad_x = torch.autograd.grad(violation.sum(), x)[0]

            with torch.no_grad():
                span = x_max - x_min
                scaled_step = span * self.step_size 
                
                # Đi THEO hướng gradient (Cộng) để LEO LÊN ĐỈNH (Gradient Ascent)
                x_adv = x + scaled_step * grad_x.sign()
                
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
                score = lyapunov_decrease_expression(self.lyapunov, x_next, x_candidate, alpha_lyap)

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
        bank_mode: str = "fifo",
        replay_new_ratio: float = 0.25,
        violation_margin: float = 5e-4,
        local_box_radius: float = 0.15,
        local_box_samples: int = 256,
        local_box_weight: float = 0.2,
        equilibrium_weight: float = 0.1,
        lqr_anchor_weight: float = 0.01,
        local_sampling_mode: str = "levelset",
        local_levelset_c: float | None = None,
        local_levelset_quantile: float = 0.6,
        local_levelset_oversample_factor: int = 6,
        ibp_ratio: float = 0.0,
        ibp_eps: float = 0.01,
        candidate_roa_weight: float = 0.0,
        candidate_roa_num_samples: int = 0,
        candidate_roa_scale: float = 0.4,
        candidate_roa_rho: float | None = None,
        candidate_roa_rho_quantile: float = 0.9,
        candidate_roa_always: bool = False,
        box_lo: torch.Tensor | None = None,
        box_up: torch.Tensor | None = None,
        loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        self.dynamics = dynamics
        self.controller = controller
        self.lyapunov = lyapunov
        self.attacker = attacker
        self.optimizer = optimizer
        self.counterexample_bank = CounterexampleBank(bank_capacity, bank_storage_device, bank_mode)
        self.replay_new_ratio = float(replay_new_ratio)
        self.violation_margin = float(violation_margin)
        self.local_box_radius = float(local_box_radius)
        self.local_box_samples = int(local_box_samples)
        self.local_box_weight = float(local_box_weight)
        self.equilibrium_weight = float(equilibrium_weight)
        self.lqr_anchor_weight = float(lqr_anchor_weight)
        self.local_sampling_mode = str(local_sampling_mode).lower()
        self.local_levelset_c = None if local_levelset_c is None else float(local_levelset_c)
        self.local_levelset_quantile = float(local_levelset_quantile)
        self.local_levelset_oversample_factor = max(1, int(local_levelset_oversample_factor))
        self.ibp_ratio = float(ibp_ratio)
        self.ibp_eps = float(ibp_eps)
        self.candidate_roa_weight = float(candidate_roa_weight)
        self.candidate_roa_num_samples = int(candidate_roa_num_samples)
        self.candidate_roa_scale = float(candidate_roa_scale)
        self.candidate_roa_rho = None if candidate_roa_rho is None else float(candidate_roa_rho)
        self.candidate_roa_rho_quantile = float(candidate_roa_rho_quantile)
        self.candidate_roa_always = bool(candidate_roa_always)
        self.box_lo = None if box_lo is None else torch.as_tensor(box_lo, dtype=torch.float32).view(1, -1)
        self.box_up = None if box_up is None else torch.as_tensor(box_up, dtype=torch.float32).view(1, -1)
        self.loss_weights = tuple(float(w) for w in loss_weights)
        self.last_candidate_roa_loss = 0.0

    def _ibp_decrease_loss(self, x_samples: torch.Tensor, alpha_lyap: float) -> torch.Tensor:
        if self.ibp_ratio <= 0.0:
            return torch.zeros((), device=x_samples.device, dtype=x_samples.dtype)
        if BoundedModule is None or BoundedTensor is None or PerturbationLpNorm is None:
            return torch.zeros((), device=x_samples.device, dtype=x_samples.dtype)

        class DecreaseModel(torch.nn.Module):
            def __init__(self, dynamics, controller, lyapunov, alpha):
                super().__init__()
                self.dynamics = dynamics
                self.controller = controller
                self.lyapunov = lyapunov
                self.alpha = alpha

            def forward(self, x):
                u = self.controller(x)
                x_next = self.dynamics.step(x, u)
                return lyapunov_decrease_expression(self.lyapunov, x_next, x, self.alpha)

        decrease_model = DecreaseModel(self.dynamics, self.controller, self.lyapunov, alpha_lyap)
        bounded_model = BoundedModule(decrease_model, x_samples, device=x_samples.device)
        ptb = PerturbationLpNorm(norm=float("inf"), eps=self.ibp_eps)
        bounded_x = BoundedTensor(x_samples, ptb)
        _, ub = bounded_model.compute_bounds(x=(bounded_x,), method="IBP", bound_upper=True)
        return torch.clamp(ub + self.violation_margin, min=0.0).mean()

    def _sample_local_points(self, x_ref: torch.Tensor) -> torch.Tensor:
        device = x_ref.device
        dtype = x_ref.dtype

        def _sample_box(n: int) -> torch.Tensor:
            return (torch.rand((n, self.dynamics.nx), device=device, dtype=dtype) * 2.0 - 1.0) * self.local_box_radius

        if self.local_sampling_mode != "levelset":
            return _sample_box(self.local_box_samples)

        with torch.no_grad():
            if self.local_levelset_c is not None:
                c_level = float(self.local_levelset_c)
            else:
                q = min(max(self.local_levelset_quantile, 0.0), 1.0)
                v_ref = self.lyapunov(x_ref).view(-1)
                c_level = float(torch.quantile(v_ref, q).item())

            n_candidates = self.local_box_samples * self.local_levelset_oversample_factor
            x_candidates = _sample_box(n_candidates)
            v_candidates = self.lyapunov(x_candidates)
            inside_idx = torch.nonzero((v_candidates <= c_level).squeeze(1), as_tuple=False).squeeze(1)

            if inside_idx.numel() <= 0:
                return _sample_box(self.local_box_samples)

            if inside_idx.numel() >= self.local_box_samples:
                keep = inside_idx[torch.randperm(inside_idx.numel(), device=device)[: self.local_box_samples]]
                return x_candidates[keep]

            pad_idx = inside_idx[torch.randint(0, inside_idx.numel(), (self.local_box_samples - inside_idx.numel(),), device=device)]
            keep = torch.cat([inside_idx, pad_idx], dim=0)
            return x_candidates[keep]

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

    def _candidate_roa_regularizer(
        self,
        x_samples: torch.Tensor,
        x_bounds: tuple,
        base_loss: torch.Tensor,
    ) -> torch.Tensor:
        # Paper-style candidate-ROA regularizer: weight * mean(relu(V(x)/rho - 1)).
        if self.candidate_roa_weight <= 0.0 or self.candidate_roa_num_samples <= 0:
            return torch.zeros((), device=x_samples.device, dtype=x_samples.dtype)

        if (not self.candidate_roa_always) and float(base_loss.detach().item()) <= 0.0:
            return torch.zeros((), device=x_samples.device, dtype=x_samples.dtype)

        x_min, x_max = x_bounds
        x_min_t = torch.as_tensor(x_min, device=x_samples.device, dtype=x_samples.dtype).view(1, -1)
        x_max_t = torch.as_tensor(x_max, device=x_samples.device, dtype=x_samples.dtype).view(1, -1)

        scale = min(max(self.candidate_roa_scale, 1e-3), 1.0)
        center = 0.5 * (x_min_t + x_max_t)
        span = (x_max_t - x_min_t) * scale
        cand_lo = center - 0.5 * span
        cand_hi = center + 0.5 * span
        x_candidate = cand_lo + torch.rand(
            (self.candidate_roa_num_samples, self.dynamics.nx),
            device=x_samples.device,
            dtype=x_samples.dtype,
        ) * (cand_hi - cand_lo)

        if self.candidate_roa_rho is not None and self.candidate_roa_rho > 0.0:
            rho = torch.tensor(self.candidate_roa_rho, device=x_samples.device, dtype=x_samples.dtype)
        else:
            with torch.no_grad():
                v_ref = self.lyapunov(x_samples).view(-1)
                q = min(max(self.candidate_roa_rho_quantile, 0.0), 1.0)
                rho = torch.quantile(v_ref, q).clamp(min=torch.tensor(1e-6, device=v_ref.device, dtype=v_ref.dtype))

        q_val = self.lyapunov(x_candidate) / rho - 1.0
        return self.candidate_roa_weight * torch.nn.functional.relu(q_val).mean()

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
            violation_debug = lyapunov_decrease_expression(self.lyapunov, x_next_debug, x_bad, alpha_lyap)
            max_violation = torch.max(violation_debug).item()
            mean_violation = torch.mean(violation_debug).item()
        
        loss = self.learner_step(
            x_train,
            x_bounds=x_bounds,
            alpha_lyap=alpha_lyap,
            K=K,
            S=S,
        )

        return {
            "loss": loss,
            "bank_size": self.counterexample_bank.size,
            "num_new_bad": int(x_bad.shape[0]),
            "train_batch_size": int(x_train.shape[0]),
            "max_violation": max_violation,
            "mean_violation": mean_violation,
            "candidate_roa_loss": float(self.last_candidate_roa_loss),
        }

    def learner_step(
        self,
        x_samples: torch.Tensor,
        x_bounds: tuple,
        alpha_lyap: float,
        K: torch.Tensor,
        S: torch.Tensor,
    ):
        self.optimizer.zero_grad()
        
        # --- 1. LOSS VI PHẠM TỪ PGD (Mở rộng ranh giới) ---
        u = self.controller(x_samples)
        x_next = self.dynamics.step(x_samples, u)
        violation = lyapunov_decrease_expression(self.lyapunov, x_next, x_samples, alpha_lyap)
        lyap_decrease_loss = torch.mean(torch.relu(violation + self.violation_margin))
        new_x = x_next

        # --- 1.0 BOUNDARY PENALTY (ép trạng thái sau bước nằm trong hộp an toàn) ---
        if self.box_lo is not None and self.box_up is not None and len(self.loss_weights) >= 3:
            box_lo = self.box_lo.to(device=x_samples.device, dtype=x_samples.dtype)
            box_up = self.box_up.to(device=x_samples.device, dtype=x_samples.dtype)
            loss3 = self.loss_weights[2] * (
                torch.nn.functional.relu(box_lo - new_x).sum(dim=1, keepdim=True)
                + torch.nn.functional.relu(new_x - box_up).sum(dim=1, keepdim=True)
            )
            boundary_penalty_loss = torch.mean(loss3)
        else:
            boundary_penalty_loss = torch.zeros((), device=x_samples.device, dtype=x_samples.dtype)

        # --- 1.1 LOSS CỤC BỘ GẦN GỐC (ép RoA lõi chặt hơn) ---
        device = x_samples.device
        x_local = self._sample_local_points(x_samples)
        u_local = self.controller(x_local)
        x_next_local = self.dynamics.step(x_local, u_local)
        local_violation = lyapunov_decrease_expression(self.lyapunov, x_next_local, x_local, alpha_lyap)
        local_decrease_loss = torch.mean(torch.relu(local_violation + self.violation_margin))
        ibp_decrease_loss = self._ibp_decrease_loss(x_samples, alpha_lyap)

        # --- 1.2 LOSS ĐIỂM CÂN BẰNG (giữ điều kiện tại gốc) ---
        x_zero = torch.zeros((1, self.dynamics.nx), device=device)
        u_zero = self.controller(x_zero)
        v_zero = self.lyapunov(x_zero)
        equilibrium_loss = torch.mean(u_zero ** 2) + torch.mean(v_zero ** 2)
        
        # --- 2. LOSS MỎ NEO LQR (Giữ hình dáng vật lý tại trung tâm) ---
        # Sinh một batch nhỏ quanh gốc tọa độ (bán kính 0.1)
        x_small = (torch.rand((128, self.dynamics.nx), device=device) * 2.0 - 1.0) * 0.1
        
        u_nn = self.controller(x_small)
        u_lqr = -torch.matmul(x_small, K.T)
        u_lqr = torch.clamp(u_lqr, min=-self.controller.u_bound, max=self.controller.u_bound)
        loss_u = torch.nn.functional.mse_loss(u_nn, u_lqr)
        
        V_nn = self.lyapunov(x_small)
        V_lqr = torch.einsum("bi,ij,bj->b", x_small, S, x_small).unsqueeze(1)
        loss_v = torch.nn.functional.mse_loss(V_nn, V_lqr)
        
        # Gộp loss: diệt vi phạm global + local + giữ cân bằng + mỏ neo LQR
        candidate_roa_loss = self._candidate_roa_regularizer(
            x_samples=x_samples,
            x_bounds=x_bounds,
            base_loss=lyap_decrease_loss,
        )
        loss = (
            lyap_decrease_loss
            + boundary_penalty_loss
            + self.local_box_weight * local_decrease_loss
            + self.ibp_ratio * ibp_decrease_loss
            + self.equilibrium_weight * equilibrium_loss
            + self.lqr_anchor_weight * (loss_u + loss_v)
            + candidate_roa_loss
        )
        
        loss.backward()
        self.optimizer.step()
        self.last_candidate_roa_loss = float(candidate_roa_loss.detach().item())
        
        # FIX: Trả về loss THỰC TẾ được backprop, không phải từng bộ phận mà bị ignore
        return loss.item()