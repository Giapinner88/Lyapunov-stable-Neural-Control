import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    # Allow running this file directly: python core/trainer.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.cegis import CEGISLoop, PGDAttacker
from core.dynamics import CartpoleDynamics, PendulumDynamics
from core.models import NeuralController, NeuralLyapunov
from core.training_config import TrainerConfig
from core.roa_utils import compute_rho_boundary, estimate_roa_size, verify_lyapunov_condition, ROATracker


class LyapunovTrainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.system_name = self.config.system.name

        self.dynamics = self._build_dynamics().to(self.device)
        self.x_min = torch.tensor(self.config.box.x_min, device=self.device)
        self.x_max = torch.tensor(self.config.box.x_max, device=self.device)

        if self.x_min.numel() != self.config.model.nx or self.x_max.numel() != self.config.model.nx:
            raise ValueError("x_min/x_max dimensions must match model.nx")

        self.controller = NeuralController(
            nx=self.config.model.nx,
            nu=self.config.model.nu,
            u_bound=self.config.model.u_bound,
            state_limits=self.config.model.state_limits,
        ).to(self.device)
        self.lyapunov = NeuralLyapunov(
            nx=self.config.model.nx,
            state_limits=self.config.model.state_limits,
        ).to(self.device)

        self.attacker = PGDAttacker(
            self.dynamics,
            self.controller,
            self.lyapunov,
            num_steps=self.config.attacker.num_steps,
            step_size=self.config.attacker.step_size,
            num_restarts=self.config.attacker.num_restarts,
        )
        self.optimizer = optim.Adam(
            list(self.controller.parameters()) + list(self.lyapunov.parameters()),
            lr=self.config.loop.learning_rate,
        )

        self.cegis = CEGISLoop(
            dynamics=self.dynamics,
            controller=self.controller,
            lyapunov=self.lyapunov,
            attacker=self.attacker,
            optimizer=self.optimizer,
            bank_capacity=self.config.cegis.bank_capacity,
            bank_storage_device=self.device.type,
            bank_mode=self.config.cegis.bank_mode,
            replay_new_ratio=self.config.cegis.replay_new_ratio,
            violation_margin=self.config.cegis.violation_margin,
            local_box_radius=self.config.cegis.local_box_radius,
            local_box_samples=self.config.cegis.local_box_samples,
            local_box_weight=self.config.cegis.local_box_weight,
            equilibrium_weight=self.config.cegis.equilibrium_weight,
        )

        self.K, self.S = self.dynamics.get_lqr_baseline()
        self.K = self.K.to(self.device)
        self.S = self.S.to(self.device)
        
        # Initialize ROA tracker for CartPole optimization
        self.roa_tracker = ROATracker(gamma=0.9)
        self.current_rho = None  # Will be computed during training

    def _save_snapshot(self, tag: str) -> None:
        controller_path = Path(self.config.output.controller_path)
        lyapunov_path = Path(self.config.output.lyapunov_path)
        snapshot_dir = controller_path.parent / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        c_name = f"{controller_path.stem}_{tag}.pth"
        v_name = f"{lyapunov_path.stem}_{tag}.pth"
        torch.save(self.controller.state_dict(), snapshot_dir / c_name)
        torch.save(self.lyapunov.state_dict(), snapshot_dir / v_name)

    def load_checkpoints(self, controller_path: str | None = None, lyapunov_path: str | None = None) -> bool:
        controller_ckpt = Path(controller_path or self.config.output.controller_path)
        lyapunov_ckpt = Path(lyapunov_path or self.config.output.lyapunov_path)
        if not (controller_ckpt.exists() and lyapunov_ckpt.exists()):
            return False

        self.controller.load_state_dict(torch.load(controller_ckpt, map_location=self.device))
        self.lyapunov.load_state_dict(torch.load(lyapunov_ckpt, map_location=self.device))
        print(f"[Resume] Loaded controller from: {controller_ckpt}")
        print(f"[Resume] Loaded lyapunov  from: {lyapunov_ckpt}")
        return True

    def _build_dynamics(self):
        if self.system_name == "pendulum":
            return PendulumDynamics()
        if self.system_name == "cartpole":
            return CartpoleDynamics()
        raise ValueError(f"Unsupported system: {self.system_name}")

    def _sample_near_origin(self, batch_size: int, radius: tuple[float, ...]) -> torch.Tensor:
        radius_tensor = torch.as_tensor(radius, device=self.device, dtype=torch.float32)
        if radius_tensor.numel() == 1:
            radius_tensor = radius_tensor.repeat(self.config.model.nx)
        if radius_tensor.numel() != self.config.model.nx:
            raise ValueError("lqr_anchor_radius must have one value or model.nx values")
        noise = torch.rand((batch_size, self.config.model.nx), device=self.device) * 2.0 - 1.0
        return noise * radius_tensor.view(1, -1)

    def _pretrain_lqr(self) -> None:
        print("\n--- BẮT ĐẦU PHASE 1: LQR PRE-TRAINING (Tập trung vùng gốc) ---")
        batch_size = self.config.loop.batch_size

        for epoch in range(self.config.loop.pretrain_epochs):
            self.optimizer.zero_grad()

            batch_origin = int(batch_size * 0.70)
            batch_wide = batch_size - batch_origin

            x_origin = self._sample_near_origin(batch_origin, tuple(v * 0.5 for v in self.config.loop.lqr_anchor_radius))
            x_wide = self._sample_near_origin(batch_wide, self.config.loop.lqr_anchor_radius)
            x_small = torch.cat([x_origin, x_wide], dim=0)

            u_nn = self.controller(x_small)
            u_lqr = -torch.matmul(x_small, self.K.T)
            u_lqr = torch.clamp(u_lqr, min=-self.controller.u_bound, max=self.controller.u_bound)
            loss_u = F.mse_loss(u_nn, u_lqr)

            V_nn = self.lyapunov(x_small)
            V_lqr = torch.einsum("bi,ij,bj->b", x_small, self.S, x_small).unsqueeze(1)
            loss_v = F.mse_loss(V_nn, V_lqr)

            loss = loss_u + loss_v
            loss.backward()
            self.optimizer.step()

            if epoch % 20 == 0:
                print(
                    f"Pre-train Epoch {epoch:03d} | "
                    f"Loss (Origin={batch_origin}, Wide={batch_wide}): {loss.item():.6f}"
                )

    def _current_bounds(self, epoch: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        total = max(1, self.config.loop.cegis_epochs - 1)
        progress = epoch / total
        scale = self.config.curriculum.start_scale + (
            self.config.curriculum.end_scale - self.config.curriculum.start_scale
        ) * progress
        return self.x_min * scale, self.x_max * scale, scale

    def _make_attack_seeds(self, current_x_min: torch.Tensor, current_x_max: torch.Tensor) -> torch.Tensor:
        attack_seed_size = self.config.loop.attack_seed_size
        bank_seed_count = int(attack_seed_size * 0.20)
        random_seed_count = attack_seed_size - bank_seed_count

        x_seeds_random = current_x_min + torch.rand((random_seed_count, self.config.model.nx), device=self.device) * (
            current_x_max - current_x_min
        )

        if self.cegis.counterexample_bank.size <= 0:
            return x_seeds_random

        try:
            x_seeds_bank = self.cegis.counterexample_bank.sample(
                bank_seed_count,
                device=self.device,
                dtype=torch.float32,
            )
            noise = torch.randn_like(x_seeds_bank) * 0.01
            x_seeds_bank = torch.clamp(x_seeds_bank + noise, min=current_x_min, max=current_x_max)
            return torch.cat([x_seeds_random, x_seeds_bank], dim=0)
        except RuntimeError:
            return x_seeds_random

    def _sweep_local_region(self, epoch: int, current_bounds: tuple[torch.Tensor, torch.Tensor]) -> None:
        if self.config.loop.sweep_every <= 0:
            return
        if epoch <= 0 or epoch % self.config.loop.sweep_every != 0:
            return

        if self.config.model.nx != 2:
            x_sweep = self._sample_near_origin(512, tuple(v * 1.5 for v in self.config.loop.lqr_anchor_radius))
            x_bad_sweep = self.cegis.attacker.attack(
                x_sweep,
                x_bounds=current_bounds,
                alpha_lyap=self.config.loop.alpha_lyap,
            )
            self.cegis.counterexample_bank.add(x_bad_sweep)
            print(
                f"  [Sweep] Thêm 512 điểm ngẫu nhiên gần gốc vào bank "
                f"(bank size: {self.cegis.counterexample_bank.size})"
            )
            return

        print(f"  [Sweep] Quét lưới xung quanh gốc (±0.15 rad) tại epoch {epoch}...")
        sweep_grid = torch.linspace(-0.15, 0.15, 11)
        x_sweep = []
        for theta in sweep_grid:
            for dot_theta in sweep_grid:
                x_sweep.append([theta.item(), dot_theta.item()])
        x_sweep = torch.tensor(x_sweep, device=self.device)

        x_bad_sweep = self.cegis.attacker.attack(
            x_sweep,
            x_bounds=current_bounds,
            alpha_lyap=self.config.loop.alpha_lyap,
        )
        self.cegis.counterexample_bank.add(x_bad_sweep)
        print(
            f"  [Sweep] Thêm {len(x_sweep)} điểm lưới vào bank "
            f"(bank size: {self.cegis.counterexample_bank.size})"
        )

    def _run_cegis(self) -> None:
        print("\n--- BẮT ĐẦU PHASE 2: CEGIS LOOP ---")
        info = {
            "max_violation": 0.0,
            "mean_violation": 0.0,
        }

        for epoch in range(self.config.loop.cegis_epochs):
            total_loss = 0.0
            bank_size = 0

            current_x_min, current_x_max, current_scale = self._current_bounds(epoch)
            current_bounds = (current_x_min, current_x_max)

            for _ in range(self.config.loop.learner_updates):
                x_seeds = self._make_attack_seeds(current_x_min, current_x_max)
                info = self.cegis.cegis_step(
                    x_seed=x_seeds,
                    x_bounds=current_bounds,
                    K=self.K,
                    S=self.S,
                    alpha_lyap=self.config.loop.alpha_lyap,
                    train_batch_size=self.config.loop.train_batch_size,
                )
                total_loss += info["loss"]
                bank_size = info["bank_size"]

            # Periodically compute ROA boundary (every 20 epochs)
            if epoch % 20 == 0 and epoch > 0:
                try:
                    self.current_rho, _ = compute_rho_boundary(
                        self.lyapunov,
                        self.dynamics,
                        self.controller,
                        current_x_min,
                        current_x_max,
                        n_boundary_samples=500,
                        n_pgd_steps=30,
                        gamma=0.9,
                        device=self.device,
                        verbose=False,
                    )
                    
                    # Estimate ROA size
                    roa_volume, roa_ratio = estimate_roa_size(
                        self.lyapunov,
                        self.current_rho,
                        current_x_min,
                        current_x_max,
                        n_samples=5000,
                        device=self.device,
                    )
                    
                    # Check verification condition
                    x_test_samples = current_x_min + torch.rand(
                        (1000, self.config.model.nx), device=self.device
                    ) * (current_x_max - current_x_min)
                    
                    verify_stats = verify_lyapunov_condition(
                        self.lyapunov,
                        self.dynamics,
                        self.controller,
                        self.current_rho,
                        x_test_samples,
                        alpha_lyap=self.config.loop.alpha_lyap,
                        x_min=current_x_min,
                        x_max=current_x_max,
                    )
                    
                    self.roa_tracker.update(
                        self.current_rho,
                        roa_volume,
                        verify_stats["overall_satisfaction"],
                    )
                except Exception as e:
                    print(f"  [Warning] ROA computation failed at epoch {epoch}: {e}")

            if epoch % self.config.loop.checkpoint_every == 0:
                rho_str = f"ρ={self.current_rho:.4f}" if self.current_rho else "ρ=pending"
                print(
                    f"CEGIS Epoch {epoch:03d} | Quy mô Box: {current_scale*100:.1f}% | "
                    f"Bank: {bank_size} | Loss: {total_loss / self.config.loop.learner_updates:.6f} | "
                    f"Max Violt: {info['max_violation']:.6f} | Mean Violt: {info['mean_violation']:.6f} | {rho_str}"
                )
                self._save_snapshot(f"ep{epoch:03d}")

            self._sweep_local_region(epoch, current_bounds)

    def _save(self) -> None:
        Path(self.config.output.controller_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.output.lyapunov_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.controller.state_dict(), self.config.output.controller_path)
        torch.save(self.lyapunov.state_dict(), self.config.output.lyapunov_path)
        print("Đã lưu mô hình thành công!")

    def run(
        self,
        resume: bool = False,
        skip_pretrain_if_resumed: bool = True,
        controller_path: str | None = None,
        lyapunov_path: str | None = None,
    ) -> None:
        print(f"Sử dụng thiết bị: {self.device}")
        print(f"Hệ động lực: {self.system_name}")
        resumed = False
        if resume:
            resumed = self.load_checkpoints(controller_path=controller_path, lyapunov_path=lyapunov_path)
            if not resumed:
                print("[Resume] Checkpoints not found, training from current initialization.")

        should_pretrain = not (resumed and skip_pretrain_if_resumed)
        if should_pretrain and self.config.loop.pretrain_epochs > 0:
            self._pretrain_lqr()
        elif resumed and skip_pretrain_if_resumed:
            print("[Resume] Skip pre-training phase and continue CEGIS directly.")

        self._run_cegis()
        self._save()


# Backward compatibility for existing imports.
PendulumTrainer = LyapunovTrainer
