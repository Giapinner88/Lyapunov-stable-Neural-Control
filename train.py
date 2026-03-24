import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from core.dynamics import PendulumDynamics
from core.models import NeuralController, NeuralLyapunov
from core.cegis import CEGISLoop, PGDAttacker


def train(pretrain_epochs=150, cegis_epochs=350, alpha_lyap=0.08):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 1. Khởi tạo hệ thống
    dynamics = PendulumDynamics().to(device)
    x_min = torch.tensor([-3.1415, -8.0], device=device)
    x_max = torch.tensor([3.1415, 8.0], device=device)
    x_bounds = (x_min, x_max)

    # Lấy ma trận LQR từ mô hình giải tích
    K, S = dynamics.get_lqr_baseline()
    K = K.to(device)
    S = S.to(device)

    # 2. Khởi tạo models
    controller = NeuralController(nx=2, nu=1, u_bound=6.0).to(device)
    lyapunov = NeuralLyapunov(nx=2).to(device)
    # TỐI ƯU HÓA PHASE 2: Giảm steps (100→50) vì backward quá chậm
    attacker = PGDAttacker(dynamics, controller, lyapunov, num_steps=80, step_size=0.02, num_restarts=6)
    optimizer = optim.Adam(list(controller.parameters()) + list(lyapunov.parameters()), lr=1e-3)

    # 3. Khởi tạo CEGIS loop có CounterexampleBank nội bộ
    cegis = CEGISLoop(
        dynamics=dynamics,
        controller=controller,
        lyapunov=lyapunov,
        attacker=attacker,
        optimizer=optimizer,
        bank_capacity=200000,         # 🆙 TĂNG từ 50k → 200k (Giải pháp 1: Tránh xóa điểm cũ quan trọng)
        bank_storage_device=device.type,
        replay_new_ratio=0.35,
        violation_margin=5e-4,
        local_box_radius=0.20,        # Tăng từ 0.15 - tấn công rộng hơn
        local_box_samples=512,        # Tăng từ 256 - tấn công mạnh hơn
        local_box_weight=0.35,        # Tăng từ 0.25 - ưu tiên phá vùng cục bộ
        equilibrium_weight=0.1,       # Giữ nguyên
    )

    batch_size = 512  # Giảm từ 1000 → 512
    attack_seed_size = 256  # Giảm từ 500 → 256
    train_batch_size = batch_size
    learner_updates = 3

    # =========================================================
    # PHASE 1: LQR PRE-TRAINING (ENHANCED: Origin-Focused)
    # =========================================================
    print("\n--- BẮT ĐẦU PHASE 1: LQR PRE-TRAINING (Tập trung vùng gốc) ---")
    for epoch in range(pretrain_epochs):
        optimizer.zero_grad()

        # CẢI TIẾN: Phân tầng dữ liệu - 70% vùng gốc [±0.05], 30% vùng rộng [±0.1]
        # Lý do: Điểm (0.1, -0.08) cần được học KỸ để sai lệch điều khiển < 0.05
        batch_origin = int(batch_size * 0.70)
        batch_wide = batch_size - batch_origin
        
        # Layer 1: Vùng gốc rất gần (±0.05 rad)
        x_origin = (torch.rand((batch_origin, 2), device=device) * 2.0 - 1.0) * 0.05
        
        # Layer 2: Vùng rộng (±0.1 rad) 
        x_wide = (torch.rand((batch_wide, 2), device=device) * 2.0 - 1.0) * 0.1
        
        # Kết hợp
        x_small = torch.cat([x_origin, x_wide], dim=0)

        u_nn = controller(x_small)
        u_lqr = -torch.matmul(x_small, K.T)
        u_lqr = torch.clamp(u_lqr, min=-2.0, max=2.0)
        loss_u = F.mse_loss(u_nn, u_lqr)

        V_nn = lyapunov(x_small)
        V_lqr = torch.einsum("bi,ij,bj->b", x_small, S, x_small).unsqueeze(1)
        loss_v = F.mse_loss(V_nn, V_lqr)

        loss = loss_u + loss_v
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Pre-train Epoch {epoch:03d} | Loss (Origin={batch_origin}, Wide={batch_wide}): {loss.item():.6f}")

    # =========================================================
    # PHASE 2: CEGIS LOOP
    # =========================================================
    print("\n--- BẮT ĐẦU PHASE 2: CEGIS LOOP ---")
    epochs = int(cegis_epochs)
    for epoch in range(epochs):
        total_loss = 0
        bank_size = 0

        # TÍNH TOÁN RANH GIỚI TÌM KIẾM ĐỘNG (CURRICULUM LEARNING)
        # CẢI TIẾN: Bắt đầu từ vùng nhỏ hơn (70% → 100%) để tập trung nhiều hơn vào gốc
        # Lý do: Điểm (0.1, -0.08) cần bảo vệ tốt trước
        progress = epoch / max(1, epochs - 1)
        current_scale = 0.7 + 0.3 * progress  # Bắt đầu 70% thay vì 50%
        current_x_min = x_min * current_scale
        current_x_max = x_max * current_scale
        current_bounds = (current_x_min, current_x_max)

        for _ in range(learner_updates):
            # 🆙 GIẢI PHÁP 3: Seed trộn 80% random + 20% từ BANK (tái-tấn công điểm xấu cũ)
            bank_seed_count = int(attack_seed_size * 0.20)
            random_seed_count = attack_seed_size - bank_seed_count
            
            # 80% Random trong vùng hiện tại
            x_seeds_random = current_x_min + torch.rand((random_seed_count, 2), device=device) * (current_x_max - current_x_min)
            
            # 20% Từ bank (nếu bank có dữ liệu) + nhiễu nhỏ
            if cegis.counterexample_bank.size > 0:
                try:
                    x_seeds_bank = cegis.counterexample_bank.sample(bank_seed_count, device=device, dtype=torch.float32)
                    # Thêm nhiễu nhỏ xung quanh điểm cũ (để PGD có basis khác)
                    noise = torch.randn_like(x_seeds_bank) * 0.01
                    x_seeds_bank = torch.clamp(x_seeds_bank + noise, min=current_x_min, max=current_x_max)
                    x_seeds = torch.cat([x_seeds_random, x_seeds_bank], dim=0)
                except RuntimeError:
                    # Bank rỗng, chỉ dùng random
                    x_seeds = x_seeds_random
            else:
                x_seeds = x_seeds_random
            
            info = cegis.cegis_step(
                x_seed=x_seeds,
                x_bounds=current_bounds, # <-- Truyền ranh giới đã thu nhỏ
                K=K,
                S=S,
                alpha_lyap=alpha_lyap,
                train_batch_size=train_batch_size,
            )
            total_loss += info["loss"]
            bank_size = info["bank_size"]

        if epoch % 30 == 0:  # Tăng checkpoint từ 20 → 30
            print(
                f"CEGIS Epoch {epoch:03d} | Quy mô Box: {current_scale*100:.1f}% | "
                f"Bank: {bank_size} | Loss: {total_loss / learner_updates:.6f} | "
                f"Max Violt: {info['max_violation']:.6f} | Mean Violt: {info['mean_violation']:.6f}"
            )
        
        # 🆙 GIẢI PHÁP 4: Sweep cục bộ xung quanh gốc mỗi 50 epochs
        # Mục đích: Bắt buộc tìm mọi điểm xấu xung quanh (0, 0)
        if epoch % 50 == 0 and epoch > 0:
            print(f"  [Sweep] Quét lưới xung quanh gốc (±0.15 rad) tại epoch {epoch}...")
            sweep_grid = torch.linspace(-0.15, 0.15, 11)
            x_sweep = []
            for theta in sweep_grid:
                for dot_theta in sweep_grid:
                    x_sweep.append([theta.item(), dot_theta.item()])
            x_sweep = torch.tensor(x_sweep, device=device)
            
            # Attack trên lưới này
            _ = cegis.attacker.attack(x_sweep, x_bounds=current_bounds, alpha_lyap=alpha_lyap)
            # Thêm vào bank
            cegis.counterexample_bank.add(_)
            print(f"  [Sweep] Thêm {len(x_sweep)} điểm lưới vào bank (bank size: {cegis.counterexample_bank.size})")

    # 4. Lưu trọng số
    torch.save(controller.state_dict(), "pendulum_controller.pth")
    torch.save(lyapunov.state_dict(), "pendulum_lyapunov.pth")
    print("Đã lưu mô hình thành công!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Lyapunov-stable controller")
    parser.add_argument("--pretrain-epochs", type=int, default=150)  # Tăng từ 80
    parser.add_argument("--cegis-epochs", type=int, default=350)     # Tăng từ 250
    parser.add_argument("--alpha-lyap", type=float, default=0.08)
    args = parser.parse_args()

    train(
        pretrain_epochs=args.pretrain_epochs,
        cegis_epochs=args.cegis_epochs,
        alpha_lyap=args.alpha_lyap,
    )