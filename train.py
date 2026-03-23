import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from core.dynamics import PendulumDynamics
from core.models import NeuralController, NeuralLyapunov
from core.cegis import CEGISLoop, PGDAttacker


def train(pretrain_epochs=80, cegis_epochs=250, alpha_lyap=0.08):
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
        bank_capacity=50000,
        bank_storage_device=device.type,
        replay_new_ratio=0.35,
        violation_margin=5e-4,
        local_box_radius=0.15,
        local_box_samples=256,
        local_box_weight=0.25,
        equilibrium_weight=0.1,
    )

    batch_size = 512  # Giảm từ 1000 → 512
    attack_seed_size = 256  # Giảm từ 500 → 256
    train_batch_size = batch_size
    learner_updates = 3

    # =========================================================
    # PHASE 1: LQR PRE-TRAINING
    # =========================================================
    print("\n--- BẮT ĐẦU PHASE 1: LQR PRE-TRAINING ---")
    for epoch in range(pretrain_epochs):
        optimizer.zero_grad()

        # Sinh điểm gần gốc để học xấp xỉ vùng cục bộ ổn định.
        x_small = (torch.rand((batch_size, 2), device=device) * 2.0 - 1.0) * 0.1

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
            print(f"Pre-train Epoch {epoch:03d} | LQR Loss: {loss.item():.6f}")

    # =========================================================
    # PHASE 2: CEGIS LOOP
    # =========================================================
    print("\n--- BẮT ĐẦU PHASE 2: CEGIS LOOP ---")
    epochs = int(cegis_epochs)
    for epoch in range(epochs):
        total_loss = 0
        bank_size = 0

        # TÍNH TOÁN RANH GIỚI TÌM KIẾM ĐỘNG (CURRICULUM LEARNING)
        # Đi từ 50% không gian ở Epoch 0 lên 100% không gian ở Epoch cuối (từ 10% → 50%)
        progress = epoch / max(1, epochs - 1)
        current_scale = 0.5 + 0.5 * progress
        current_x_min = x_min * current_scale
        current_x_max = x_max * current_scale
        current_bounds = (current_x_min, current_x_max)

        for _ in range(learner_updates):
            # Seed rải đều trong vùng giới hạn hiện tại, KHÔNG rải toàn miền
            x_seeds = current_x_min + torch.rand((attack_seed_size, 2), device=device) * (current_x_max - current_x_min)
            
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

    # 4. Lưu trọng số
    torch.save(controller.state_dict(), "pendulum_controller.pth")
    torch.save(lyapunov.state_dict(), "pendulum_lyapunov.pth")
    print("Đã lưu mô hình thành công!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Lyapunov-stable controller")
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--cegis-epochs", type=int, default=250)
    parser.add_argument("--alpha-lyap", type=float, default=0.08)
    args = parser.parse_args()

    train(
        pretrain_epochs=args.pretrain_epochs,
        cegis_epochs=args.cegis_epochs,
        alpha_lyap=args.alpha_lyap,
    )