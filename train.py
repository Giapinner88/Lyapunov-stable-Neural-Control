import torch
import torch.nn.functional as F
import torch.optim as optim
from core.dynamics import PendulumDynamics
from core.models import NeuralController, NeuralLyapunov
from core.cegis import CEGISLoop, PGDAttacker


def train():
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
    controller = NeuralController(nx=2, nu=1, u_bound=2.0).to(device)
    lyapunov = NeuralLyapunov(nx=2).to(device)
    attacker = PGDAttacker(dynamics, controller, lyapunov, num_steps=10, num_restarts=3)
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
        replay_new_ratio=0.25,
    )

    batch_size = 1000
    attack_seed_size = 500
    train_batch_size = batch_size
    learner_updates = 5

    # =========================================================
    # PHASE 1: LQR PRE-TRAINING
    # =========================================================
    print("\n--- BẮT ĐẦU PHASE 1: LQR PRE-TRAINING ---")
    for epoch in range(100):
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
    epochs = 500
    alpha_lyap = 0.05
    for epoch in range(epochs):
        total_loss = 0
        bank_size = 0

        for _ in range(learner_updates):
            # Seed rải đều toàn miền để attacker khám phá phản ví dụ mới liên tục.
            x_seeds = x_min + torch.rand((attack_seed_size, 2), device=device) * (x_max - x_min)
            info = cegis.cegis_step(
                x_seed=x_seeds,
                x_bounds=x_bounds,
                alpha_lyap=alpha_lyap,
                train_batch_size=train_batch_size,
            )
            total_loss += info["loss"]
            bank_size = info["bank_size"]

        if epoch % 20 == 0:
            print(
                f"CEGIS Epoch {epoch:03d} | Counterexample Bank: {bank_size} | "
                f"Violation Loss: {total_loss / learner_updates:.6f}"
            )

    # 4. Lưu trọng số
    torch.save(controller.state_dict(), "pendulum_controller.pth")
    torch.save(lyapunov.state_dict(), "pendulum_lyapunov.pth")
    print("Đã lưu mô hình thành công!")

if __name__ == "__main__":
    train()