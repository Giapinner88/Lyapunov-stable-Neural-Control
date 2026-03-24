import torch
import numpy as np
import matplotlib.pyplot as plt
from core.dynamics import PendulumDynamics
from core.models import NeuralController, NeuralLyapunov

PENDULUM_CONTROLLER_PATH = "checkpoints/pendulum/pendulum_controller.pth"
PENDULUM_LYAPUNOV_PATH = "checkpoints/pendulum/pendulum_lyapunov.pth"
PHASE_PORTRAIT_PATH = "reports/pendulum_phase_portrait.png"

def plot_phase_portrait():
    # 1. Khởi tạo môi trường và tải trọng số (Load Weights)
    device = torch.device("cpu") # Vẽ đồ thị thì dùng CPU cho tiện xử lý mảng NumPy
    dynamics = PendulumDynamics().to(device)
    
    controller = NeuralController(nx=2, nu=1, u_bound=6.0).to(device)
    lyapunov = NeuralLyapunov(nx=2).to(device)
    
    try:
        controller.load_state_dict(torch.load(PENDULUM_CONTROLLER_PATH, map_location=device))
        lyapunov.load_state_dict(torch.load(PENDULUM_LYAPUNOV_PATH, map_location=device))
        print("Đã tải thành công trọng số mô hình!")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file trọng số (.pth). Bạn đã chạy train.py xong chưa?")
        return

    controller.eval()
    lyapunov.eval()

    # 2. Xây dựng Lưới Không gian Trạng thái (Meshgrid)
    # Lấy góc từ -pi đến pi, vận tốc từ -8 đến 8
    theta_vals = np.linspace(-np.pi, np.pi, 200)
    theta_dot_vals = np.linspace(-8.0, 8.0, 200)
    Theta, Theta_dot = np.meshgrid(theta_vals, theta_dot_vals)

    # Duỗi lưới thành một Batch khổng lồ (40.000, 2) để đẩy qua mạng Nơ-ron 1 lần duy nhất
    grid_points = np.column_stack([Theta.ravel(), Theta_dot.ravel()])
    x_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)

    # 3. Chạy qua Mạng Nơ-ron và Phương trình Vi phân
    with torch.no_grad():
        # Tính năng lượng Lyapunov V(x)
        V_tensor = lyapunov(x_tensor)
        
        # Tính lực điều khiển u(x)
        u_tensor = controller(x_tensor)
        
        # Tính đạo hàm dx/dt để lấy hướng của vector vận tốc
        dx_tensor = dynamics.continuous_dynamics(x_tensor, u_tensor)

    # Đưa kết quả về lại định dạng ma trận 2D (200x200) để vẽ
    V = V_tensor.numpy().reshape(Theta.shape)
    dTheta = dx_tensor[:, 0].numpy().reshape(Theta.shape)
    dTheta_dot = dx_tensor[:, 1].numpy().reshape(Theta.shape)

    # 4. Vẽ Trực quan (Visualization)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 4.1. Vẽ cái "bát năng lượng" (Lyapunov Level Sets)
    # Dùng hàm contourf (fill) để tạo bản đồ nhiệt
    contour = ax.contourf(Theta, Theta_dot, V, levels=30, cmap='viridis', alpha=0.8)
    fig.colorbar(contour, ax=ax, label='Lyapunov Energy V(x)')
    
    # Vẽ các đường viền rõ nét (Level sets)
    ax.contour(Theta, Theta_dot, V, levels=10, colors='white', linewidths=0.5)

    # 4.2. Vẽ Dòng chảy Vật lý (Vector Field)
    # Dùng streamplot để thấy rõ hệ thống bị "hút" về đâu
    ax.streamplot(Theta, Theta_dot, dTheta, dTheta_dot, color='black', linewidth=0.8, density=1.5, arrowsize=1.2)

    # Đánh dấu điểm cân bằng gốc tọa độ (0,0)
    ax.plot(0, 0, marker='*', color='red', markersize=15, label='Equilibrium (0,0)')

    ax.set_title("Phase Portrait & Lyapunov Level Sets: Inverted Pendulum")
    ax.set_xlabel("Angle Theta (rad)")
    ax.set_ylabel("Angular Velocity Theta_dot (rad/s)")
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-8, 8])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(PHASE_PORTRAIT_PATH, dpi=300)
    print(f"Đã lưu ảnh vẽ tại: {PHASE_PORTRAIT_PATH}")
    plt.show()

if __name__ == "__main__":
    plot_phase_portrait()