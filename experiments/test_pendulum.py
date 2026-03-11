import torch
import numpy as np
import matplotlib.pyplot as plt

# Giả định các module đã được import từ thư mục dự án
# from core.systems import InvertedPendulum
# from core.models import Controller, LyapunovNetwork
# from synthesis.attacks import PGDAttacker
# from synthesis.trainer import CEGISTrainer

def plot_lyapunov_analysis(system, controller, lyapunov, rho, epoch):
    """
    Trực quan hóa Hàm Lyapunov và Không gian Pha.
    """
    system.eval()
    controller.eval()
    lyapunov.eval()

    # Tạo lưới không gian trạng thái (Grid)
    theta_range = np.linspace(-np.pi, np.pi, 100)
    dot_theta_range = np.linspace(-8, 8, 100)
    Theta, DotTheta = np.meshgrid(theta_range, dot_theta_range)
    
    # Chuyển lưới thành Tensor để đưa vào mạng NN
    grid_states = np.vstack([Theta.ravel(), DotTheta.ravel()]).T
    x_tensor = torch.tensor(grid_states, dtype=torch.float32)

    with torch.no_grad():
        # Tính toán u, x_next, V(x) và V(x_next)
        u = controller(x_tensor)
        x_next = system(x_tensor, u)
        
        v_curr = lyapunov(x_tensor).squeeze()
        v_next = lyapunov(x_next).squeeze()
        
        # F(x) = V(x_next) - (1 - kappa) * V(x)
        kappa = 0.1
        F_x = v_next - (1 - kappa) * v_curr

    # Định dạng lại kích thước lưới
    V_map = v_curr.numpy().reshape(100, 100)
    F_map = F_x.numpy().reshape(100, 100)

    # --- Vẽ biểu đồ ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Biểu đồ Đường mức V(x) (Lyapunov Level Sets)
    ax1 = axes[0]
    contour1 = ax1.contourf(Theta, DotTheta, V_map, levels=50, cmap='viridis')
    plt.colorbar(contour1, ax=ax1, label='V(x)')
    # Vẽ ranh giới ROA được chứng nhận (Certified ROA boundary)
    ax1.contour(Theta, DotTheta, V_map, levels=[rho], colors='red', linewidths=3)
    ax1.plot(0, 0, 'r*', markersize=10) # Gốc tọa độ
    ax1.set_title(f"Lyapunov Function V(x) - Epoch {epoch}\nRed line: V(x) = {rho}")
    ax1.set_xlabel(r"$\theta$ (rad)")
    ax1.set_ylabel(r"$\dot{\theta}$ (rad/s)")

    # 2. Bản đồ nhiệt F(x) (Violation Landscape)
    ax2 = axes[1]
    # Thiết lập màu: Đỏ (Vi phạm: F > 0), Xanh (An toàn: F < 0)
    vmax = max(abs(F_map.min()), abs(F_map.max()))
    contour2 = ax2.contourf(Theta, DotTheta, F_map, levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(contour2, ax=ax2, label='F(x)')
    # Hiển thị lại ranh giới ROA để đối chiếu
    ax2.contour(Theta, DotTheta, V_map, levels=[rho], colors='black', linewidths=2, linestyles='dashed')
    ax2.set_title(r"Lyapunov Derivative Formulation $F(x)$")
    ax2.set_xlabel(r"$\theta$ (rad)")
    ax2.set_ylabel(r"$\dot{\theta}$ (rad/s)")

    plt.tight_layout()
    plt.show()

# --- Giả lập Quá trình chạy thử (Dummy Runner) ---
def run_test():
    print("Khởi tạo hệ thống và mạng nơ-ron...")
    # Khởi tạo mô hình
    # sys = InvertedPendulum()
    # ctrl = Controller()
    # lyap = LyapunovNetwork()
    
    rho_current = 0.1 # Bắt đầu với ROA nhỏ
    
    print("Vẽ trạng thái khởi tạo ngẫu nhiên (Epoch 0)...")
    # plot_lyapunov_analysis(sys, ctrl, lyap, rho_current, 0)
    
    # Ở đây bạn sẽ gọi vòng lặp huấn luyện:
    # trainer.train(epochs=100)
    
    print("Vẽ trạng thái sau khi huấn luyện (Epoch 100)...")
    # plot_lyapunov_analysis(sys, ctrl, lyap, rho_current, 100)

# run_test()