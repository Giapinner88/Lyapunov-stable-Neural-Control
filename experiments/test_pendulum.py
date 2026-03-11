import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core.systems import InvertedPendulum
from core.models import Controller, LyapunovNetwork

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
    fig.colorbar(contour1, ax=ax1, label='V(x)')
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
    fig.colorbar(contour2, ax=ax2, label='F(x)')
    # Hiển thị lại ranh giới ROA để đối chiếu
    ax2.contour(Theta, DotTheta, V_map, levels=[rho], colors='black', linewidths=2, linestyles='dashed')
    ax2.set_title(r"Lyapunov Derivative Formulation $F(x)$")
    ax2.set_xlabel(r"$\theta$ (rad)")
    ax2.set_ylabel(r"$\dot{\theta}$ (rad/s)")

    plt.tight_layout()
    plt.show()

# --- Phân tích Trực quan Hệ thống đã Huấn luyện ---
def run_visualization():
    print("Đang khởi tạo cấu trúc hệ thống...")
    sys = InvertedPendulum()
    ctrl = Controller(x_dim=2, u_dim=1)
    lyap = LyapunovNetwork(x_dim=2)
    
    # Lấy dữ liệu từ vòng lặp CEGIS số 5 (Theo log bạn vừa cung cấp)
    iter_num = 5
    rho_target = 1.6485 
    
    ctrl_path = f"results/controller_iter_{iter_num}.pth"
    lyap_path = f"results/lyapunov_iter_{iter_num}.pth"
    
    # Tải trọng số nếu tồn tại
    if os.path.exists(ctrl_path) and os.path.exists(lyap_path):
        print(f"[*] Tải thành công bộ trọng số tại CEGIS Iteration {iter_num}.")
        ctrl.load_state_dict(torch.load(ctrl_path, weights_only=True))
        lyap.load_state_dict(torch.load(lyap_path, weights_only=True))
    else:
        print("[-] CẢNH BÁO: Không tìm thấy file trọng số trong thư mục results/.")
        return
        
    print(f"Đang nội suy và kết xuất đồ thị tại mức \u03C1 = {rho_target}...")
    plot_lyapunov_analysis(sys, ctrl, lyap, rho_target, epoch=f"CEGIS Iter {iter_num}")

# Kích hoạt thực thi
if __name__ == "__main__":
    run_visualization()