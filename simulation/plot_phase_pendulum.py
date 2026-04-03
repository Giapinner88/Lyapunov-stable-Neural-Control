"""
Vẽ biểu đồ chân dung pha (Phase Portrait) dạng heatmap với:
- Hình nền: Heatmap Lyapunov function V(x)
- Vùng ROA: Đường contour rho
- Vector quỹ đạo: Arrows chỉ hướng động lực
- Điểm cân bằng và giới hạn vùng quan tâm
"""
import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from neural_lyapunov_training.pendulum import PendulumDynamics
from neural_lyapunov_training.dynamical_system import SecondOrderDiscreteTimeSystem
from neural_lyapunov_training.controllers import NeuralNetworkController, NeuralNetworkLuenbergerObserver
from neural_lyapunov_training.lyapunov import NeuralNetworkLyapunov


FONT_SCALE = 0.9


def fs(size):
    return max(6, size * FONT_SCALE)


def load_lyapunov_and_controller(device, load_controllers=False):
    """Tải Lyapunov function và optionally neural controller"""
    pendulum_continuous = PendulumDynamics(m=0.15, l=0.5, beta=0.1)
    dynamics = SecondOrderDiscreteTimeSystem(pendulum_continuous, dt=0.01)
    
    # Khởi tạo Lyapunov network
    lyapunov_nn = NeuralNetworkLyapunov(
        goal_state=torch.tensor([0.0, 0.0]),
        hidden_widths=[64, 64, 32],
        x_dim=2,
        R_rows=2,
        absolute_output=True,
        eps=0.01,
        activation=nn.LeakyReLU,
    ).to(device)
    
    controller = None
    observer = None
    
    if load_controllers:
        # Khởi tạo Controller
        controller = NeuralNetworkController(
            in_dim=3, out_dim=1, hidden_dim=8, nlayer=4, clip_output="clamp",
            # MuJoCo pendulum.xml uses actuator ctrlrange [-1, 1].
            # Đổi lại 0.25 nếu khảo sát output-feedback đã train, hoặc giữ ±1 nếu khảo sát mujoco-feedback.
            u_lo=torch.tensor([-1.0]), u_up=torch.tensor([1.0]),
            x_equilibrium=torch.zeros(3), u_equilibrium=torch.zeros(1)
        ).to(device)
        
        # Khởi tạo Observer
        h = lambda x: pendulum_continuous.h(x)
        observer = NeuralNetworkLuenbergerObserver(
            z_dim=2, y_dim=1, dynamics=dynamics, h=h,
            zero_obs_error=torch.zeros(1, 1), fc_hidden_dim=[8, 8]
        ).to(device)

    # Tải trọng số nếu có
    try:
        ckpt_path = PROJECT_ROOT / "models" / "pendulum_output_feedback.pth"
        if ckpt_path.exists():
            print(f"Checkpoint tồn tại: {ckpt_path}")
    except Exception as e:
        print(f"Lỗi kiểm tra checkpoint: {e}")

    lyapunov_nn.eval()
    if controller is not None:
        controller.eval()
    if observer is not None:
        observer.eval()
    
    return lyapunov_nn, controller, observer, dynamics, pendulum_continuous


def simulate_trajectory(
    x0_list, 
    dynamics, 
    max_steps=100, 
    device=torch.device("cpu")
):
    """Mô phỏng quỹ đạo từ các điểm ban đầu với điều khiển neutral (u=0)"""
    trajectories = []
    
    for x0 in x0_list:
        traj = [x0.clone()]
        
        with torch.no_grad():
            for step in range(max_steps):
                x = traj[-1].unsqueeze(0)  # Thêm batch dimension: [state_dim] => [1, state_dim]
                u = torch.zeros(1, 1, device=device)  # [1, 1]
                
                # Cập nhật trạng thái
                x_next = dynamics(x, u).squeeze(0)  # [1, state_dim] => [state_dim]
                
                traj.append(x_next.cpu())
        
        trajectories.append(torch.stack(traj))
    
    return trajectories


from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_phase_portrait_heatmap(
    lyapunov_nn,
    dynamics,
    lower_limit=None,
    upper_limit=None,
    rho=0.1,
    roa_limits=None,
    grid_size=500,
    device=torch.device("cpu"),
    save_path=None,
    dpi=150
):
    if lower_limit is None:
        lower_limit = torch.tensor([-np.pi, -2*np.pi], device=device)
    if upper_limit is None:
        upper_limit = torch.tensor([np.pi,   2*np.pi], device=device)
    if roa_limits is None:
        roa_limits = (lower_limit, upper_limit)

    lower_limit = lower_limit.to(device)
    upper_limit = upper_limit.to(device)
    roa_lower, roa_upper = roa_limits
    roa_lower = roa_lower.to(device)
    roa_upper = roa_upper.to(device)

    xl = lower_limit[0].item();  xu = upper_limit[0].item()
    yl = lower_limit[1].item();  yu = upper_limit[1].item()

    # ── Tạo grid & tính V ────────────────────────────────────────────────────
    x_ticks = torch.linspace(lower_limit[0], upper_limit[0], grid_size, device=device)
    y_ticks = torch.linspace(lower_limit[1], upper_limit[1], grid_size, device=device)
    grid_x, grid_y = torch.meshgrid(x_ticks, y_ticks, indexing='ij')

    X_flat = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    with torch.no_grad():
        V_flat = lyapunov_nn(X_flat).squeeze()
    V_grid    = V_flat.reshape(grid_size, grid_size).cpu().numpy()
    grid_x_np = grid_x.cpu().numpy()
    grid_y_np = grid_y.cpu().numpy()

    # ── ROA expansion: A(rho) và dA/drho ─────────────────────────────────────
    N_RHO     = 60
    rho_max   = float(V_grid.max()) * 0.85
    rho_arr   = np.linspace(rho_max / N_RHO, rho_max, N_RHO)
    cell_area = (xu - xl) * (yu - yl) / (grid_size ** 2)
    area_arr  = np.array([(V_grid <= r).sum() * cell_area for r in rho_arr])
    deriv_arr = np.gradient(area_arr, rho_arr[1] - rho_arr[0])

    # ── Figure Layout (Sử dụng constrained_layout) ───────────────────────────
    fig = plt.figure(figsize=(16, 9), dpi=dpi, layout='constrained')
    gs  = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], height_ratios=[1, 1])
    
    ax_phase = fig.add_subplot(gs[:, 0])   
    ax_area  = fig.add_subplot(gs[0, 1])   
    ax_deriv = fig.add_subplot(gs[1, 1])   

    # ════════════════════════════════════════════════════════════════════════
    # AX_PHASE — Phase portrait + heatmap + contours
    # ════════════════════════════════════════════════════════════════════════
    cf = ax_phase.contourf(grid_x_np, grid_y_np, V_grid,
                           levels=60, cmap='viridis', alpha=0.82)
    
    # ── Xử lý Colorbar bằng make_axes_locatable (Khắc phục lỗi layout) ──────
    divider = make_axes_locatable(ax_phase)
    
    # Colorbar 1: Giá trị V(x)
    cax1 = divider.append_axes("right", size="4%", pad=0.15)
    cbar = fig.colorbar(cf, cax=cax1)
    cbar.set_label(r'$V(x)$', fontsize=fs(10))
    cbar.ax.tick_params(labelsize=fs(8))

    ax_phase.contour(grid_x_np, grid_y_np, V_grid,
                     levels=14, colors='white', linewidths=0.35, alpha=0.18)

    # Contours mở rộng ROA
    N_LEVELS   = 10
    rho_levels = np.linspace(rho_max / N_LEVELS, rho_max, N_LEVELS)
    cmap_exp   = plt.cm.plasma
    for idx, r in enumerate(rho_levels):
        t     = idx / (N_LEVELS - 1)
        color = cmap_exp(t)
        lw    = 0.9 + t * 1.2
        alpha = 0.55 + t * 0.35
        ax_phase.contour(grid_x_np, grid_y_np, V_grid,
                         levels=[r], colors=[color], linewidths=lw, alpha=alpha)

    # Colorbar 2: Mức rho (expansion)
    sm = plt.cm.ScalarMappable(
        cmap=cmap_exp,
        norm=plt.Normalize(vmin=rho_levels[0], vmax=rho_levels[-1])
    )
    sm.set_array([])
    cax2 = divider.append_axes("right", size="4%", pad=0.55)
    cbar2 = fig.colorbar(sm, cax=cax2)
    cbar2.set_label(r'$\rho$ (expansion)', fontsize=fs(10))
    cbar2.ax.tick_params(labelsize=fs(8))

    # ROA boundary \rho được chọn
    cs = ax_phase.contour(grid_x_np, grid_y_np, V_grid,
                          levels=[rho], colors='#ff4444',
                          linewidths=2.8, linestyles='--')
    ax_phase.clabel(cs, fmt=rf'$\rho$ = {rho:.3f}', fontsize=fs(9),
                    inline=True, inline_spacing=4, colors='#ff4444')

    # Vùng quan tâm ROA
    has_roa_box = (
        not torch.allclose(roa_lower, lower_limit) or
        not torch.allclose(roa_upper, upper_limit)
    )
    if has_roa_box:
        rx = roa_lower[0].item();  rw = (roa_upper[0] - roa_lower[0]).item()
        ry = roa_lower[1].item();  rh = (roa_upper[1] - roa_lower[1]).item()
        ax_phase.add_patch(plt.Rectangle(
            (rx, ry), rw, rh,
            linewidth=2, edgecolor='#ffaa00', facecolor='#ffaa0015',
            linestyle=':', zorder=2
        ))

    # Vector field
    n_arr  = 16
    X_arr  = torch.tensor(
        [[xv, yv]
         for xv in np.linspace(xl, xu, n_arr)
         for yv in np.linspace(yl, yu, n_arr)],
        dtype=torch.float32, device=device
    )
    u_zero = torch.zeros(len(X_arr), 1, device=device)
    with torch.no_grad():
        X_next = dynamics(X_arr, u_zero)
    dX  = (X_next - X_arr).cpu().numpy()
    mag = np.hypot(dX[:, 0], dX[:, 1]) + 1e-8
    pts = X_arr.cpu().numpy()
    ax_phase.quiver(
        pts[:, 0], pts[:, 1],
        dX[:, 0] / mag * (xu - xl) / n_arr * 0.42,
        dX[:, 1] / mag * (yu - yl) / n_arr * 0.42,
        color='white', alpha=0.45, width=0.003,
        headwidth=4, headlength=4, headaxislength=3.5,
        scale=1, scale_units='xy', angles='xy', zorder=4
    )

    # Quỹ đạo mẫu
    init_points = [
        torch.tensor([th, dth], device=device)
        for th  in np.linspace(xl * 0.9, xu * 0.9, 6)
        for dth in np.linspace(yl * 0.7, yu * 0.7, 4)
    ]
    for traj in simulate_trajectory(init_points, dynamics, max_steps=120, device=device):
        traj_np = traj.numpy()
        ax_phase.plot(traj_np[:, 0], traj_np[:, 1],
                      color='#00ddcc', linewidth=0.9, alpha=0.40, zorder=5)
        ax_phase.plot(traj_np[0, 0], traj_np[0, 1],
                      'o', color='#00ddcc', markersize=3, alpha=0.60, zorder=6)

    # Điểm cân bằng
    ax_phase.plot(0, 0, '*', color='#ff3333', markersize=13, zorder=7,
                  markeredgecolor='white', markeredgewidth=0.7)

    # Legend & trục
    from matplotlib.lines   import Line2D
    from matplotlib.patches import Patch
    leg_elems = [
        Line2D([0], [0], color='#ff4444', lw=2.2, linestyle='--',
               label=rf'$\rho$ = {rho:.3f} (selected)'),
        Line2D([0], [0], color='#00ddcc', lw=1.4, alpha=0.6,
               label='Quỹ đạo (u = 0)'),
        Line2D([0], [0], color='white', lw=0, marker='*', markersize=10,
               markerfacecolor='#ff3333', label='Điểm cân bằng'),
    ]
    if has_roa_box:
        leg_elems.append(Patch(facecolor='#ffaa0015', edgecolor='#ffaa00',
                               linestyle=':', linewidth=1.5,
                               label='Vùng ROA interest'))
    
    ax_phase.legend(handles=leg_elems, loc='upper right', fontsize=fs(8),
                    framealpha=0.55, edgecolor='white')
    ax_phase.set_xlim(xl, xu);  ax_phase.set_ylim(yl, yu)
    
    # Loại bỏ set_box_aspect để tránh xung đột với constrained_layout
    ax_phase.set_xlabel(r'$\theta$ (rad)', fontsize=fs(11))
    ax_phase.set_ylabel(r'$\dot{\theta}$ (rad/s)', fontsize=fs(11))
    ax_phase.set_title(
        rf'Phase Portrait — Lyapunov Heatmap + ROA Expansion  ($\rho$ = {rho:.4f})',
        fontsize=fs(12), pad=8
    )
    ax_phase.tick_params(labelsize=fs(9))
    ax_phase.grid(True, color='white', alpha=0.10, linewidth=0.5)

    # ════════════════════════════════════════════════════════════════════════
    # AX_AREA — A(ρ)
    # ════════════════════════════════════════════════════════════════════════
    ax_area.plot(rho_arr, area_arr, color='#534AB7', linewidth=2.0)
    ax_area.fill_between(rho_arr, area_arr, alpha=0.12, color='#534AB7')

    rho_hl_idx = int(np.argmin(np.abs(rho_arr - rho)))
    ax_area.axvline(rho, color='#ff4444', linewidth=1.5, linestyle='--', alpha=0.8)
    ax_area.plot(rho, area_arr[rho_hl_idx], 'o', color='#ff4444', markersize=7, zorder=5)
    ax_area.annotate(
        rf'A = {area_arr[rho_hl_idx]:.3f}',
        xy=(rho, area_arr[rho_hl_idx]),
        xytext=(8, -14), textcoords='offset points',
        fontsize=fs(8), color='#ff4444'
    )
    ax_area.set_xlabel(r'$\rho$', fontsize=fs(10))
    ax_area.set_ylabel(r'Diện tích ($rad \times rad/s$)', fontsize=fs(9))
    ax_area.set_title(r'$A(\rho)$ — Diện tích ROA theo $\rho$', fontsize=fs(11))
    ax_area.tick_params(labelsize=fs(8))
    ax_area.grid(True, alpha=0.3)

    # ════════════════════════════════════════════════════════════════════════
    # AX_DERIV — dA/dρ
    # ════════════════════════════════════════════════════════════════════════
    ax_deriv.plot(rho_arr, deriv_arr, color='#2B9E8F', linewidth=2.0)
    ax_deriv.fill_between(rho_arr, deriv_arr, alpha=0.12, color='#2B9E8F')
    ax_deriv.plot(rho, deriv_arr[rho_hl_idx], 'o', color='#ff4444', markersize=7, zorder=5)
    ax_deriv.annotate(
        rf"$\frac{{dA}}{{d\rho}}$ = {deriv_arr[rho_hl_idx]:.2f}",
        xy=(rho, deriv_arr[rho_hl_idx]),
        xytext=(8, 6), textcoords='offset points',
        fontsize=fs(8), color='#ff4444'
    )

    peak_idx = int(np.argmax(deriv_arr))
    ax_deriv.plot(rho_arr[peak_idx], deriv_arr[peak_idx], '^',
                  color='#ffd56b', markersize=8, zorder=5,
                  label=rf'peak  $\rho$ = {rho_arr[peak_idx]:.3f}')
    ax_deriv.legend(fontsize=fs(8), loc='upper right', framealpha=0.5)
    ax_deriv.set_xlabel(r'$\rho$', fontsize=fs(10))
    ax_deriv.set_ylabel(r'$\frac{dA}{d\rho}$', fontsize=fs(9))
    ax_deriv.set_title(r'$\frac{dA}{d\rho}$ — Tốc độ mở rộng ROA', fontsize=fs(11))
    ax_deriv.tick_params(labelsize=fs(8))
    ax_deriv.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Lưu ảnh: {save_path}")

    return fig, (ax_phase, ax_area, ax_deriv)

def main():
    device = torch.device("cpu")
    print("Đang tải Lyapunov model...")
    lyapunov_nn, _, _, dynamics, pendulum_continuous = load_lyapunov_and_controller(device, load_controllers=False)
    
    # Tham số vẽ
    lower_limit = torch.tensor([-np.pi, -3.0], device=device)
    upper_limit = torch.tensor([np.pi, 3.0], device=device)
    
    # Vùng quan tâm ROA (thu hẹp hơn)
    roa_lower = torch.tensor([-0.4, -1.0], device=device)
    roa_upper = torch.tensor([0.4, 1.0], device=device)
    
    rho = 0.1  # Level set value
    
    print("Đang vẽ biểu đồ...")
    fig, ax = plot_phase_portrait_heatmap(
        lyapunov_nn=lyapunov_nn,
        dynamics=dynamics,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        rho=rho,
        roa_limits=(roa_lower, roa_upper),
        grid_size=400,
        device=device,
        save_path=str(PROJECT_ROOT / "plots" / "phase_portrait_heatmap.png"),
        dpi=180
    )
    
    plt.show()


if __name__ == "__main__":
    main()
