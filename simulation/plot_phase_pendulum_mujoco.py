"""
Phase portrait for the MuJoCo branch.

This script is intentionally separate from plot_phase_pendulum.py because:
- the MuJoCo branch uses a surrogate dynamics model,
- the Lyapunov checkpoint is trained for that branch,
- and the phase portrait should be evaluated on the MuJoCo state slice with u=0.

The plot keeps the same overall layout:
- Lyapunov heatmap
- rho contour
- vector field for u = 0
- sample trajectories
- ROA area curve and dA/drho curve
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.pendulum.mujoco_feedback import MuJoCoSurrogateSecondOrderDiscreteTimeSystem
from neural_lyapunov_training.controllers import NeuralNetworkController, NeuralNetworkLuenbergerObserver
from neural_lyapunov_training.lyapunov import NeuralNetworkQuadraticLyapunov
from neural_lyapunov_training.pendulum import PendulumDynamics

FONT_SCALE = 0.9


def fs(size):
    return max(6, size * FONT_SCALE)


def load_mujoco_models(device):
    """Load the MuJoCo surrogate dynamics and its trained Lyapunov/controller/observer checkpoint."""
    surrogate_path = PROJECT_ROOT / "models" / "mujoco_surrogate" / "pendulum_mujoco_surrogate_dynamics.pth"
    checkpoint_path = PROJECT_ROOT / "models" / "mujoco_feedback" / "pendulum_mujoco_feedback.pth"

    if not surrogate_path.exists():
        raise FileNotFoundError(
            f"MuJoCo surrogate checkpoint not found: {surrogate_path}. Run train_surrogate first."
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"MuJoCo feedback checkpoint not found: {checkpoint_path}. Run mujoco_feedback.py all first."
        )

    pendulum_continuous = PendulumDynamics(m=0.15, l=0.5, beta=0.1)
    dynamics = MuJoCoSurrogateSecondOrderDiscreteTimeSystem(
        continuous_time_system=pendulum_continuous,
        dt=0.01,
        surrogate_path=str(surrogate_path),
        predict_delta=True,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "lyapunov_state_dict" not in ckpt:
        raise RuntimeError(
            f"Checkpoint format invalid: {checkpoint_path}. Missing lyapunov_state_dict."
        )

    if "controller_state_dict" not in ckpt or "observer_state_dict" not in ckpt:
        raise RuntimeError(
            f"Checkpoint format invalid: {checkpoint_path}. Missing controller_state_dict or observer_state_dict."
        )

    controller = NeuralNetworkController(
        nlayer=4,
        in_dim=3,
        out_dim=1,
        hidden_dim=8,
        clip_output="clamp",
        u_lo=torch.tensor([-1.0]),
        u_up=torch.tensor([1.0]),
        x_equilibrium=torch.zeros(3, dtype=torch.float32),
        u_equilibrium=torch.zeros(1, dtype=torch.float32),
    ).to(device)
    controller.load_state_dict(ckpt["controller_state_dict"])
    controller.eval()

    observer = NeuralNetworkLuenbergerObserver(
        z_dim=2,
        y_dim=1,
        dynamics=dynamics,
        h=lambda x: pendulum_continuous.h(x),
        zero_obs_error=torch.zeros(1, 1),
        fc_hidden_dim=[8, 8],
    ).to(device)
    observer.load_state_dict(ckpt["observer_state_dict"])
    observer.eval()

    lyapunov_nn = NeuralNetworkQuadraticLyapunov(
        goal_state=torch.zeros(4, dtype=torch.float32),
        x_dim=4,
        R_rows=4,
        eps=0.01,
        R=torch.eye(4, dtype=torch.float32),
    ).to(device)
    lyapunov_nn.load_state_dict(ckpt["lyapunov_state_dict"])
    lyapunov_nn.eval()

    rho = float(ckpt.get("rho", 0.1))
    return lyapunov_nn, controller, observer, dynamics, pendulum_continuous, rho


def simulate_trajectory(x0_list, dynamics, max_steps=120, device=torch.device("cpu")):
    trajectories = []
    with torch.no_grad():
        for x0 in x0_list:
            traj = [x0.clone()]
            for _ in range(max_steps):
                x = traj[-1].unsqueeze(0)
                u = torch.zeros(1, 1, device=device)
                x_next = dynamics(x, u).squeeze(0)
                traj.append(x_next.cpu())
            trajectories.append(torch.stack(traj))
    return trajectories


def simulate_closed_loop(x0_list, controller, observer, dynamics, max_steps=200, device=torch.device("cpu")):
    """Simulate closed-loop rollouts and record u(t)."""
    x_traj_list = []
    u_traj_list = []
    with torch.no_grad():
        for x0 in x0_list:
            x = x0.clone().to(device)
            z = x.clone()
            x_hist = [x.squeeze(0).cpu()]
            u_hist = []
            for _ in range(max_steps):
                y = x[:, :1]
                nn_input = torch.cat([z, y - z[:, :1]], dim=1)
                u = controller(nn_input)
                x = dynamics(x, u)
                z = observer(z, u, y)
                x_hist.append(x.squeeze(0).cpu())
                u_hist.append(u.squeeze(0).cpu())
            x_traj_list.append(torch.stack(x_hist))
            u_traj_list.append(torch.stack(u_hist).squeeze(-1))
    return x_traj_list, u_traj_list


def plot_phase_portrait_heatmap(
    lyapunov_nn,
    dynamics,
    lower_limit=None,
    upper_limit=None,
    rho=0.1,
    roa_limits=None,
    grid_size=400,
    device=torch.device("cpu"),
    save_path=None,
    dpi=180,
):
    if lower_limit is None:
        lower_limit = torch.tensor([-np.pi, -3.0], device=device)
    if upper_limit is None:
        upper_limit = torch.tensor([np.pi, 3.0], device=device)
    if roa_limits is None:
        roa_limits = (lower_limit, upper_limit)

    lower_limit = lower_limit.to(device)
    upper_limit = upper_limit.to(device)
    roa_lower, roa_upper = roa_limits
    roa_lower = roa_lower.to(device)
    roa_upper = roa_upper.to(device)

    xl = lower_limit[0].item()
    xu = upper_limit[0].item()
    yl = lower_limit[1].item()
    yu = upper_limit[1].item()

    x_ticks = torch.linspace(lower_limit[0], upper_limit[0], grid_size, device=device)
    y_ticks = torch.linspace(lower_limit[1], upper_limit[1], grid_size, device=device)
    grid_x, grid_y = torch.meshgrid(x_ticks, y_ticks, indexing="ij")

    # Embed the 2D physical state into the 4D Lyapunov state slice with zero observer error.
    X_flat = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    X_aug = torch.cat([X_flat, torch.zeros((X_flat.shape[0], 2), device=device)], dim=1)
    with torch.no_grad():
        V_flat = lyapunov_nn(X_aug).squeeze()
    V_grid = V_flat.reshape(grid_size, grid_size).cpu().numpy()
    grid_x_np = grid_x.cpu().numpy()
    grid_y_np = grid_y.cpu().numpy()

    N_RHO = 60
    rho_max = float(V_grid.max()) * 0.85
    rho_arr = np.linspace(rho_max / N_RHO, rho_max, N_RHO)
    cell_area = (xu - xl) * (yu - yl) / (grid_size**2)
    area_arr = np.array([(V_grid <= r).sum() * cell_area for r in rho_arr])
    deriv_arr = np.gradient(area_arr, rho_arr[1] - rho_arr[0])

    fig = plt.figure(figsize=(16, 9), dpi=dpi, layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], height_ratios=[1, 1])

    ax_phase = fig.add_subplot(gs[:, 0])
    ax_area = fig.add_subplot(gs[0, 1])
    ax_deriv = fig.add_subplot(gs[1, 1])

    cf = ax_phase.contourf(grid_x_np, grid_y_np, V_grid, levels=60, cmap="viridis", alpha=0.82)
    divider = make_axes_locatable(ax_phase)
    cax1 = divider.append_axes("right", size="4%", pad=0.15)
    cbar = fig.colorbar(cf, cax=cax1)
    cbar.set_label(r"$V(x)$", fontsize=fs(10))
    cbar.ax.tick_params(labelsize=fs(8))

    ax_phase.contour(grid_x_np, grid_y_np, V_grid, levels=14, colors="white", linewidths=0.35, alpha=0.18)

    N_LEVELS = 10
    rho_levels = np.linspace(rho_max / N_LEVELS, rho_max, N_LEVELS)
    cmap_exp = plt.cm.plasma
    for idx, r in enumerate(rho_levels):
        t = idx / (N_LEVELS - 1)
        color = cmap_exp(t)
        lw = 0.9 + t * 1.2
        alpha = 0.55 + t * 0.35
        ax_phase.contour(grid_x_np, grid_y_np, V_grid, levels=[r], colors=[color], linewidths=lw, alpha=alpha)

    sm = plt.cm.ScalarMappable(cmap=cmap_exp, norm=plt.Normalize(vmin=rho_levels[0], vmax=rho_levels[-1]))
    sm.set_array([])
    cax2 = divider.append_axes("right", size="4%", pad=0.55)
    cbar2 = fig.colorbar(sm, cax=cax2)
    cbar2.set_label(r"$\rho$ (expansion)", fontsize=fs(10))
    cbar2.ax.tick_params(labelsize=fs(8))

    cs = ax_phase.contour(grid_x_np, grid_y_np, V_grid, levels=[rho], colors="#ff4444", linewidths=2.8, linestyles="--")
    ax_phase.clabel(cs, fmt=rf'$\rho$ = {rho:.3f}', fontsize=fs(9), inline=True, inline_spacing=4, colors="#ff4444")

    has_roa_box = not torch.allclose(roa_lower, lower_limit) or not torch.allclose(roa_upper, upper_limit)
    if has_roa_box:
        rx = roa_lower[0].item()
        rw = (roa_upper[0] - roa_lower[0]).item()
        ry = roa_lower[1].item()
        rh = (roa_upper[1] - roa_lower[1]).item()
        ax_phase.add_patch(
            plt.Rectangle((rx, ry), rw, rh, linewidth=2, edgecolor="#ffaa00", facecolor="#ffaa0015", linestyle=":", zorder=2)
        )

    n_arr = 16
    X_arr = torch.tensor(
        [[xv, yv] for xv in np.linspace(xl, xu, n_arr) for yv in np.linspace(yl, yu, n_arr)],
        dtype=torch.float32,
        device=device,
    )
    u_zero = torch.zeros(len(X_arr), 1, device=device)
    with torch.no_grad():
        X_next = dynamics(X_arr, u_zero)
    dX = (X_next - X_arr).cpu().numpy()
    mag = np.hypot(dX[:, 0], dX[:, 1]) + 1e-8
    pts = X_arr.cpu().numpy()
    ax_phase.quiver(
        pts[:, 0],
        pts[:, 1],
        dX[:, 0] / mag * (xu - xl) / n_arr * 0.42,
        dX[:, 1] / mag * (yu - yl) / n_arr * 0.42,
        color="white",
        alpha=0.45,
        width=0.003,
        headwidth=4,
        headlength=4,
        headaxislength=3.5,
        scale=1,
        scale_units="xy",
        angles="xy",
        zorder=4,
    )

    init_points = [
        torch.tensor([th, dth], dtype=torch.float32, device=device)
        for th in np.linspace(xl * 0.9, xu * 0.9, 6)
        for dth in np.linspace(yl * 0.7, yu * 0.7, 4)
    ]
    for traj in simulate_trajectory(init_points, dynamics, max_steps=120, device=device):
        traj_np = traj.numpy()
        ax_phase.plot(traj_np[:, 0], traj_np[:, 1], color="#00ddcc", linewidth=0.9, alpha=0.40, zorder=5)
        ax_phase.plot(traj_np[0, 0], traj_np[0, 1], "o", color="#00ddcc", markersize=3, alpha=0.60, zorder=6)

    ax_phase.plot(0, 0, "*", color="#ff3333", markersize=13, zorder=7, markeredgecolor="white", markeredgewidth=0.7)

    leg_elems = [
        Line2D([0], [0], color="#ff4444", lw=2.2, linestyle="--", label=rf'$\rho$ = {rho:.3f} (selected)'),
        Line2D([0], [0], color="#00ddcc", lw=1.4, alpha=0.6, label="Trajectories (u = 0)"),
        Line2D([0], [0], color="white", lw=0, marker="*", markersize=10, markerfacecolor="#ff3333", label="Equilibrium"),
    ]
    if has_roa_box:
        leg_elems.append(
            Patch(facecolor="#ffaa0015", edgecolor="#ffaa00", linestyle=":", linewidth=1.5, label="ROA interest region")
        )

    ax_phase.legend(handles=leg_elems, loc="upper right", fontsize=fs(8), framealpha=0.55, edgecolor="white")
    ax_phase.set_xlim(xl, xu)
    ax_phase.set_ylim(yl, yu)
    ax_phase.set_xlabel(r"$\theta$ (rad)", fontsize=fs(11))
    ax_phase.set_ylabel(r"$\dot{\theta}$ (rad/s)", fontsize=fs(11))
    ax_phase.set_title(rf"MuJoCo Phase Portrait — Lyapunov Heatmap + ROA Expansion  ($\rho$ = {rho:.4f})", fontsize=fs(12), pad=8)
    ax_phase.tick_params(labelsize=fs(9))
    ax_phase.grid(True, color="white", alpha=0.10, linewidth=0.5)

    ax_area.plot(rho_arr, area_arr, color="#534AB7", linewidth=2.0)
    ax_area.fill_between(rho_arr, area_arr, alpha=0.12, color="#534AB7")
    rho_hl_idx = int(np.argmin(np.abs(rho_arr - rho)))
    ax_area.axvline(rho, color="#ff4444", linewidth=1.5, linestyle="--", alpha=0.8)
    ax_area.plot(rho, area_arr[rho_hl_idx], "o", color="#ff4444", markersize=7, zorder=5)
    ax_area.annotate(rf'A = {area_arr[rho_hl_idx]:.3f}', xy=(rho, area_arr[rho_hl_idx]), xytext=(8, -14), textcoords="offset points", fontsize=fs(8), color="#ff4444")
    ax_area.set_xlabel(r"$\rho$", fontsize=fs(10))
    ax_area.set_ylabel(r"Area ($rad \times rad/s$)", fontsize=fs(9))
    ax_area.set_title(r"$A(\rho)$ — ROA area vs $\rho$", fontsize=fs(11))
    ax_area.tick_params(labelsize=fs(8))
    ax_area.grid(True, alpha=0.3)

    ax_deriv.plot(rho_arr, deriv_arr, color="#2B9E8F", linewidth=2.0)
    ax_deriv.fill_between(rho_arr, deriv_arr, alpha=0.12, color="#2B9E8F")
    ax_deriv.plot(rho, deriv_arr[rho_hl_idx], "o", color="#ff4444", markersize=7, zorder=5)
    ax_deriv.annotate(rf"$\frac{{dA}}{{d\rho}}$ = {deriv_arr[rho_hl_idx]:.2f}", xy=(rho, deriv_arr[rho_hl_idx]), xytext=(8, 6), textcoords="offset points", fontsize=fs(8), color="#ff4444")
    peak_idx = int(np.argmax(deriv_arr))
    ax_deriv.plot(
        rho_arr[peak_idx],
        deriv_arr[peak_idx],
        "^",
        color="#ffd56b",
        markersize=8,
        zorder=5,
        label=rf"peak $\rho$ = {rho_arr[peak_idx]:.3f}",
    )
    ax_deriv.legend(fontsize=fs(8), loc="upper right", framealpha=0.5)
    ax_deriv.set_xlabel(r"$\rho$", fontsize=fs(10))
    ax_deriv.set_ylabel(r"$\frac{dA}{d\rho}$", fontsize=fs(9))
    ax_deriv.set_title(r"$\frac{dA}{d\rho}$ — expansion speed", fontsize=fs(11))
    ax_deriv.tick_params(labelsize=fs(8))
    ax_deriv.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure: {save_path}")

    return fig, (ax_phase, ax_area, ax_deriv)


def plot_control_effort(
    u_traj_list,
    rho,
    save_path=None,
    dpi=180,
):
    """Plot closed-loop control effort statistics."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    max_len = max(len(u_traj) for u_traj in u_traj_list)
    padded = np.full((len(u_traj_list), max_len), np.nan, dtype=np.float32)
    for i, u_traj in enumerate(u_traj_list):
        u_np = u_traj.cpu().numpy().reshape(-1)
        padded[i, : len(u_np)] = u_np
        ax.plot(np.arange(len(u_np)) * 0.01, u_np, color="#2B9E8F", alpha=0.20, linewidth=0.9)

    time_axis = np.arange(max_len) * 0.01
    u_mean = np.nanmean(padded, axis=0)
    u_median = np.nanmedian(padded, axis=0)
    ax.plot(time_axis, u_mean, color="tab:blue", linewidth=2.0, label="mean u (all)")
    ax.plot(time_axis, u_median, color="tab:orange", linewidth=2.0, linestyle="--", label="median u (all)")

    ax.axhline(1.0, color="tab:red", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.axhline(-1.0, color="tab:red", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("u")
    ax.set_title("MuJoCo control effort trajectories (closed-loop)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    ax.text(
        0.01,
        0.98,
        f"trajectories={len(u_traj_list)}\nrho={rho:.4g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure: {save_path}")

    return fig, ax


def main():
    device = torch.device("cpu")
    print("Loading MuJoCo models...")
    lyapunov_nn, controller, observer, dynamics, _, rho = load_mujoco_models(device)

    lower_limit = torch.tensor([-np.pi, -3.0], device=device)
    upper_limit = torch.tensor([np.pi, 3.0], device=device)
    roa_lower = torch.tensor([-0.4, -1.0], device=device)
    roa_upper = torch.tensor([0.4, 1.0], device=device)

    print("Rendering MuJoCo phase portrait...")
    fig, _ = plot_phase_portrait_heatmap(
        lyapunov_nn=lyapunov_nn,
        dynamics=dynamics,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        rho=rho,
        roa_limits=(roa_lower, roa_upper),
        grid_size=400,
        device=device,
        save_path=str(PROJECT_ROOT / "plots" / "phase_portrait_heatmap_mujoco.png"),
        dpi=180,
    )

    initial_states = [
        torch.tensor([[th, dth]], dtype=torch.float32, device=device)
        for th in np.linspace(-0.35, 0.35, 6)
        for dth in np.linspace(-0.6, 0.6, 4)
    ]
    _, u_traj_list = simulate_closed_loop(
        initial_states,
        controller=controller,
        observer=observer,
        dynamics=dynamics,
        max_steps=200,
        device=device,
    )
    plot_control_effort(
        u_traj_list,
        rho=rho,
        save_path=str(PROJECT_ROOT / "plots" / "mujoco_control_effort.png"),
        dpi=180,
    )
    plt.show()


if __name__ == "__main__":
    main()
