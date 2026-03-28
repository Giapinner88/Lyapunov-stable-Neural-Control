from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy_2d(grid: torch.Tensor, n_x: int, n_y: int) -> np.ndarray:
    return grid.detach().cpu().reshape(n_x, n_y).numpy()


def _compute_grid_fields(
    derivative_lyaloss,
    x_grid: torch.Tensor,
    n_x: int,
    n_y: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        u = derivative_lyaloss.controller(x_grid)
        x_next = derivative_lyaloss.dynamics.forward(x_grid, u)
        V = derivative_lyaloss.lyapunov(x_grid)
        V_next = derivative_lyaloss.lyapunov(x_next)

    dt = getattr(derivative_lyaloss.dynamics, "dt", 1.0)
    if dt is None or dt == 0:
        dt = 1.0

    dV = V_next - V
    dV_dt = dV / float(dt)
    flow = (x_next - x_grid) / float(dt)

    V_2d = _to_numpy_2d(V.squeeze(1), n_x, n_y)
    dV_dt_2d = _to_numpy_2d(dV_dt.squeeze(1), n_x, n_y)
    u_2d = _to_numpy_2d(u.squeeze(1), n_x, n_y)
    flow_0 = _to_numpy_2d(flow[:, 0], n_x, n_y)
    flow_1 = _to_numpy_2d(flow[:, 1], n_x, n_y)
    return V_2d, dV_dt_2d, u_2d, flow_0, flow_1


def _simulate_trajectories(
    derivative_lyaloss,
    x0: torch.Tensor,
    steps: int,
) -> np.ndarray:
    x = x0
    traj = [x.detach().cpu().numpy()]
    with torch.no_grad():
        for _ in range(steps):
            u = derivative_lyaloss.controller(x)
            x = derivative_lyaloss.dynamics.forward(x, u)
            traj.append(x.detach().cpu().numpy())
    return np.stack(traj, axis=0)


def _resolve_rho_values(derivative_lyaloss, rho_values: Iterable[float] | None) -> list[float]:
    if rho_values is not None:
        return [float(v) for v in rho_values]
    if hasattr(derivative_lyaloss, "get_rho") and derivative_lyaloss.x_boundary is not None:
        return [float(derivative_lyaloss.get_rho().item())]
    return []


def plot_pendulum_diagnostics(
    derivative_lyaloss,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    out_dir: str | Path,
    rho_values: Iterable[float] | None = None,
    grid_points: int = 140,
    traj_count: int = 24,
    traj_steps: int = 220,
    filename: str = "pendulum_diagnostics.png",
) -> Path:
    """Render one comprehensive pendulum diagnostics figure.

    The chart includes:
    1) 3D surface of V(x)
    2) 3D surface of dV/dt
    3) Heatmap of V with ROA contours for every rho
    4) Heatmap of dV/dt with ROA contours and dV/dt=0 contour
    5) Closed-loop phase portrait with controller-aware trajectories
    6) Heatmap of control action u(x)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = lower_limit.device
    x0_ticks = torch.linspace(lower_limit[0], upper_limit[0], grid_points, device=device)
    x1_ticks = torch.linspace(lower_limit[1], upper_limit[1], grid_points, device=device)
    grid_x0, grid_x1 = torch.meshgrid(x0_ticks, x1_ticks, indexing="ij")

    x_grid = torch.stack((grid_x0.flatten(), grid_x1.flatten()), dim=1)
    V, dV_dt, u, flow_0, flow_1 = _compute_grid_fields(
        derivative_lyaloss, x_grid, grid_points, grid_points
    )
    x0_np = grid_x0.detach().cpu().numpy()
    x1_np = grid_x1.detach().cpu().numpy()

    rho_list = _resolve_rho_values(derivative_lyaloss, rho_values)

    fig = plt.figure(figsize=(24, 13), constrained_layout=True)
    axes = fig.subplot_mosaic(
        [["V3D", "dV3D", "VHM"], ["dVHM", "PHASE", "CTRL"]],
        per_subplot_kw={"V3D": {"projection": "3d"}, "dV3D": {"projection": "3d"}},
    )

    ax = axes["V3D"]
    ax.plot_surface(x0_np, x1_np, V, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_title("Pendulum: V(x)")
    ax.set_xlabel("theta")
    ax.set_ylabel("theta_dot")
    ax.set_zlabel("V")

    ax = axes["dV3D"]
    ax.plot_surface(x0_np, x1_np, dV_dt, cmap="coolwarm", linewidth=0, antialiased=True)
    ax.set_title("Pendulum: dV/dt")
    ax.set_xlabel("theta")
    ax.set_ylabel("theta_dot")
    ax.set_zlabel("dV/dt")

    ax = axes["VHM"]
    im_v = ax.pcolormesh(x0_np, x1_np, V, shading="auto", cmap="viridis")
    for rho in rho_list:
        ax.contour(x0_np, x1_np, V, levels=[rho], colors="red", linewidths=2.0)
    ax.set_title("V heatmap + ROA contours")
    ax.set_xlabel("theta")
    ax.set_ylabel("theta_dot")
    fig.colorbar(im_v, ax=ax, fraction=0.047)

    ax = axes["dVHM"]
    im_dv = ax.pcolormesh(x0_np, x1_np, dV_dt, shading="auto", cmap="coolwarm")
    ax.contour(x0_np, x1_np, dV_dt, levels=[0.0], colors="black", linewidths=1.3)
    for rho in rho_list:
        ax.contour(x0_np, x1_np, V, levels=[rho], colors="gold", linewidths=1.6)
    ax.set_title("dV/dt heatmap + dV/dt=0 + ROA")
    ax.set_xlabel("theta")
    ax.set_ylabel("theta_dot")
    fig.colorbar(im_dv, ax=ax, fraction=0.047)

    ax = axes["PHASE"]
    step = max(1, grid_points // 28)
    x0_1d = x0_ticks.detach().cpu().numpy()[::step]
    x1_1d = x1_ticks.detach().cpu().numpy()[::step]
    flow_0_stream = flow_0[::step, ::step].T
    flow_1_stream = flow_1[::step, ::step].T
    ax.streamplot(
        x0_1d,
        x1_1d,
        flow_0_stream,
        flow_1_stream,
        color=np.hypot(flow_0_stream, flow_1_stream),
        cmap="plasma",
        density=1.0,
        linewidth=1.0,
        arrowsize=0.9,
    )
    x0_init = (torch.rand((traj_count, 2), device=device) - 0.5) * 2.0 * upper_limit
    traj = _simulate_trajectories(derivative_lyaloss, x0_init, steps=traj_steps)
    for i in range(traj.shape[1]):
        ax.plot(traj[:, i, 0], traj[:, i, 1], color="white", linewidth=0.9, alpha=0.7)
    for rho in rho_list:
        ax.contour(x0_np, x1_np, V, levels=[rho], colors="cyan", linewidths=2.0)
    ax.set_title("Closed-loop phase portrait + trajectories")
    ax.set_xlabel("theta")
    ax.set_ylabel("theta_dot")

    ax = axes["CTRL"]
    im_u = ax.pcolormesh(x0_np, x1_np, u, shading="auto", cmap="cividis")
    for rho in rho_list:
        ax.contour(x0_np, x1_np, V, levels=[rho], colors="magenta", linewidths=1.8)
    ax.set_title("Controller output u(x)")
    ax.set_xlabel("theta")
    ax.set_ylabel("theta_dot")
    fig.colorbar(im_u, ax=ax, fraction=0.047)

    figure_path = out_dir / filename
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return figure_path


def plot_cartpole_diagnostics(
    derivative_lyaloss,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    out_dir: str | Path,
    rho_values: Iterable[float] | None = None,
    phase_indices: Sequence[int] = (1, 3),
    grid_points: int = 120,
    traj_count: int = 20,
    traj_steps: int = 220,
    filename: str = "cartpole_diagnostics.png",
) -> Path:
    """Render one comprehensive cartpole diagnostics figure on a 2D phase slice.

    The default phase slice is (theta, theta_dot) with other states fixed to equilibrium.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx0, idx1 = int(phase_indices[0]), int(phase_indices[1])
    device = lower_limit.device
    nx = lower_limit.numel()

    s0 = torch.linspace(lower_limit[idx0], upper_limit[idx0], grid_points, device=device)
    s1 = torch.linspace(lower_limit[idx1], upper_limit[idx1], grid_points, device=device)
    grid_s0, grid_s1 = torch.meshgrid(s0, s1, indexing="ij")

    x_grid = torch.zeros((grid_points * grid_points, nx), device=device, dtype=lower_limit.dtype)
    x_grid[:, idx0] = grid_s0.flatten()
    x_grid[:, idx1] = grid_s1.flatten()

    with torch.no_grad():
        u_full = derivative_lyaloss.controller(x_grid)
        x_next_full = derivative_lyaloss.dynamics.forward(x_grid, u_full)
        V = derivative_lyaloss.lyapunov(x_grid)
        V_next = derivative_lyaloss.lyapunov(x_next_full)

    dt = getattr(derivative_lyaloss.dynamics, "dt", 1.0)
    if dt is None or dt == 0:
        dt = 1.0

    dV_dt = (V_next - V) / float(dt)
    flow_slice_0 = (x_next_full[:, idx0] - x_grid[:, idx0]) / float(dt)
    flow_slice_1 = (x_next_full[:, idx1] - x_grid[:, idx1]) / float(dt)

    V_2d = _to_numpy_2d(V.squeeze(1), grid_points, grid_points)
    dV_dt_2d = _to_numpy_2d(dV_dt.squeeze(1), grid_points, grid_points)
    u_2d = _to_numpy_2d(u_full.squeeze(1), grid_points, grid_points)
    flow_0 = _to_numpy_2d(flow_slice_0, grid_points, grid_points)
    flow_1 = _to_numpy_2d(flow_slice_1, grid_points, grid_points)

    s0_np = grid_s0.detach().cpu().numpy()
    s1_np = grid_s1.detach().cpu().numpy()

    rho_list = _resolve_rho_values(derivative_lyaloss, rho_values)

    fig = plt.figure(figsize=(24, 13), constrained_layout=True)
    axes = fig.subplot_mosaic(
        [["V3D", "dV3D", "VHM"], ["dVHM", "PHASE", "CTRL"]],
        per_subplot_kw={"V3D": {"projection": "3d"}, "dV3D": {"projection": "3d"}},
    )

    label0 = f"x[{idx0}]"
    label1 = f"x[{idx1}]"

    ax = axes["V3D"]
    ax.plot_surface(s0_np, s1_np, V_2d, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_title("Cartpole slice: V(x)")
    ax.set_xlabel(label0)
    ax.set_ylabel(label1)
    ax.set_zlabel("V")

    ax = axes["dV3D"]
    ax.plot_surface(s0_np, s1_np, dV_dt_2d, cmap="coolwarm", linewidth=0, antialiased=True)
    ax.set_title("Cartpole slice: dV/dt")
    ax.set_xlabel(label0)
    ax.set_ylabel(label1)
    ax.set_zlabel("dV/dt")

    ax = axes["VHM"]
    im_v = ax.pcolormesh(s0_np, s1_np, V_2d, shading="auto", cmap="viridis")
    for rho in rho_list:
        ax.contour(s0_np, s1_np, V_2d, levels=[rho], colors="red", linewidths=2.0)
    ax.set_title("V heatmap + ROA contours")
    ax.set_xlabel(label0)
    ax.set_ylabel(label1)
    fig.colorbar(im_v, ax=ax, fraction=0.047)

    ax = axes["dVHM"]
    im_dv = ax.pcolormesh(s0_np, s1_np, dV_dt_2d, shading="auto", cmap="coolwarm")
    ax.contour(s0_np, s1_np, dV_dt_2d, levels=[0.0], colors="black", linewidths=1.3)
    for rho in rho_list:
        ax.contour(s0_np, s1_np, V_2d, levels=[rho], colors="gold", linewidths=1.6)
    ax.set_title("dV/dt heatmap + dV/dt=0 + ROA")
    ax.set_xlabel(label0)
    ax.set_ylabel(label1)
    fig.colorbar(im_dv, ax=ax, fraction=0.047)

    ax = axes["PHASE"]
    step = max(1, grid_points // 28)
    s0_1d = s0.detach().cpu().numpy()[::step]
    s1_1d = s1.detach().cpu().numpy()[::step]
    flow_0_stream = flow_0[::step, ::step].T
    flow_1_stream = flow_1[::step, ::step].T
    ax.streamplot(
        s0_1d,
        s1_1d,
        flow_0_stream,
        flow_1_stream,
        color=np.hypot(flow_0_stream, flow_1_stream),
        cmap="plasma",
        density=1.0,
        linewidth=1.0,
        arrowsize=0.9,
    )

    x0 = torch.zeros((traj_count, nx), device=device, dtype=lower_limit.dtype)
    x0[:, idx0] = (torch.rand((traj_count,), device=device) - 0.5) * 2.0 * upper_limit[idx0]
    x0[:, idx1] = (torch.rand((traj_count,), device=device) - 0.5) * 2.0 * upper_limit[idx1]
    traj = _simulate_trajectories(derivative_lyaloss, x0, steps=traj_steps)
    for i in range(traj.shape[1]):
        ax.plot(traj[:, i, idx0], traj[:, i, idx1], color="white", linewidth=0.9, alpha=0.7)
    for rho in rho_list:
        ax.contour(s0_np, s1_np, V_2d, levels=[rho], colors="cyan", linewidths=2.0)
    ax.set_title("Closed-loop phase portrait + trajectories")
    ax.set_xlabel(label0)
    ax.set_ylabel(label1)

    ax = axes["CTRL"]
    im_u = ax.pcolormesh(s0_np, s1_np, u_2d, shading="auto", cmap="cividis")
    for rho in rho_list:
        ax.contour(s0_np, s1_np, V_2d, levels=[rho], colors="magenta", linewidths=1.8)
    ax.set_title("Controller output u(x)")
    ax.set_xlabel(label0)
    ax.set_ylabel(label1)
    fig.colorbar(im_u, ax=ax, fraction=0.047)

    figure_path = out_dir / filename
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return figure_path
