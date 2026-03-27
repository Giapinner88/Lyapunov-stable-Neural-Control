import argparse
from pathlib import Path
import sys
from matplotlib.lines import Line2D
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.dynamics import CartpoleDynamics
from core.roa_utils import compute_rho_boundary
from core.runtime_utils import box_tensors, choose_device, load_trained_system


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detailed CartPole phase portrait on theta-theta_dot slice")
    parser.add_argument("--controller", type=str, default="checkpoints/cartpole/cartpole_controller.pth")
    parser.add_argument("--lyapunov", type=str, default="checkpoints/cartpole/cartpole_lyapunov.pth")
    parser.add_argument("--train-log", type=str, default=None, help="Training log used to extract rho from actual run")
    parser.add_argument("--output", type=str, default="reports/cartpole_phase_portrait_detailed.png")
    parser.add_argument("--theta-min", type=float, default=-2.2)
    parser.add_argument("--theta-max", type=float, default=2.2)
    parser.add_argument("--thetadot-min", type=float, default=-3.5)
    parser.add_argument("--thetadot-max", type=float, default=3.5)
    parser.add_argument("--grid", type=int, default=180)
    parser.add_argument("--x-fixed", type=float, default=0.0)
    parser.add_argument("--xdot-fixed", type=float, default=0.0)
    parser.add_argument("--alpha-lyap", type=float, default=0.05)
    parser.add_argument(
        "--rho-multipliers",
        type=str,
        default="0.25,0.5,0.75,1.0,1.25,1.5,2.0",
        help="Comma-separated multipliers for rho contour ladder",
    )
    parser.add_argument(
        "--rho-mode",
        type=str,
        default="history",
        choices=["history", "base-multipliers"],
        help="history: use logged rho values by epoch; base-multipliers: use base rho from log times multipliers",
    )
    parser.add_argument("--max-rho-levels", type=int, default=10, help="Max rho contours to draw from history mode")
    parser.add_argument("--traj-count", type=int, default=8)
    parser.add_argument("--traj-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def parse_rho_multipliers(raw: str) -> list[float]:
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    vals = sorted(set(v for v in vals if v > 0.0))
    if not vals:
        vals = [1.0]
    return vals


def find_latest_train_log(repo_root: Path) -> Path | None:
    candidates = sorted((repo_root / "reports").glob("cartpole_*.log"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_rho_history_from_log(log_path: Path) -> list[tuple[int, float]]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\n", " ")
    pattern = re.compile(
        r"CEGIS Epoch\s+(?P<epoch>\d+)\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*(?:rho|ρ)=(?P<rho>[-+0-9.eE]+)",
        flags=re.IGNORECASE,
    )
    rows = []
    by_epoch: dict[int, float] = {}
    for m in pattern.finditer(text):
        ep = int(m.group("epoch"))
        rho = float(m.group("rho"))
        by_epoch[ep] = rho
    for ep in sorted(by_epoch):
        if by_epoch[ep] > 0.0:
            rows.append((ep, by_epoch[ep]))
    return rows


def downsample_history(history: list[tuple[int, float]], max_levels: int) -> list[tuple[int, float]]:
    if len(history) <= max_levels:
        return history
    # Keep temporal coverage by uniform indexing.
    idx = np.linspace(0, len(history) - 1, max_levels).round().astype(int)
    unique_idx = sorted(set(idx.tolist()))
    return [history[i] for i in unique_idx]


def build_slice_grid(args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    theta_vals = np.linspace(args.theta_min, args.theta_max, args.grid)
    theta_dot_vals = np.linspace(args.thetadot_min, args.thetadot_max, args.grid)
    theta_mesh, theta_dot_mesh = np.meshgrid(theta_vals, theta_dot_vals)

    x = torch.zeros((args.grid * args.grid, 4), dtype=torch.float32, device=device)
    x[:, 0] = args.x_fixed
    x[:, 1] = args.xdot_fixed
    x[:, 2] = torch.as_tensor(theta_mesh.reshape(-1), device=device)
    x[:, 3] = torch.as_tensor(theta_dot_mesh.reshape(-1), device=device)

    return theta_mesh, theta_dot_mesh, x


def simulate_trajectory(
    dynamics: CartpoleDynamics,
    controller: torch.nn.Module,
    x0: torch.Tensor,
    steps: int,
) -> np.ndarray:
    xs = [x0.detach().cpu().numpy().copy()]
    x = x0.view(1, -1)
    with torch.no_grad():
        for _ in range(steps):
            u = controller(x)
            x = dynamics.step(x, u)
            xs.append(x[0].detach().cpu().numpy().copy())
    return np.asarray(xs)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    bundle = load_trained_system(
        args.controller,
        args.lyapunov,
        system_name="cartpole",
        device=device,
    )
    dynamics = bundle.dynamics
    controller = bundle.controller
    lyapunov = bundle.lyapunov
    config = bundle.config
    rho_multipliers = parse_rho_multipliers(args.rho_multipliers)

    repo_root = Path(__file__).resolve().parent
    log_path = Path(args.train_log) if args.train_log else find_latest_train_log(repo_root)
    rho_history: list[tuple[int, float]] = []
    if log_path is not None and log_path.exists():
        rho_history = parse_rho_history_from_log(log_path)

    theta_mesh, theta_dot_mesh, x_slice = build_slice_grid(args, device)

    with torch.no_grad():
        # Lyapunov landscape on slice.
        v = lyapunov(x_slice).squeeze(1)

        # Closed-loop vector field on slice (projected to theta, theta_dot).
        u = controller(x_slice)
        xdot = dynamics.continuous_dynamics(x_slice, u)
        dtheta = xdot[:, 2].detach().cpu().numpy().reshape(theta_mesh.shape)
        dtheta_dot = xdot[:, 3].detach().cpu().numpy().reshape(theta_mesh.shape)

        # One-step Lyapunov violation map.
        x_next = dynamics.step(x_slice, u)
        v_next = lyapunov(x_next).squeeze(1)
        violation = (v_next - (1.0 - args.alpha_lyap) * v).detach().cpu().numpy().reshape(theta_mesh.shape)

    v_map = v.detach().cpu().numpy().reshape(theta_mesh.shape)

    x_min, x_max = box_tensors(config, device=device, dtype=torch.float32)
    if rho_history:
        rho_base = rho_history[-1][1]
        if args.rho_mode == "history":
            sampled = downsample_history(rho_history, max_levels=max(1, args.max_rho_levels))
            rho_levels = [r for _, r in sampled]
            rho_labels = [f"ep{ep:03d}: {r:.5f}" for ep, r in sampled]
        else:
            rho_levels = [rho_base * m for m in rho_multipliers]
            rho_labels = [f"rho x {m:.2f}" for m in rho_multipliers]
        rho_source = f"log:{log_path.name}"
    else:
        # Fallback only when no usable logged rho found.
        rho_base, _ = compute_rho_boundary(
            lyapunov,
            dynamics,
            controller,
            x_min,
            x_max,
            n_boundary_samples=700,
            n_pgd_steps=35,
            gamma=0.9,
            device=device,
            verbose=False,
        )
        rho_levels = [rho_base * m for m in rho_multipliers]
        rho_labels = [f"rho x {m:.2f}" for m in rho_multipliers]
        rho_source = "computed-fallback"

    # Build trajectories from random in-slice initial states.
    trajs = []
    for _ in range(args.traj_count):
        th0 = np.random.uniform(args.theta_min, args.theta_max)
        thd0 = np.random.uniform(args.thetadot_min, args.thetadot_max)
        x0 = torch.tensor([args.x_fixed, args.xdot_fixed, th0, thd0], dtype=torch.float32, device=device)
        trajs.append(simulate_trajectory(dynamics, controller, x0, args.traj_steps))

    fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)

    # Left: phase portrait like paper/evaluate style.
    ax = axes[0]
    cf = ax.contourf(theta_mesh, theta_dot_mesh, v_map, levels=30, cmap="viridis", alpha=0.85)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Lyapunov V(x)")

    ax.contour(theta_mesh, theta_dot_mesh, v_map, levels=12, colors="white", linewidths=0.6, alpha=0.9)
    ax.streamplot(theta_mesh, theta_dot_mesh, dtheta, dtheta_dot, color="black", density=1.3, linewidth=0.8, arrowsize=1.2)

    # ROA ladder from full-state rho projected on this slice.
    ladder_colors = plt.cm.plasma(np.linspace(0.15, 0.95, len(rho_levels)))
    for lvl, col in zip(rho_levels, ladder_colors):
        ax.contour(theta_mesh, theta_dot_mesh, v_map, levels=[lvl], colors=[col], linewidths=1.8)

    for t in trajs:
        ax.plot(t[:, 2], t[:, 3], color="tomato", alpha=0.7, linewidth=1.2)

    ax.plot(0.0, 0.0, marker="*", color="red", markersize=15, label="Equilibrium (theta=0, theta_dot=0)")

    # Build compact legend entries for rho ladder.
    legend_handles = [
        Line2D([0], [0], color=ladder_colors[i], lw=2, label=rho_labels[i])
        for i in range(len(rho_levels))
    ]
    legend_handles.append(Line2D([0], [0], color="tomato", lw=1.5, label="sample trajectories"))
    legend_handles.append(Line2D([0], [0], color="black", lw=1.0, label="closed-loop vector field"))
    ax.set_title("CartPole Phase Portrait (theta-theta_dot slice)")
    ax.set_xlabel("theta (rad)")
    ax.set_ylabel("theta_dot (rad/s)")
    ax.set_xlim(args.theta_min, args.theta_max)
    ax.set_ylim(args.thetadot_min, args.thetadot_max)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    # Right: Lyapunov decrease violation heatmap.
    ax2 = axes[1]
    vmax = float(np.percentile(np.abs(violation), 97))
    im = ax2.contourf(theta_mesh, theta_dot_mesh, violation, levels=35, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    cbar2 = fig.colorbar(im, ax=ax2)
    cbar2.set_label("Violation: V(x+) - (1-alpha)V(x)")

    ax2.contour(theta_mesh, theta_dot_mesh, violation, levels=[0.0], colors="black", linewidths=1.8)
    for lvl, col in zip(rho_levels, ladder_colors):
        ax2.contour(theta_mesh, theta_dot_mesh, v_map, levels=[lvl], colors=[col], linewidths=1.5, alpha=0.95)
    ax2.set_title("Lyapunov Decrease Violation Map")
    ax2.set_xlabel("theta (rad)")
    ax2.set_ylabel("theta_dot (rad/s)")
    ax2.set_xlim(args.theta_min, args.theta_max)
    ax2.set_ylim(args.thetadot_min, args.thetadot_max)
    ax2.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(
        f"CartPole Detailed Diagnostics | x={args.x_fixed:.2f}, x_dot={args.xdot_fixed:.2f}, "
        f"rho_base={rho_base:.5f}, alpha={args.alpha_lyap}, rho_source={rho_source}",
        fontsize=13,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print("[PhasePortrait] Done")
    print(f"- output: {out_path}")
    print(f"- rho source: {rho_source}")


if __name__ == "__main__":
    main()
