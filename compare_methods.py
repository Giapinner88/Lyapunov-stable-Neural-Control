import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from core.baselines import LQRController, QuadraticLyapunov
from core.dynamics import PendulumDynamics
from core.models import NeuralController, NeuralLyapunov
from core.verifier import SystemViolationGraph


@dataclass
class Bundle:
    name: str
    controller: torch.nn.Module
    lyapunov: torch.nn.Module
    dynamics: PendulumDynamics


def build_quadratic_bundle(device: torch.device, u_bound: float) -> Bundle:
    dynamics = PendulumDynamics().to(device)
    K, S = dynamics.get_lqr_baseline()
    controller = LQRController(K.to(device), u_bound=u_bound).to(device)
    lyapunov = QuadraticLyapunov(S.to(device)).to(device)
    controller.eval()
    lyapunov.eval()
    return Bundle("quadratic_lqr", controller, lyapunov, dynamics)


def build_neural_bundle(device: torch.device, controller_path: str, lyapunov_path: str, u_bound: float) -> Bundle:
    dynamics = PendulumDynamics().to(device)
    controller = NeuralController(nx=2, nu=1, u_bound=u_bound).to(device)
    lyapunov = NeuralLyapunov(nx=2).to(device)

    if not os.path.exists(controller_path) or not os.path.exists(lyapunov_path):
        raise FileNotFoundError(
            "Missing neural checkpoints. Expected files: "
            f"{controller_path} and {lyapunov_path}"
        )

    controller.load_state_dict(torch.load(controller_path, map_location=device))
    lyapunov.load_state_dict(torch.load(lyapunov_path, map_location=device))
    controller.eval()
    lyapunov.eval()
    return Bundle("neural", controller, lyapunov, dynamics)


def compute_violation(bundle: Bundle, x: torch.Tensor, rho: float) -> torch.Tensor:
    with torch.no_grad():
        v_t = bundle.lyapunov(x)
        u_t = bundle.controller(x)
        x_next = bundle.dynamics.step(x, u_t)
        v_next = bundle.lyapunov(x_next)
    return v_next - v_t + rho * v_t


def pointwise_stats(bundle: Bundle, eps: float, rho: float, grid_n: int, device: torch.device) -> dict:
    coords = torch.linspace(-eps, eps, grid_n, device=device)
    gx, gy = torch.meshgrid(coords, coords, indexing="ij")
    x = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

    violation = compute_violation(bundle, x, rho).squeeze(1)
    return {
        "max": float(torch.max(violation).item()),
        "min": float(torch.min(violation).item()),
        "mean": float(torch.mean(violation).item()),
        "pos_ratio": float((violation >= 0.0).float().mean().item()),
        "n_points": int(violation.numel()),
    }


def trajectory_stats(bundle: Bundle, eps: float, horizon: int, n_samples: int, device: torch.device) -> dict:
    x = (torch.rand((n_samples, 2), device=device) * 2.0 - 1.0) * eps
    x0_norm = torch.norm(x, dim=1)
    max_norm = x0_norm.clone()

    with torch.no_grad():
        for _ in range(horizon):
            u = bundle.controller(x)
            x = bundle.dynamics.step(x, u)
            max_norm = torch.maximum(max_norm, torch.norm(x, dim=1))

    final_norm = torch.norm(x, dim=1)
    converged = float((final_norm < 0.05).float().mean().item())
    diverged = float((max_norm > 2.0).float().mean().item())

    return {
        "converged_ratio": converged,
        "diverged_ratio": diverged,
        "final_norm_mean": float(final_norm.mean().item()),
        "final_norm_max": float(final_norm.max().item()),
    }


def crown_upper_bound(bundle: Bundle, eps: float, rho: float, method: str, device: torch.device) -> tuple[float, float]:
    graph = SystemViolationGraph(bundle.controller, bundle.lyapunov, bundle.dynamics, rho=rho).to(device)
    graph.eval()

    if method == "alpha-CROWN":
        # alpha optimization is effective on piecewise-linear activations (e.g., ReLU).
        has_relu = any(isinstance(m, torch.nn.ReLU) for m in graph.modules())
        if not has_relu:
            method = "CROWN"

    x0 = torch.zeros(1, 2, device=device)
    bounded = BoundedModule(graph, x0, bound_opts={"relu": "adaptive"})
    ptb = PerturbationLpNorm(norm=torch.inf, eps=eps)
    bx = BoundedTensor(x0, ptb)

    with torch.no_grad():
        lb, ub = bounded.compute_bounds(x=(bx,), method=method)
    return float(lb.item()), float(ub.item())


def sweep_certified_radius(bundle: Bundle, rho: float, method: str, eps_min: float, eps_max: float, steps: int, device: torch.device) -> dict:
    eps_values = torch.linspace(eps_min, eps_max, steps, device=device)
    rows = []
    certified = 0.0

    for eps in eps_values:
        e = float(eps.item())
        lb, ub = crown_upper_bound(bundle, e, rho, method, device)
        rows.append((e, lb, ub))
        if ub < 0.0:
            certified = e

    return {"certified_eps": certified, "rows": rows}


def check_complete_verifier_available() -> bool:
    try:
        import complete_verifier  # noqa: F401

        return True
    except Exception:
        return False


def to_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def build_report(
    eps_eval: float,
    rho: float,
    point_grid: int,
    horizon: int,
    traj_samples: int,
    quad_point: dict,
    quad_traj: dict,
    neural_point: dict,
    neural_traj: dict,
    neural_crown: tuple[float, float],
    neural_alpha_crown: tuple[float, float],
    quad_crown: tuple[float, float],
    quad_alpha_crown: tuple[float, float],
    neural_radius_crown: float,
    neural_radius_alpha: float,
    quad_radius_crown: float,
    quad_radius_alpha: float,
    has_complete_verifier: bool,
) -> str:
    lines = []
    lines.append("# Method Comparison: Quadratic Lyapunov vs alpha-CROWN")
    lines.append("")
    lines.append("## Experiment Setup")
    lines.append(f"- eval_box: [-{eps_eval}, {eps_eval}]^2")
    lines.append(f"- rho: {rho}")
    lines.append(f"- pointwise_grid: {point_grid}x{point_grid}")
    lines.append(f"- trajectory_samples: {traj_samples}, horizon: {horizon}")
    lines.append("")
    lines.append("## Point-wise and Trajectory Metrics")
    lines.append("| Method | Max violation | Mean violation | Violation >= 0 | Converged | Diverged |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| Quadratic (LQR + x^T P x) | {quad_point['max']:.6f} | {quad_point['mean']:.6f} | {to_pct(quad_point['pos_ratio'])} | {to_pct(quad_traj['converged_ratio'])} | {to_pct(quad_traj['diverged_ratio'])} |"
    )
    lines.append(
        f"| Neural (NN + NN) | {neural_point['max']:.6f} | {neural_point['mean']:.6f} | {to_pct(neural_point['pos_ratio'])} | {to_pct(neural_traj['converged_ratio'])} | {to_pct(neural_traj['diverged_ratio'])} |"
    )
    lines.append("")
    lines.append("## Formal Bounds (auto_LiRPA)")
    lines.append("| Method | CROWN UB | alpha-CROWN UB | Certified radius CROWN | Certified radius alpha-CROWN |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| Quadratic | {quad_crown[1]:.6f} | {quad_alpha_crown[1]:.6f} | {quad_radius_crown:.6f} | {quad_radius_alpha:.6f} |"
    )
    lines.append(
        f"| Neural | {neural_crown[1]:.6f} | {neural_alpha_crown[1]:.6f} | {neural_radius_crown:.6f} | {neural_radius_alpha:.6f} |"
    )
    lines.append("")
    lines.append("## alpha-beta-CROWN Availability")
    if has_complete_verifier:
        lines.append("- complete_verifier is installed and can be used for full branch-and-bound (beta splits).")
    else:
        lines.append("- complete_verifier is NOT installed in this environment.")
        lines.append("- This report uses alpha-CROWN (CROWN-Optimized) from auto_LiRPA as the tight bound baseline.")
        lines.append("- To run full alpha-beta-CROWN, install complete_verifier and reuse the same ONNX export graph.")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Quadratic Lyapunov and alpha-CROWN pipelines")
    parser.add_argument("--controller-path", type=str, default="checkpoints/pendulum/pendulum_controller.pth")
    parser.add_argument("--lyapunov-path", type=str, default="checkpoints/pendulum/pendulum_lyapunov.pth")
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--point-grid", type=int, default=41)
    parser.add_argument("--traj-samples", type=int, default=500)
    parser.add_argument("--traj-horizon", type=int, default=120)
    parser.add_argument("--radius-min", type=float, default=0.005)
    parser.add_argument("--radius-max", type=float, default=0.12)
    parser.add_argument("--radius-steps", type=int, default=16)
    parser.add_argument("--u-bound", type=float, default=6.0)
    parser.add_argument("--output", type=str, default="reports/comparison_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    quad = build_quadratic_bundle(device, u_bound=args.u_bound)
    neural = build_neural_bundle(
        device,
        controller_path=args.controller_path,
        lyapunov_path=args.lyapunov_path,
        u_bound=args.u_bound,
    )

    print("[1/4] Running point-wise statistics...")
    quad_point = pointwise_stats(quad, args.eps, args.rho, args.point_grid, device)
    neural_point = pointwise_stats(neural, args.eps, args.rho, args.point_grid, device)

    print("[2/4] Running trajectory statistics...")
    quad_traj = trajectory_stats(quad, args.eps, args.traj_horizon, args.traj_samples, device)
    neural_traj = trajectory_stats(neural, args.eps, args.traj_horizon, args.traj_samples, device)

    print("[3/4] Computing CROWN and alpha-CROWN bounds...")
    quad_crown = crown_upper_bound(quad, args.eps, args.rho, method="CROWN", device=device)
    quad_alpha_crown = crown_upper_bound(quad, args.eps, args.rho, method="alpha-CROWN", device=device)
    neural_crown = crown_upper_bound(neural, args.eps, args.rho, method="CROWN", device=device)
    neural_alpha_crown = crown_upper_bound(neural, args.eps, args.rho, method="alpha-CROWN", device=device)

    print("[4/4] Sweeping certified radius...")
    quad_sweep_crown = sweep_certified_radius(
        quad,
        args.rho,
        method="CROWN",
        eps_min=args.radius_min,
        eps_max=args.radius_max,
        steps=args.radius_steps,
        device=device,
    )
    quad_sweep_alpha = sweep_certified_radius(
        quad,
        args.rho,
        method="alpha-CROWN",
        eps_min=args.radius_min,
        eps_max=args.radius_max,
        steps=args.radius_steps,
        device=device,
    )
    neural_sweep_crown = sweep_certified_radius(
        neural,
        args.rho,
        method="CROWN",
        eps_min=args.radius_min,
        eps_max=args.radius_max,
        steps=args.radius_steps,
        device=device,
    )
    neural_sweep_alpha = sweep_certified_radius(
        neural,
        args.rho,
        method="alpha-CROWN",
        eps_min=args.radius_min,
        eps_max=args.radius_max,
        steps=args.radius_steps,
        device=device,
    )

    report = build_report(
        eps_eval=args.eps,
        rho=args.rho,
        point_grid=args.point_grid,
        horizon=args.traj_horizon,
        traj_samples=args.traj_samples,
        quad_point=quad_point,
        quad_traj=quad_traj,
        neural_point=neural_point,
        neural_traj=neural_traj,
        neural_crown=neural_crown,
        neural_alpha_crown=neural_alpha_crown,
        quad_crown=quad_crown,
        quad_alpha_crown=quad_alpha_crown,
        neural_radius_crown=neural_sweep_crown["certified_eps"],
        neural_radius_alpha=neural_sweep_alpha["certified_eps"],
        quad_radius_crown=quad_sweep_crown["certified_eps"],
        quad_radius_alpha=quad_sweep_alpha["certified_eps"],
        has_complete_verifier=check_complete_verifier_available(),
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n=== SUMMARY ===")
    print(report)
    print(f"[Saved] {args.output}")


if __name__ == "__main__":
    main()