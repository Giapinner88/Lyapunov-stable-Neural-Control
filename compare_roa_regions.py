import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from core.baselines import LQRController, QuadraticLyapunov
from core.dynamics import CartpoleDynamics
from core.roa_utils import compute_rho_boundary
from core.runtime_utils import box_tensors, choose_device, load_trained_system


@dataclass
class CheckpointEntry:
    label: str
    epoch: int
    controller_path: Path
    lyapunov_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare RoA regions and expansion trajectories")
    parser.add_argument("--controller", type=str, default="checkpoints/cartpole/cartpole_controller.pth")
    parser.add_argument("--lyapunov", type=str, default="checkpoints/cartpole/cartpole_lyapunov.pth")
    parser.add_argument("--snapshots-dir", type=str, default="checkpoints/cartpole/snapshots")
    parser.add_argument("--output-dir", type=str, default="reports/roa_method_comparison")
    parser.add_argument("--grid-size", type=int, default=161)
    parser.add_argument("--rho-steps", type=int, default=12)
    parser.add_argument("--theta-min", type=float, default=-1.0)
    parser.add_argument("--theta-max", type=float, default=1.0)
    parser.add_argument("--thetadot-min", type=float, default=-1.0)
    parser.add_argument("--thetadot-max", type=float, default=1.0)
    parser.add_argument("--boundary-samples", type=int, default=600)
    parser.add_argument("--boundary-pgd-steps", type=int, default=30)
    parser.add_argument("--alpha-lyap", type=float, default=0.05)
    parser.add_argument("--select-best-checkpoint", action="store_true")
    parser.add_argument(
        "--best-metric",
        type=str,
        default="verified_area_pct",
        choices=["verified_area_pct", "sublevel_area_pct", "rho"],
        help="Metric for selecting best checkpoint",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def extract_epoch_from_name(name: str) -> Optional[int]:
    m = re.search(r"_ep(\d+)", name)
    if not m:
        return None
    return int(m.group(1))


def build_checkpoint_entries(controller_path: Path, lyapunov_path: Path, snapshots_dir: Path) -> list[CheckpointEntry]:
    entries: list[CheckpointEntry] = []

    if snapshots_dir.exists():
        controllers = sorted(snapshots_dir.glob("cartpole_controller_ep*.pth"))
        for c_path in controllers:
            epoch = extract_epoch_from_name(c_path.name)
            if epoch is None:
                continue
            l_path = snapshots_dir / c_path.name.replace("controller", "lyapunov")
            if not l_path.exists():
                continue
            entries.append(
                CheckpointEntry(
                    label=f"ep{epoch:03d}",
                    epoch=epoch,
                    controller_path=c_path,
                    lyapunov_path=l_path,
                )
            )

    # Always include latest checkpoint as final reference.
    entries.append(
        CheckpointEntry(
            label="final",
            epoch=max([e.epoch for e in entries], default=-1) + 1,
            controller_path=controller_path,
            lyapunov_path=lyapunov_path,
        )
    )

    # Remove duplicates by controller path while keeping order.
    unique: list[CheckpointEntry] = []
    seen = set()
    for e in sorted(entries, key=lambda x: x.epoch):
        key = str(e.controller_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(e)

    return unique


def build_grid(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    theta = torch.linspace(args.theta_min, args.theta_max, args.grid_size, device=device)
    thetadot = torch.linspace(args.thetadot_min, args.thetadot_max, args.grid_size, device=device)
    th, thd = torch.meshgrid(theta, thetadot, indexing="ij")

    x = torch.zeros((args.grid_size * args.grid_size, 4), device=device)
    x[:, 2] = th.reshape(-1)
    x[:, 3] = thd.reshape(-1)

    return x, th.detach().cpu().numpy(), thd.detach().cpu().numpy()


def load_neural_pair(controller_path: Path, lyapunov_path: Path, device: torch.device):
    bundle = load_trained_system(
        controller_path,
        lyapunov_path,
        system_name="cartpole",
        device=device,
    )
    return bundle.controller, bundle.lyapunov


def compute_masks(
    dynamics: CartpoleDynamics,
    controller: torch.nn.Module,
    lyapunov: torch.nn.Module,
    grid_x: torch.Tensor,
    rho: float,
    alpha_lyap: float,
    x_min: torch.Tensor,
    x_max: torch.Tensor,
) -> dict:
    with torch.no_grad():
        v = lyapunov(grid_x).squeeze(1)
        u = controller(grid_x)
        x_next = dynamics.step(grid_x, u)
        v_next = lyapunov(x_next).squeeze(1)
        violation = v_next - (1.0 - alpha_lyap) * v

        in_box_next = ((x_next >= x_min) & (x_next <= x_max)).all(dim=1)
        sublevel = v < rho
        verified = sublevel & (violation <= 0.0) & in_box_next

    return {
        "sublevel": sublevel.detach().cpu().numpy(),
        "verified": verified.detach().cpu().numpy(),
        "violation": violation.detach().cpu().numpy(),
        "v": v.detach().cpu().numpy(),
    }


def area_ratio(mask: np.ndarray) -> float:
    return float(mask.mean() * 100.0)


def select_best_entry(expansion_rows: list[dict], metric: str) -> Optional[dict]:
    if not expansion_rows:
        return None
    metric = metric.lower()
    if metric == "verified_area_pct":
        return max(expansion_rows, key=lambda r: float(r["neural_verified_area_pct"]))
    if metric == "sublevel_area_pct":
        return max(expansion_rows, key=lambda r: float(r["neural_sublevel_area_pct"]))
    if metric == "rho":
        return max(expansion_rows, key=lambda r: float(r["rho"]))
    raise ValueError(f"Unsupported best metric: {metric}")


def compare_methods_plot(
    th_np: np.ndarray,
    thd_np: np.ndarray,
    neural_sub: np.ndarray,
    neural_ver: np.ndarray,
    quad_sub: np.ndarray,
    quad_ver: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.contourf(th_np, thd_np, neural_sub.reshape(th_np.shape).astype(float), levels=[0.5, 1.5], alpha=0.35, cmap="Blues")
    ax.contour(th_np, thd_np, neural_ver.reshape(th_np.shape).astype(float), levels=[0.5], colors=["navy"], linewidths=2)
    ax.set_title("Neural RoA (sublevel + SMT-like verified)")
    ax.set_xlabel("theta")
    ax.set_ylabel("theta_dot")
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.contourf(th_np, thd_np, quad_sub.reshape(th_np.shape).astype(float), levels=[0.5, 1.5], alpha=0.35, cmap="Oranges")
    ax.contour(th_np, thd_np, quad_ver.reshape(th_np.shape).astype(float), levels=[0.5], colors=["darkred"], linewidths=2)
    ax.set_title("SOS/SDP proxy (Quadratic V) + SMT-like verified")
    ax.set_xlabel("theta")
    ax.set_ylabel("theta_dot")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def expansion_plot(expansion_rows: list[dict], out_path: Path) -> None:
    labels = [r["label"] for r in expansion_rows]
    sub = [r["neural_sublevel_area_pct"] for r in expansion_rows]
    ver = [r["neural_verified_area_pct"] for r in expansion_rows]
    rho = [r["rho"] for r in expansion_rows]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(x - width / 2, sub, width, label="sublevel area %")
    ax1.bar(x + width / 2, ver, width, label="verified area %")
    ax1.set_ylabel("Area Percent on 2D Slice")
    ax1.set_xlabel("Checkpoint")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30)
    ax1.grid(alpha=0.2, axis="y")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, rho, marker="o", color="black", label="rho")
    ax2.set_ylabel("rho")

    fig.suptitle("RoA Expansion Across Checkpoints")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def area_vs_rho_plot(rho_curve_rows: list[dict], out_path: Path) -> None:
    by_label: dict[str, list[dict]] = {}
    for row in rho_curve_rows:
        by_label.setdefault(row["label"], []).append(row)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, rows in by_label.items():
        rows = sorted(rows, key=lambda r: r["rho"])
        rho = [r["rho"] for r in rows]
        area = [r["verified_area_pct"] for r in rows]
        ax.plot(rho, area, marker="o", label=label)

    ax.set_title("Verified Area vs rho")
    ax.set_xlabel("rho")
    ax.set_ylabel("Verified area % on 2D slice")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_bundle = load_trained_system(
        args.controller,
        args.lyapunov,
        system_name="cartpole",
        device=device,
    )
    config = base_bundle.config
    dynamics = base_bundle.dynamics
    x_min, x_max = box_tensors(config, device=device, dtype=torch.float32)

    # Build methods: neural and SOS/SDP proxy (quadratic baseline).
    K, S = dynamics.get_lqr_baseline()
    K = K.to(device)
    S = S.to(device)
    quad_controller = LQRController(K, u_bound=config.model.u_bound).to(device)
    quad_lyapunov = QuadraticLyapunov(S).to(device)
    quad_controller.eval()
    quad_lyapunov.eval()

    grid_x, th_np, thd_np = build_grid(args, device)

    controller_path = Path(args.controller)
    lyapunov_path = Path(args.lyapunov)
    snapshots_dir = Path(args.snapshots_dir)

    entries = build_checkpoint_entries(controller_path, lyapunov_path, snapshots_dir)
    if not entries:
        raise RuntimeError("No checkpoint pairs found.")

    # Expansion along checkpoints.
    expansion_rows: list[dict] = []
    rho_curve_rows: list[dict] = []

    for entry in entries:
        c, v = load_neural_pair(entry.controller_path, entry.lyapunov_path, device)
        rho, _ = compute_rho_boundary(
            v,
            dynamics,
            c,
            x_min,
            x_max,
            n_boundary_samples=max(200, args.boundary_samples // 2),
            n_pgd_steps=max(12, args.boundary_pgd_steps // 2),
            gamma=0.9,
            device=device,
            verbose=False,
        )
        masks = compute_masks(
            dynamics,
            c,
            v,
            grid_x,
            rho=rho,
            alpha_lyap=args.alpha_lyap,
            x_min=x_min,
            x_max=x_max,
        )

        expansion_rows.append(
            {
                "label": entry.label,
                "epoch": entry.epoch,
                "rho": float(rho),
                "neural_sublevel_area_pct": area_ratio(masks["sublevel"]),
                "neural_verified_area_pct": area_ratio(masks["verified"]),
                "controller_path": str(entry.controller_path),
                "lyapunov_path": str(entry.lyapunov_path),
            }
        )

        rho_values = np.linspace(max(1e-6, rho / args.rho_steps), rho, args.rho_steps)
        for r in rho_values:
            curve_masks = compute_masks(
                dynamics,
                c,
                v,
                grid_x,
                rho=float(r),
                alpha_lyap=args.alpha_lyap,
                x_min=x_min,
                x_max=x_max,
            )
            rho_curve_rows.append(
                {
                    "label": entry.label,
                    "epoch": entry.epoch,
                    "rho": float(r),
                    "verified_area_pct": area_ratio(curve_masks["verified"]),
                    "sublevel_area_pct": area_ratio(curve_masks["sublevel"]),
                }
            )

    expansion_rows = sorted(expansion_rows, key=lambda x: x["epoch"])
    rho_curve_rows = sorted(rho_curve_rows, key=lambda x: (x["epoch"], x["rho"]))

    best_entry = None
    selected_controller_path = controller_path
    selected_lyapunov_path = lyapunov_path
    selected_source = "final"
    if args.select_best_checkpoint:
        best_entry = select_best_entry(expansion_rows, args.best_metric)
        if best_entry is not None:
            selected_controller_path = Path(best_entry["controller_path"])
            selected_lyapunov_path = Path(best_entry["lyapunov_path"])
            selected_source = best_entry["label"]

    # Compare selected neural checkpoint vs quadratic proxy.
    neural_controller, neural_lyapunov = load_neural_pair(
        selected_controller_path,
        selected_lyapunov_path,
        device,
    )

    neural_rho, _ = compute_rho_boundary(
        neural_lyapunov,
        dynamics,
        neural_controller,
        x_min,
        x_max,
        n_boundary_samples=args.boundary_samples,
        n_pgd_steps=args.boundary_pgd_steps,
        gamma=0.9,
        device=device,
        verbose=False,
    )
    quad_rho, _ = compute_rho_boundary(
        quad_lyapunov,
        dynamics,
        quad_controller,
        x_min,
        x_max,
        n_boundary_samples=args.boundary_samples,
        n_pgd_steps=args.boundary_pgd_steps,
        gamma=0.9,
        device=device,
        verbose=False,
    )

    neural_masks = compute_masks(
        dynamics,
        neural_controller,
        neural_lyapunov,
        grid_x,
        rho=neural_rho,
        alpha_lyap=args.alpha_lyap,
        x_min=x_min,
        x_max=x_max,
    )
    quad_masks = compute_masks(
        dynamics,
        quad_controller,
        quad_lyapunov,
        grid_x,
        rho=quad_rho,
        alpha_lyap=args.alpha_lyap,
        x_min=x_min,
        x_max=x_max,
    )

    compare_methods_plot(
        th_np,
        thd_np,
        neural_masks["sublevel"],
        neural_masks["verified"],
        quad_masks["sublevel"],
        quad_masks["verified"],
        out_dir / "roa_method_regions.png",
    )

    method_summary = {
        "notes": [
            "SOS/SDP curve here is a proxy using quadratic Lyapunov V(x)=x^T P x from LQR Riccati solution.",
            "SMT-like region is approximated by dense-grid one-step Lyapunov/decrease verification.",
            "For exact SOS/SMT replication, external solver integration is required.",
        ],
        "selected_neural_checkpoint": {
            "source": selected_source,
            "controller_path": str(selected_controller_path),
            "lyapunov_path": str(selected_lyapunov_path),
            "select_best_checkpoint": bool(args.select_best_checkpoint),
            "best_metric": args.best_metric,
        },
        "neural": {
            "rho": float(neural_rho),
            "sublevel_area_pct": area_ratio(neural_masks["sublevel"]),
            "verified_area_pct": area_ratio(neural_masks["verified"]),
        },
        "sos_sdp_proxy_quadratic": {
            "rho": float(quad_rho),
            "sublevel_area_pct": area_ratio(quad_masks["sublevel"]),
            "verified_area_pct": area_ratio(quad_masks["verified"]),
        },
    }

    expansion_plot(expansion_rows, out_dir / "roa_expansion_checkpoints.png")
    area_vs_rho_plot(rho_curve_rows, out_dir / "verified_area_vs_rho.png")

    save_csv(expansion_rows, out_dir / "roa_expansion_checkpoints.csv")
    save_csv(rho_curve_rows, out_dir / "verified_area_vs_rho.csv")

    summary = {
        "method_comparison": method_summary,
        "checkpoint_count": len(expansion_rows),
        "grid_size": args.grid_size,
        "slice": {
            "fixed": {"x": 0.0, "x_dot": 0.0},
            "varying": {
                "theta": [args.theta_min, args.theta_max],
                "theta_dot": [args.thetadot_min, args.thetadot_max],
            },
        },
    }
    (out_dir / "roa_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if best_entry is not None:
        (out_dir / "best_checkpoint.json").write_text(json.dumps(best_entry, indent=2), encoding="utf-8")

    print("[RoA Compare] Done")
    print(f"- regions plot: {out_dir / 'roa_method_regions.png'}")
    print(f"- expansion plot: {out_dir / 'roa_expansion_checkpoints.png'}")
    print(f"- rho curve plot: {out_dir / 'verified_area_vs_rho.png'}")
    print(f"- summary: {out_dir / 'roa_comparison_summary.json'}")
    if best_entry is not None:
        print(f"- best checkpoint: {out_dir / 'best_checkpoint.json'}")


if __name__ == "__main__":
    main()
