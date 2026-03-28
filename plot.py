#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn

import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.pendulum as pendulum
import neural_lyapunov_training.visualization as visualization


ROOT = Path(__file__).resolve().parent

@dataclass
class SelectedRun:
    run_dir: Path
    checkpoint: Path
    seed: int | None


def _cfg_seed(cfg: DictConfig) -> int | None:
    if "seed" in cfg:
        return int(cfg.seed)
    if "pendulum" in cfg and "seed" in cfg.pendulum:
        return int(cfg.pendulum.seed)
    return None


def _unwrap_cfg(cfg: DictConfig) -> DictConfig:
    if "model" in cfg and "train" in cfg:
        return cfg
    if "pendulum" in cfg and "model" in cfg.pendulum:
        return cfg.pendulum
    raise ValueError("Could not find model/train section in config.yaml")


def _parse_scale(limit_scale: Any) -> float:
    if isinstance(limit_scale, (list, tuple)):
        return float(limit_scale[-1])
    if hasattr(limit_scale, "__len__") and not isinstance(limit_scale, (str, bytes)):
        return float(limit_scale[-1])
    return float(limit_scale)


def _parse_checkpoint_scale(path: Path) -> float:
    stem = path.stem
    # Expected format: lyaloss_0.8
    if "_" not in stem:
        return float("-inf")
    try:
        return float(stem.split("_", 1)[1])
    except ValueError:
        return float("-inf")


def _latest_pendulum_run(seed: int | None) -> SelectedRun:
    base = ROOT / "output" / "pendulum_state"
    if not base.exists():
        raise FileNotFoundError(f"Pendulum output directory not found: {base}")

    candidates: list[SelectedRun] = []
    for cfg_path in base.rglob("config.yaml"):
        run_dir = cfg_path.parent
        lyaloss_files = sorted(
            run_dir.glob("lyaloss_*.pth"),
            key=lambda p: (_parse_checkpoint_scale(p), p.stat().st_mtime),
        )
        if not lyaloss_files:
            continue
        cfg = OmegaConf.load(cfg_path)
        run_seed = _cfg_seed(cfg)
        if seed is not None and run_seed != seed:
            continue
        candidates.append(SelectedRun(run_dir=run_dir, checkpoint=lyaloss_files[-1], seed=run_seed))

    if not candidates:
        seed_msg = "" if seed is None else f" for seed={seed}"
        raise FileNotFoundError(f"No pendulum checkpoint found{seed_msg} under {base}")

    candidates.sort(key=lambda c: c.checkpoint.stat().st_mtime)
    return candidates[-1]


def _latest_cartpole_run(seed: int | None) -> SelectedRun:
    base = ROOT / "outputs" / "cartpole_std"
    if not base.exists():
        raise FileNotFoundError(f"Cartpole output directory not found: {base}")

    candidates: list[SelectedRun] = []
    for ckpt in base.rglob("cartpole_lyaloss_std.pth"):
        run_dir = ckpt.parent
        run_seed: int | None = None
        for parent in run_dir.parents:
            if parent.name.startswith("seed_"):
                try:
                    run_seed = int(parent.name.split("_", 1)[1])
                except ValueError:
                    run_seed = None
                break
        if seed is not None and run_seed != seed:
            continue
        candidates.append(SelectedRun(run_dir=run_dir, checkpoint=ckpt, seed=run_seed))

    if not candidates:
        seed_msg = "" if seed is None else f" for seed={seed}"
        raise FileNotFoundError(f"No cartpole checkpoint found{seed_msg} under {base}")

    candidates.sort(key=lambda c: c.checkpoint.stat().st_mtime)
    return candidates[-1]


def _load_model_dict(checkpoint: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict):
        return ckpt
    return {"state_dict": ckpt}


def plot_latest_pendulum(seed: int | None) -> Path:
    selected = _latest_pendulum_run(seed)
    cfg = OmegaConf.load(selected.run_dir / "config.yaml")
    cfg_u = _unwrap_cfg(cfg)

    device = torch.device("cpu")
    dtype = torch.float32

    dt = float(cfg_u.model.dt)
    limit_scale = _parse_scale(cfg_u.model.limit_scale)
    base_limit = torch.tensor(list(cfg_u.model.limit), dtype=dtype, device=device)
    limit = limit_scale * base_limit
    lower_limit = -limit
    upper_limit = limit

    pendulum_ct = pendulum.PendulumDynamics(m=0.15, l=0.5, beta=0.1)
    dynamics = dynamical_system.SecondOrderDiscreteTimeSystem(
        pendulum_ct,
        dt=dt,
        position_integration=dynamical_system.IntegrationMethod[cfg_u.model.position_integration],
        velocity_integration=dynamical_system.IntegrationMethod[cfg_u.model.velocity_integration],
    )

    controller = controllers.NeuralNetworkController(
        nlayer=int(cfg_u.model.controller_nlayer),
        in_dim=2,
        out_dim=1,
        hidden_dim=int(cfg_u.model.controller_hidden_dim),
        clip_output="clamp",
        u_lo=torch.tensor([-float(cfg_u.model.u_max)], dtype=dtype),
        u_up=torch.tensor([float(cfg_u.model.u_max)], dtype=dtype),
        x_equilibrium=pendulum_ct.x_equilibrium,
        u_equilibrium=pendulum_ct.u_equilibrium,
    )

    if bool(cfg_u.model.lyapunov.quadratic):
        lyapunov_nn = lyapunov.NeuralNetworkQuadraticLyapunov(
            goal_state=torch.zeros(2, dtype=dtype),
            x_dim=2,
            R_rows=2,
            eps=0.01,
        )
    else:
        lyapunov_nn = lyapunov.NeuralNetworkLyapunov(
            goal_state=torch.tensor([0.0, 0.0], dtype=dtype),
            hidden_widths=list(cfg_u.model.lyapunov.hidden_widths),
            x_dim=2,
            R_rows=3,
            absolute_output=True,
            eps=0.01,
            activation=nn.LeakyReLU,
            V_psd_form=str(cfg_u.model.V_psd_form),
        )

    derivative_lyaloss = lyapunov.LyapunovDerivativeLoss(
        dynamics=dynamics,
        controller=controller,
        lyap_nn=lyapunov_nn,
        box_lo=lower_limit,
        box_up=upper_limit,
        rho_multiplier=float(cfg_u.model.rho_multiplier),
        kappa=float(cfg_u.model.kappa),
        hard_max=bool(cfg_u.train.hard_max),
    )

    model_dict = _load_model_dict(selected.checkpoint, device)
    derivative_lyaloss.load_state_dict(model_dict["state_dict"])
    if "x_boundary" in model_dict:
        derivative_lyaloss.x_boundary = model_dict["x_boundary"]

    rho_values = None
    if "rho" in model_dict:
        rho_raw = model_dict["rho"]
        rho_values = [float(rho_raw.item() if hasattr(rho_raw, "item") else rho_raw)]

    out_dir = ROOT / "plots" / "pendulum"
    run_tag = selected.run_dir.relative_to(ROOT / "output" / "pendulum_state").as_posix().replace("/", "_")
    figure_path = visualization.plot_pendulum_diagnostics(
        derivative_lyaloss=derivative_lyaloss,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        out_dir=out_dir,
        rho_values=rho_values,
        filename=f"pendulum_{run_tag}.png",
    )

    print(f"Selected run: {selected.run_dir}")
    print(f"Seed: {selected.seed}")
    print(f"Checkpoint: {selected.checkpoint}")
    print(f"Figure: {figure_path}")
    return figure_path


def _load_cartpole_train_cfg(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "train_config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    return {
        "seed": 123,
        "dt": 0.05,
        "u_max": 30.0,
        "limit": 1.0,
        "rho_multiplier": 0.9,
        "kappa": 0.01,
    }


def plot_latest_cartpole(seed: int | None) -> Path:
    from examples.cartpole.state_training import CartPoleDynamicsStd

    selected = _latest_cartpole_run(seed)
    cfg = _load_cartpole_train_cfg(selected.run_dir)

    device = torch.device("cpu")
    dtype = torch.float32

    limit_value = float(cfg.get("limit", 1.0))
    lower_limit = torch.tensor([-limit_value] * 4, dtype=dtype, device=device)
    upper_limit = torch.tensor([limit_value] * 4, dtype=dtype, device=device)

    dynamics_ct = CartPoleDynamicsStd(mc=10.0, mp=1.0, l=1.0, gravity=9.81)
    dynamics = dynamical_system.FirstOrderDiscreteTimeSystem(
        dynamics_ct,
        dt=float(cfg.get("dt", 0.05)),
        integration=dynamical_system.IntegrationMethod.ExplicitEuler,
    )

    controller = controllers.NeuralNetworkController(
        nlayer=2,
        in_dim=4,
        out_dim=1,
        hidden_dim=64,
        clip_output="clamp",
        u_lo=torch.tensor([-float(cfg.get("u_max", 30.0))], dtype=dtype),
        u_up=torch.tensor([float(cfg.get("u_max", 30.0))], dtype=dtype),
        x_equilibrium=dynamics_ct.x_equilibrium,
        u_equilibrium=dynamics_ct.u_equilibrium,
        activation=nn.ReLU,
    )

    lyapunov_nn = lyapunov.NeuralNetworkLyapunov(
        goal_state=torch.zeros((4,), dtype=dtype),
        hidden_widths=[64, 64],
        x_dim=4,
        R_rows=4,
        absolute_output=True,
        eps=0.01,
        activation=nn.LeakyReLU,
        V_psd_form="L1",
    )

    derivative_lyaloss = lyapunov.LyapunovDerivativeLoss(
        dynamics=dynamics,
        controller=controller,
        lyap_nn=lyapunov_nn,
        box_lo=lower_limit,
        box_up=upper_limit,
        rho_multiplier=float(cfg.get("rho_multiplier", 0.9)),
        kappa=float(cfg.get("kappa", 0.01)),
        hard_max=True,
    )

    model_dict = _load_model_dict(selected.checkpoint, device)
    derivative_lyaloss.load_state_dict(model_dict["state_dict"])
    if "x_boundary" in model_dict:
        derivative_lyaloss.x_boundary = model_dict["x_boundary"]
    if "lower_limit" in model_dict:
        lower_limit = model_dict["lower_limit"].to(device)
    if "upper_limit" in model_dict:
        upper_limit = model_dict["upper_limit"].to(device)

    rho_values = None
    if "rho" in model_dict:
        rho_raw = model_dict["rho"]
        rho_values = [float(rho_raw.item() if hasattr(rho_raw, "item") else rho_raw)]

    out_dir = ROOT / "plots" / "cartpole"
    run_tag = selected.run_dir.relative_to(ROOT / "outputs" / "cartpole_std").as_posix().replace("/", "_")
    figure_path = visualization.plot_cartpole_diagnostics(
        derivative_lyaloss=derivative_lyaloss,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        out_dir=out_dir,
        rho_values=rho_values,
        phase_indices=(1, 3),
        filename=f"cartpole_{run_tag}.png",
    )

    print(f"Selected run: {selected.run_dir}")
    print(f"Seed: {selected.seed}")
    print(f"Checkpoint: {selected.checkpoint}")
    print(f"Figure: {figure_path}")
    return figure_path


def _interactive_choice() -> tuple[str, int | None] | None:
    options = {
        "1": "pendulum",
        "2": "cartpole",
        "pendulum": "pendulum",
        "cartpole": "cartpole",
    }

    print("Select model to plot:")
    print("  1) pendulum")
    print("  2) cartpole")
    print("Type number/name, or q to quit. Default: 1")

    raw = input("> ").strip().lower()
    if raw == "":
        raw = "1"
    if raw in {"q", "quit", "exit"}:
        return None

    model = options.get(raw)
    if model is None:
        print("Invalid selection.")
        return None

    seed_raw = input("Seed (optional, Enter = latest): ").strip()
    if seed_raw == "":
        return model, None
    try:
        return model, int(seed_raw)
    except ValueError:
        print("Invalid seed. Using latest.")
        return model, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot diagnostics from latest trained seed.")
    subparsers = parser.add_subparsers(dest="model")

    pendulum_parser = subparsers.add_parser("pendulum", help="Plot latest pendulum run")
    pendulum_parser.add_argument("--seed", type=int, default=None, help="Optional seed to filter")

    cartpole_parser = subparsers.add_parser("cartpole", help="Plot latest cartpole run")
    cartpole_parser.add_argument("--seed", type=int, default=None, help="Optional seed to filter")

    args = parser.parse_args()

    if args.model is None:
        selected = _interactive_choice()
        if selected is None:
            print("Exit without plotting.")
            return
        model, seed = selected
        if model == "pendulum":
            plot_latest_pendulum(seed)
            return
        plot_latest_cartpole(seed)
        return

    if args.model == "pendulum":
        plot_latest_pendulum(args.seed)
        return
    plot_latest_cartpole(args.seed)


if __name__ == "__main__":
    main()
