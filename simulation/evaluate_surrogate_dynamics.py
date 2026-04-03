import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


class SurrogateMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, nlayer: int = 3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(0, nlayer - 2)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def resolve_existing_path(requested: Path, fallback_candidates):
    if requested.exists():
        return requested
    for candidate in fallback_candidates:
        p = Path(candidate)
        if p.exists():
            return p
    return None


def make_windows(episode_id, horizon):
    windows = []
    n = len(episode_id)
    start = 0
    while start < n:
        ep = episode_id[start]
        end = start
        while end < n and episode_id[end] == ep:
            end += 1
        length = end - start
        if length >= horizon:
            for idx in range(start, end - horizon + 1):
                windows.append(idx)
        start = end
    return windows


@torch.no_grad()
def evaluate_one_step(model, xu, target, device, batch_size=4096):
    mse_sum = 0.0
    mae_sum = 0.0
    max_abs = 0.0
    n = xu.shape[0]
    n_batches = 0
    for i in range(0, n, batch_size):
        xb = xu[i : i + batch_size].to(device)
        yb = target[i : i + batch_size].to(device)
        pred = model(xb)
        err = pred - yb
        mse_sum += torch.mean(err ** 2).item()
        mae_sum += torch.mean(torch.abs(err)).item()
        max_abs = max(max_abs, torch.max(torch.abs(err)).item())
        n_batches += 1
    return {
        "mse": mse_sum / max(1, n_batches),
        "mae": mae_sum / max(1, n_batches),
        "max_abs": max_abs,
    }


@torch.no_grad()
def evaluate_rollout(model, x, u, x_next, episode_id, horizon, predict_delta, device, max_windows=2000):
    if horizon <= 1:
        return {"rollout_mse": 0.0, "rollout_mae": 0.0}

    windows = make_windows(episode_id, horizon)
    if len(windows) > max_windows:
        windows = windows[:max_windows]

    total_mse = 0.0
    total_mae = 0.0
    count = 0
    for start in windows:
        xk = torch.from_numpy(x[start : start + 1]).to(device)
        for k in range(horizon):
            uk = torch.from_numpy(u[start + k : start + k + 1]).to(device)
            target = torch.from_numpy(x_next[start + k : start + k + 1]).to(device)
            pred = model(torch.cat((xk, uk), dim=1))
            xk = xk + pred if predict_delta else pred
            err = xk - target
            total_mse += torch.mean(err ** 2).item()
            total_mae += torch.mean(torch.abs(err)).item()
            count += 1
    return {
        "rollout_mse": total_mse / max(1, count),
        "rollout_mae": total_mae / max(1, count),
        "windows": len(windows),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MuJoCo surrogate dynamics checkpoint.")
    parser.add_argument("--dataset", type=str, default="data/pendulum/mujoco_feedback/mujoco_transitions.npz", help="Dataset NPZ path.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/mujoco_surrogate/pendulum_mujoco_surrogate_dynamics.pth",
        help="Surrogate checkpoint path.",
    )
    parser.add_argument("--rollout-horizon", type=int, default=10, help="Rollout horizon for drift evaluation.")
    parser.add_argument("--max-windows", type=int, default=2000, help="Cap number of rollout windows.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda; default auto.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    ckpt_path = Path(args.checkpoint)

    dataset_path = resolve_existing_path(
        dataset_path,
        fallback_candidates=[
            "data/pendulum/mujoco_feedback/mujoco_transitions.npz",
            "data/pendulum/mujoco_transitions.npz",
        ],
    )
    if dataset_path is None:
        raise FileNotFoundError(
            "Dataset not found. Checked paths: "
            "data/pendulum/mujoco_feedback/mujoco_transitions.npz, "
            "data/pendulum/mujoco_transitions.npz"
        )

    ckpt_path = resolve_existing_path(
        ckpt_path,
        fallback_candidates=[
            "models/mujoco_surrogate/pendulum_mujoco_surrogate_dynamics.pth",
            "models/mujoco_surrogate_dynamics.pth",
        ],
    )
    if ckpt_path is None:
        raise FileNotFoundError(
            "Checkpoint not found. Checked paths: "
            "models/mujoco_surrogate/pendulum_mujoco_surrogate_dynamics.pth, "
            "models/mujoco_surrogate_dynamics.pth. "
            "Run simulation/train_surrogate_dynamics.py first or pass --checkpoint."
        )

    data = np.load(dataset_path)
    x = data["x"].astype(np.float32)
    u = data["u"].astype(np.float32)
    x_next = data["x_next"].astype(np.float32)
    episode_id = data["episode_id"].astype(np.int32) if "episode_id" in data.files else np.zeros((x.shape[0],), dtype=np.int32)

    xu = np.concatenate([x, u], axis=1)

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    hidden_dim = int(meta.get("hidden_dim", 64))
    nlayer = int(meta.get("nlayer", 3))
    predict_delta = bool(meta.get("predict_delta", False))

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SurrogateMLP(in_dim=xu.shape[1], out_dim=x_next.shape[1], hidden_dim=hidden_dim, nlayer=nlayer).to(device)

    if isinstance(payload, dict):
        state_dict = payload.get("model_state_dict", None)
        if state_dict is None:
            state_dict = payload.get("state_dict", None)
        if state_dict is None:
            raise RuntimeError("Unsupported checkpoint format: missing model_state_dict/state_dict")
        model.load_state_dict(state_dict)
    elif isinstance(payload, torch.nn.Module):
        model = payload.to(device)
    else:
        raise RuntimeError("Unsupported checkpoint format")

    model.eval()

    one_step = evaluate_one_step(
        model=model,
        xu=torch.from_numpy(xu),
        target=torch.from_numpy((x_next - x) if predict_delta else x_next),
        device=device,
        batch_size=args.batch_size,
    )
    rollout = evaluate_rollout(
        model=model,
        x=x,
        u=u,
        x_next=x_next,
        episode_id=episode_id,
        horizon=args.rollout_horizon,
        predict_delta=predict_delta,
        device=device,
        max_windows=args.max_windows,
    )

    print("Surrogate evaluation report")
    print(f"checkpoint: {ckpt_path}")
    print(f"dataset: {dataset_path}")
    print(f"hidden_dim: {hidden_dim}, nlayer: {nlayer}, predict_delta: {predict_delta}")
    print(f"one_step_mse: {one_step['mse']:.6e}")
    print(f"one_step_mae: {one_step['mae']:.6e}")
    print(f"one_step_max_abs: {one_step['max_abs']:.6e}")
    print(f"rollout_horizon: {args.rollout_horizon}")
    print(f"rollout_mse: {rollout['rollout_mse']:.6e}")
    print(f"rollout_mae: {rollout['rollout_mae']:.6e}")
    print(f"rollout_windows: {rollout['windows']}")


if __name__ == "__main__":
    main()
