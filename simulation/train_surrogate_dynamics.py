import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


def make_rollout_windows(x, episode_id, horizon, max_windows: int = 2000):
    if horizon <= 1:
        return []

    windows = []
    n = x.shape[0]
    i = 0
    while i < n:
        ep = episode_id[i]
        j = i
        while j < n and episode_id[j] == ep:
            j += 1

        ep_len = j - i
        if ep_len >= horizon:
            for s in range(i, j - horizon + 1):
                windows.append((s, horizon))
                if len(windows) >= max_windows:
                    return windows
        i = j
    return windows


def rollout_loss(model, x, u, x_next, windows: Sequence, predict_delta, device, max_eval_windows: int = 128):
    if not windows:
        return torch.tensor(0.0, device=device)

    if len(windows) > max_eval_windows:
        choice = np.random.choice(len(windows), size=max_eval_windows, replace=False)
        windows = [windows[i] for i in choice]

    total = torch.tensor(0.0, device=device)
    for start, horizon in windows:
        xk = torch.from_numpy(x[start : start + 1]).to(device)
        for k in range(horizon):
            uk = torch.from_numpy(u[start + k : start + k + 1]).to(device)
            y = model(torch.cat((xk, uk), dim=1))
            xk = xk + y if predict_delta else y
            target = torch.from_numpy(x_next[start + k : start + k + 1]).to(device)
            total = total + nn.functional.mse_loss(xk, target)
    return total / len(windows)


def main():
    parser = argparse.ArgumentParser(description="Train surrogate dynamics from MuJoCo transition dataset.")
    parser.add_argument("--dataset", type=str, default="data/pendulum/mujoco_feedback/mujoco_transitions.npz", help="NPZ transition dataset path.")
    parser.add_argument(
        "--out",
        type=str,
        default="models/mujoco_surrogate/pendulum_mujoco_surrogate_dynamics.pth",
        help="Output checkpoint path.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--nlayer", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If >0, randomly subsample at most this many transitions before train/val split.",
    )
    parser.add_argument("--predict-delta", action="store_true", help="Train model to predict dx = x_next - x.")
    parser.add_argument("--rollout-horizon", type=int, default=1, help="If >1, add rollout loss for this horizon.")
    parser.add_argument("--rollout-weight", type=float, default=0.0, help="Weight for rollout loss term.")
    parser.add_argument("--rollout-window-cap", type=int, default=2000, help="Maximum rollout windows kept in memory.")
    parser.add_argument("--rollout-every", type=int, default=1, help="Compute rollout loss every N epochs to reduce memory/compute.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = np.load(dataset_path)
    x = data["x"].astype(np.float32)
    u = data["u"].astype(np.float32)
    x_next = data["x_next"].astype(np.float32)
    episode_id = data["episode_id"].astype(np.int32) if "episode_id" in data.files else np.zeros((x.shape[0],), dtype=np.int32)

    y = (x_next - x) if args.predict_delta else x_next
    xu = np.concatenate([x, u], axis=1)

    n = xu.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)

    if args.max_samples > 0 and n > args.max_samples:
        idx = idx[: args.max_samples]
        n = idx.shape[0]
        print(f"Subsampled dataset to {n} samples (max_samples={args.max_samples}).")

    n_val = int(n * args.val_ratio)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    xu_train = torch.from_numpy(xu[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    xu_val = torch.from_numpy(xu[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    x_train_np = x[train_idx]
    u_train_np = u[train_idx]
    x_next_train_np = x_next[train_idx]
    episode_train_np = episode_id[train_idx]

    train_loader = DataLoader(TensorDataset(xu_train, y_train), batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SurrogateMLP(
        in_dim=xu.shape[1],
        out_dim=y.shape[1],
        hidden_dim=args.hidden_dim,
        nlayer=args.nlayer,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse = nn.MSELoss()

    windows = []
    if args.rollout_horizon > 1 and args.rollout_weight > 0.0:
        windows = make_rollout_windows(
            x_train_np,
            episode_train_np,
            args.rollout_horizon,
            max_windows=args.rollout_window_cap,
        )

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train = 0.0

        for xu_b, y_b in train_loader:
            xu_b = xu_b.to(device)
            y_b = y_b.to(device)

            pred = model(xu_b)
            loss = mse(pred, y_b)

            if windows and args.rollout_weight > 0.0 and (epoch % args.rollout_every == 0):
                rloss = rollout_loss(
                    model=model,
                    x=x_train_np,
                    u=u_train_np,
                    x_next=x_next_train_np,
                    windows=windows,
                    predict_delta=args.predict_delta,
                    device=device,
                    max_eval_windows=128,
                )
                loss = loss + args.rollout_weight * rloss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train += float(loss.item())

        model.eval()
        with torch.no_grad():
            pred_val = model(xu_val.to(device))
            val_loss = mse(pred_val, y_val.to(device)).item()

        avg_train = total_train / max(1, len(train_loader))
        print(f"Epoch {epoch:03d} | train_loss={avg_train:.6e} | val_loss={val_loss:.6e}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": best_state,
        "meta": {
            "in_dim": int(xu.shape[1]),
            "out_dim": int(y.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "nlayer": int(args.nlayer),
            "predict_delta": bool(args.predict_delta),
            "best_val_loss": float(best_val),
        },
    }
    torch.save(payload, out_path)

    print(f"Saved surrogate checkpoint: {out_path}")
    print(f"Best val loss: {best_val:.6e}")


if __name__ == "__main__":
    main()
