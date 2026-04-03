import sys
import shutil
import subprocess
import os
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# Allow running this file directly without requiring PYTHONPATH setup.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import apps.pendulum.output_feedback as output_feedback
import neural_lyapunov_training.dynamical_system as dynamical_system


def _run_module_script(script_rel_path: str, forwarded_args):
    script_path = PROJECT_ROOT / script_rel_path
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    cmd = [sys.executable, str(script_path), *forwarded_args]
    print(f"[mujoco_feedback] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _train_observer_mujoco(args):
    """
    Train observer specifically for MuJoCo surrogate dynamics.
    
    Usage:
      python apps/pendulum/mujoco_feedback.py train_observer \
        --surrogate models/mujoco_surrogate/pendulum_mujoco_surrogate_dynamics.pth \
        --dataset data/pendulum/mujoco_feedback/mujoco_transitions.npz \
        --out models/mujoco_feedback/observer_mujoco_surrogate_[8,8].pth \
        [--epochs 100] [--batch-size 256]
    """
    import argparse
    import numpy as np
    from neural_lyapunov_training.controllers import NeuralNetworkLuenbergerObserver
    from neural_lyapunov_training.pendulum import PendulumDynamics
    
    parser = argparse.ArgumentParser(description="Train observer for MuJoCo surrogate dynamics")
    parser.add_argument("--surrogate", required=True, help="Path to surrogate model checkpoint")
    parser.add_argument("--dataset", required=True, help="Path to transition dataset NPZ")
    parser.add_argument("--out", required=True, help="Output observer checkpoint path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    parsed_args = parser.parse_args(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = 0.01
    
    # Load surrogate dynamics
    print(f"[train_observer] Loading surrogate from {parsed_args.surrogate}")
    pendulum_continuous = PendulumDynamics(m=0.15, l=0.5, beta=0.1)
    surrogate_dynamics = MuJoCoSurrogateSecondOrderDiscreteTimeSystem(
        continuous_time_system=pendulum_continuous,
        dt=dt,
        surrogate_path=str(PROJECT_ROOT / parsed_args.surrogate),
        predict_delta=True,
    )
    surrogate_dynamics.to(device)
    
    # Create observer
    print("[train_observer] Creating observer")
    h = lambda x: pendulum_continuous.h(x)
    observer = NeuralNetworkLuenbergerObserver(
        z_dim=2,
        y_dim=1,
        dynamics=surrogate_dynamics,
        h=h,
        zero_obs_error=torch.zeros(1, 1),
        fc_hidden_dim=[8, 8],
    ).to(device)
    
    # Load dataset
    print(f"[train_observer] Loading dataset from {parsed_args.dataset}")
    dataset = np.load(str(PROJECT_ROOT / parsed_args.dataset))
    x_data = torch.from_numpy(dataset['x']).float()
    u_data = torch.from_numpy(dataset['u']).float()
    x_next_data = torch.from_numpy(dataset['x_next']).float()
    
    # Training loop
    optimizer = torch.optim.Adam(observer.parameters(), lr=parsed_args.lr)
    batch_size = parsed_args.batch_size
    n_samples = x_data.shape[0]
    
    print(f"[train_observer] Training for {parsed_args.epochs} epochs on {n_samples} samples")
    for epoch in range(parsed_args.epochs):
        total_loss = 0.0
        n_batches = 0
        
        # Shuffle data
        indices = torch.randperm(n_samples)
        
        for batch_start in range(0, n_samples, batch_size):
            batch_indices = indices[batch_start:batch_start+batch_size]
            x_batch = x_data[batch_indices].to(device)
            u_batch = u_data[batch_indices].to(device)
            x_next_batch = x_next_data[batch_indices].to(device)
            
            # Observer update: z_{k+1} = observer(z_k, u_k, y_k)
            # where y_k = h(x_k) = x_k[:, :1]  (only theta)
            y_batch = x_batch[:, :1]  # Measurement (theta only)
            
            # Initialize z from x with zero error
            z_batch = x_batch.clone()
            
            # Observer prediction step
            z_next_pred = observer(z_batch, u_batch, y_batch)
            
            # Expectation: z should track x, so z_next should be close to x_next
            loss = torch.mean((z_next_pred - x_next_batch) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % max(1, parsed_args.epochs // 10) == 0:
            print(f"  Epoch {epoch+1}/{parsed_args.epochs}: loss={avg_loss:.6f}")
    
    # Save checkpoint
    output_path = PROJECT_ROOT / parsed_args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"observer_state_dict": observer.state_dict()}
    torch.save(checkpoint, str(output_path))
    print(f"[train_observer] Observer saved to: {output_path}")


def _dispatch_subcommand_if_requested():
    """
    Optional single-entrypoint mode.

    Usage:
      python apps/pendulum/mujoco_feedback.py collect [args...]
      python apps/pendulum/mujoco_feedback.py train_surrogate [args...]
      python apps/pendulum/mujoco_feedback.py evaluate_surrogate [args...]
      python apps/pendulum/mujoco_feedback.py train_observer [args...]
      python apps/pendulum/mujoco_feedback.py train_policy [hydra args...]
      python apps/pendulum/mujoco_feedback.py all
    """
    if len(sys.argv) < 2:
        return

    subcommand = sys.argv[1]
    forwarded_args = sys.argv[2:]

    if subcommand == "collect":
        _run_module_script("simulation/collect_mujoco_transitions.py", forwarded_args)
        raise SystemExit(0)
    if subcommand == "train_surrogate":
        _run_module_script("simulation/train_surrogate_dynamics.py", forwarded_args)
        raise SystemExit(0)
    if subcommand == "evaluate_surrogate":
        _run_module_script("simulation/evaluate_surrogate_dynamics.py", forwarded_args)
        raise SystemExit(0)
    if subcommand == "train_policy":
        # Remove subcommand token so Hydra receives a clean argv.
        sys.argv = [sys.argv[0], *forwarded_args]
        return

    if subcommand == "train_observer":
        _train_observer_mujoco(forwarded_args)
        raise SystemExit(0)

    if subcommand == "all":
        dataset_path = "data/pendulum/mujoco_feedback/mujoco_transitions.npz"
        surrogate_path = "models/mujoco_surrogate/pendulum_mujoco_surrogate_dynamics.pth"
        observer_path = "models/mujoco_feedback/observer_mujoco_surrogate_[8,8].pth"
        max_samples = os.environ.get("MUJOCO_SURROGATE_MAX_SAMPLES", "120000")

        _run_module_script(
            "simulation/collect_mujoco_transitions.py",
            ["--out", dataset_path],
        )
        _run_module_script(
            "simulation/train_surrogate_dynamics.py",
            [
                "--dataset",
                dataset_path,
                "--out",
                surrogate_path,
                "--predict-delta",
                "--max-samples",
                max_samples,
                "--rollout-every",
                "5",
                "--rollout-window-cap",
                "1000",
            ],
        )
        _run_module_script(
            "simulation/evaluate_surrogate_dynamics.py",
            ["--dataset", dataset_path, "--checkpoint", surrogate_path],
        )
        
        # Train observer specifically for surrogate dynamics
        _train_observer_mujoco([
            "--surrogate", surrogate_path,
            "--dataset", dataset_path,
            "--out", observer_path,
            "--epochs", "50",
            "--batch-size", "256",
        ])

        # Continue into Hydra-based policy training with surrogate backend.
        sys.argv = [
            sys.argv[0],
            "+model.dynamics_backend=mujoco_surrogate",
            f"+model.mujoco_surrogate.path={surrogate_path}",
        ]
        return

    # Unknown command: leave argv untouched so normal Hydra flow still works.


class _MLPTransition(torch.nn.Module):
    """Simple fallback MLP transition model for surrogate dynamics."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, nlayer: int = 3):
        super().__init__()
        layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU()]
        for _ in range(max(0, nlayer - 2)):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_dim, out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MuJoCoSurrogateSecondOrderDiscreteTimeSystem(dynamical_system.DiscreteTimeSystem):
    """
    Discrete-time system wrapper using a MuJoCo-trained surrogate transition model.

    It keeps the same public interface as the existing dynamics class so the
    Lyapunov/controller/observer pipeline can remain unchanged.
    """

    def __init__(
        self,
        continuous_time_system,
        dt: float,
        surrogate_path: str,
        predict_delta: bool = False,
        hidden_dim: int = 64,
        nlayer: int = 3,
        *args,
        **kwargs,
    ):
        super().__init__(continuous_time_system.nx, continuous_time_system.nu)
        self.nx = continuous_time_system.nx
        self.nu = continuous_time_system.nu
        self.nq = self.nx // 2
        self.dt = dt
        self.continuous_time_system = continuous_time_system
        self.predict_delta = predict_delta

        surrogate_model = self._load_surrogate(
            surrogate_path=surrogate_path,
            in_dim=self.nx + self.nu,
            out_dim=self.nx,
            hidden_dim=hidden_dim,
            nlayer=nlayer,
        )
        # Keep surrogate model out of nn.Module state_dict so legacy
        # controller/observer checkpoints (without dynamics.model.* keys)
        # remain loadable.
        self.__dict__["model"] = surrogate_model

    def _load_surrogate(
        self,
        surrogate_path: str,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        nlayer: int,
    ) -> torch.nn.Module:
        ckpt_path = Path(surrogate_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"MuJoCo surrogate checkpoint not found: {ckpt_path}"
            )

        # Try TorchScript first.
        try:
            model = torch.jit.load(str(ckpt_path), map_location="cpu")
            model.eval()
            return model
        except Exception:
            pass

        # Try plain nn.Module state_dict format.
        payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            model = _MLPTransition(
                in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, nlayer=nlayer
            )
            model.load_state_dict(payload["model_state_dict"])
            model.eval()
            return model

        if isinstance(payload, dict) and "state_dict" in payload:
            model = _MLPTransition(
                in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, nlayer=nlayer
            )
            model.load_state_dict(payload["state_dict"])
            model.eval()
            return model

        if isinstance(payload, torch.nn.Module):
            payload.eval()
            return payload

        raise RuntimeError(
            "Unsupported surrogate checkpoint format. Expected TorchScript or a dict "
            "with model_state_dict/state_dict."
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == u.shape[0]
        xu = torch.cat((x, u), dim=1)
        model = self.model.to(x.device)
        pred = model(xu)
        if self.predict_delta:
            return x + pred
        return pred

    def linearized_dynamics(self, x: torch.Tensor, u: torch.Tensor):
        """Compute Jacobians A,B by autograd for EKF/Lyapunov utilities."""
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype
        x_dim = x.shape[1]
        u_dim = u.shape[1]

        # Output dim should follow the transition output, which matches the
        # physical state dimension expected by observer/EKF.
        with torch.enable_grad():
            yi_probe = self.forward(
                x[0:1].clone().detach().requires_grad_(True),
                u[0:1].clone().detach().requires_grad_(True),
            )
        y_dim = yi_probe.shape[1]

        A = torch.zeros((batch_size, y_dim, x_dim), device=device, dtype=dtype)
        B = torch.zeros((batch_size, y_dim, u_dim), device=device, dtype=dtype)

        for i in range(batch_size):
            xi = x[i : i + 1].clone().detach().requires_grad_(True)
            ui = u[i : i + 1].clone().detach().requires_grad_(True)
            yi = self.forward(xi, ui)
            for j in range(y_dim):
                grads = torch.autograd.grad(
                    yi[0, j],
                    [xi, ui],
                    retain_graph=(j < y_dim - 1),
                    allow_unused=False,
                )
                A[i, j, :] = grads[0][0]
                B[i, j, :] = grads[1][0]
        return A, B

    @property
    def x_equilibrium(self):
        return self.continuous_time_system.x_equilibrium

    @property
    def u_equilibrium(self):
        return self.continuous_time_system.u_equilibrium


def _surrogate_ctor_from_cfg(cfg: DictConfig):
    configured_path = OmegaConf.select(cfg, "model.mujoco_surrogate.path", default=None)
    candidates = []
    if configured_path:
        candidates.append(Path(configured_path))
    candidates.extend(
        [
            PROJECT_ROOT / "models" / "mujoco_surrogate" / "pendulum_mujoco_surrogate_dynamics.pth",
            PROJECT_ROOT / "models" / "mujoco_surrogate_dynamics.pth",
        ]
    )

    surrogate_path = None
    for path in candidates:
        if path.exists():
            surrogate_path = str(path)
            break

    if surrogate_path is None:
        checked = ", ".join(str(p) for p in candidates)
        raise ValueError(
            "Could not find surrogate checkpoint. Checked: "
            f"{checked}. Pass +model.mujoco_surrogate.path=<path>."
        )

    predict_delta = OmegaConf.select(
        cfg, "model.mujoco_surrogate.predict_delta", default=None
    )
    hidden_dim = OmegaConf.select(cfg, "model.mujoco_surrogate.hidden_dim", default=None)
    nlayer = OmegaConf.select(cfg, "model.mujoco_surrogate.nlayer", default=None)

    # Auto-infer architecture/options from checkpoint metadata when available.
    ckpt_meta = None
    try:
        payload = torch.load(str(surrogate_path), map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            ckpt_meta = payload.get("meta", None)
    except Exception:
        ckpt_meta = None

    if hidden_dim is None:
        if isinstance(ckpt_meta, dict) and "hidden_dim" in ckpt_meta:
            hidden_dim = int(ckpt_meta["hidden_dim"])
        else:
            hidden_dim = 64
    else:
        hidden_dim = int(hidden_dim)

    if nlayer is None:
        if isinstance(ckpt_meta, dict) and "nlayer" in ckpt_meta:
            nlayer = int(ckpt_meta["nlayer"])
        else:
            nlayer = 3
    else:
        nlayer = int(nlayer)

    if predict_delta is None:
        if isinstance(ckpt_meta, dict) and "predict_delta" in ckpt_meta:
            predict_delta = bool(ckpt_meta["predict_delta"])
        else:
            predict_delta = False
    else:
        predict_delta = bool(predict_delta)

    def _ctor(continuous_time_system, dt: float, *args, **kwargs):
        return MuJoCoSurrogateSecondOrderDiscreteTimeSystem(
            continuous_time_system=continuous_time_system,
            dt=dt,
            surrogate_path=surrogate_path,
            predict_delta=predict_delta,
            hidden_dim=hidden_dim,
            nlayer=nlayer,
            *args,
            **kwargs,
        )

    return _ctor


@hydra.main(version_base=None, config_path="./config", config_name="mujoco_feedback")
def main(cfg: DictConfig):
    """
    Entry point that keeps output_feedback pipeline intact while allowing
    a separate MuJoCo-surrogate dynamics branch.

    Usage examples:
    - Analytic dynamics (default):
      python apps/pendulum/mujoco_feedback.py

    - MuJoCo surrogate branch:
      python apps/pendulum/mujoco_feedback.py \
        +model.dynamics_backend=mujoco_surrogate \
                +model.mujoco_surrogate.path=models/mujoco_surrogate/pendulum_mujoco_surrogate_dynamics.pth
    """
    backend = OmegaConf.select(cfg, "model.dynamics_backend", default="analytic")
    original_ctor = dynamical_system.SecondOrderDiscreteTimeSystem

    if backend == "mujoco_surrogate":
        dynamical_system.SecondOrderDiscreteTimeSystem = _surrogate_ctor_from_cfg(cfg)
        print("[mujoco_feedback] Using MuJoCo surrogate dynamics backend.")
    else:
        print("[mujoco_feedback] Using analytic dynamics backend.")

    try:
        # Call the original undecorated function with this Hydra cfg.
        output_feedback.main.__wrapped__(cfg)
    finally:
        # Always restore the original constructor.
        dynamical_system.SecondOrderDiscreteTimeSystem = original_ctor

    # Keep a dedicated artifact for MuJoCo-based training without overwriting the
    # canonical output_feedback checkpoint naming convention.
    canonical_checkpoint = PROJECT_ROOT / "models" / "pendulum_output_feedback.pth"
    mujoco_checkpoint = Path(
        OmegaConf.select(
            cfg,
            "model.mujoco_feedback.save_path",
            default=str(PROJECT_ROOT / "models" / "mujoco_feedback" / "pendulum_mujoco_feedback.pth"),
        )
    )
    if canonical_checkpoint.exists():
        mujoco_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(canonical_checkpoint, mujoco_checkpoint)
        print(f"[mujoco_feedback] Dedicated checkpoint saved to: {mujoco_checkpoint}")


if __name__ == "__main__":
    _dispatch_subcommand_if_requested()
    main()
