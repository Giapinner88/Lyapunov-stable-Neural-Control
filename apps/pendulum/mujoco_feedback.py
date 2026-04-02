import sys
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

        self.model = self._load_surrogate(
            surrogate_path=surrogate_path,
            in_dim=self.nx + self.nu,
            out_dim=self.nx,
            hidden_dim=hidden_dim,
            nlayer=nlayer,
        )

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

        A = torch.zeros((batch_size, self.nx, self.nx), device=device, dtype=dtype)
        B = torch.zeros((batch_size, self.nx, self.nu), device=device, dtype=dtype)

        for i in range(batch_size):
            xi = x[i : i + 1].clone().detach().requires_grad_(True)
            ui = u[i : i + 1].clone().detach().requires_grad_(True)
            yi = self.forward(xi, ui)
            for j in range(self.nx):
                grads = torch.autograd.grad(
                    yi[0, j],
                    [xi, ui],
                    retain_graph=(j < self.nx - 1),
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
    surrogate_path = OmegaConf.select(cfg, "model.mujoco_surrogate.path", default=None)
    if not surrogate_path:
        raise ValueError(
            "model.mujoco_surrogate.path is required when using mujoco_surrogate backend"
        )

    predict_delta = bool(
        OmegaConf.select(cfg, "model.mujoco_surrogate.predict_delta", default=False)
    )
    hidden_dim = int(OmegaConf.select(cfg, "model.mujoco_surrogate.hidden_dim", default=64))
    nlayer = int(OmegaConf.select(cfg, "model.mujoco_surrogate.nlayer", default=3))

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


@hydra.main(version_base=None, config_path="./config", config_name="output_feedback")
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
        +model.mujoco_surrogate.path=models/mujoco_surrogate_dynamics.pt
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


if __name__ == "__main__":
    main()
