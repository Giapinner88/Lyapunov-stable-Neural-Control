from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from .dynamics import CartpoleDynamics, PendulumDynamics
from .models import NeuralController, NeuralLyapunov
from .training_config import TrainerConfig, get_default_config


PathLike = Union[str, Path]


@dataclass
class LoadedSystemBundle:
    controller: nn.Module
    lyapunov: nn.Module
    dynamics: nn.Module
    config: TrainerConfig


def choose_device(device_opt: str = "auto") -> torch.device:
    if device_opt == "cpu":
        return torch.device("cpu")
    if device_opt == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def box_tensors(
    config: TrainerConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_min = torch.tensor(config.box.x_min, device=device, dtype=dtype)
    x_max = torch.tensor(config.box.x_max, device=device, dtype=dtype)
    return x_min, x_max


def load_trained_system(
    controller_path: PathLike,
    lyapunov_path: PathLike,
    system_name: str = "cartpole",
    device: torch.device = torch.device("cpu"),
) -> LoadedSystemBundle:
    config = get_default_config(system_name)

    if system_name == "cartpole":
        dynamics = CartpoleDynamics(
            max_force=config.model.u_bound,
            position_integration="midpoint",
        ).to(device)
    elif system_name == "pendulum":
        dynamics = PendulumDynamics().to(device)
    else:
        raise ValueError(f"Unknown system: {system_name}")

    controller = NeuralController(
        nx=config.model.nx,
        nu=config.model.nu,
        u_bound=config.model.u_bound,
        state_limits=config.model.state_limits,
    ).to(device)

    lyapunov = NeuralLyapunov(
        nx=config.model.nx,
        phi_dim=config.model.lyapunov_phi_dim,
        absolute_output=config.model.lyapunov_absolute_output,
        state_limits=config.model.state_limits,
    ).to(device)

    controller.load_state_dict(torch.load(str(controller_path), map_location=device))
    lyapunov.load_state_dict(torch.load(str(lyapunov_path), map_location=device))

    controller.eval()
    lyapunov.eval()
    dynamics.eval()

    return LoadedSystemBundle(
        controller=controller,
        lyapunov=lyapunov,
        dynamics=dynamics,
        config=config,
    )
