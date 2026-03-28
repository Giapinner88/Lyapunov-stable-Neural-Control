import argparse
from pathlib import Path

import torch
import torch.nn as nn

import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.train_utils as train_utils


class CartPoleDynamicsStd(nn.Module):
    """Reference cartpole continuous dynamics with explicit nx/nu/equilibrium API."""

    def __init__(self, mc: float = 10.0, mp: float = 1.0, l: float = 1.0, gravity: float = 9.81):
        super().__init__()
        self.mc = float(mc)
        self.mp = float(mp)
        self.l = float(l)
        self.gravity = float(gravity)
        self.nx = 4
        self.nu = 1

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # State: [px, theta, px_dot, theta_dot]
        px_dot = x[:, 2]
        theta_dot = x[:, 3]
        s = torch.sin(x[:, 1])
        c = torch.cos(x[:, 1])
        px_ddot = (
            u.squeeze(1) + self.mp * s * (self.l * theta_dot**2 + self.gravity * c)
        ) / (self.mp * s**2 + self.mc)
        theta_ddot = (
            -u.squeeze(1) * c
            - self.mp * self.l * theta_dot**2 * c * s
            - (self.mc + self.mp) * self.gravity * s
        ) / (self.l * self.mc + self.mp * s**2)
        return torch.cat(
            (
                px_dot.unsqueeze(1),
                theta_dot.unsqueeze(1),
                px_ddot.unsqueeze(1),
                theta_ddot.unsqueeze(1),
            ),
            dim=1,
        )

    @property
    def x_equilibrium(self):
        return torch.zeros((4,), dtype=torch.float32)

    @property
    def u_equilibrium(self):
        return torch.zeros((1,), dtype=torch.float32)


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    train_utils.set_seed(int(args.seed))

    dt = float(args.dt)
    u_max = float(args.u_max)
    limit = torch.tensor([args.limit] * 4, device=device, dtype=dtype)
    lower_limit = -limit
    upper_limit = limit

    dynamics_ct = CartPoleDynamicsStd(mc=10.0, mp=1.0, l=1.0, gravity=9.81)
    dynamics = dynamical_system.FirstOrderDiscreteTimeSystem(
        dynamics_ct,
        dt=dt,
        integration=dynamical_system.IntegrationMethod.ExplicitEuler,
    )

    controller = controllers.NeuralNetworkController(
        nlayer=2,
        in_dim=4,
        out_dim=1,
        hidden_dim=64,
        clip_output="clamp",
        u_lo=torch.tensor([-u_max], dtype=dtype),
        u_up=torch.tensor([u_max], dtype=dtype),
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
        rho_multiplier=float(args.rho_multiplier),
        kappa=float(args.kappa),
        hard_max=True,
    )

    dynamics.to(device)
    controller.to(device)
    lyapunov_nn.to(device)
    derivative_lyaloss.to(device)

    grid_size = torch.tensor([20, 20, 20, 20], device=device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "cartpole_lyaloss_std.pth"

    candidate_roa_states = train_utils.get_candidate_roa_states(
        V=lyapunov_nn,
        rho=None,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        box_scale=0.4,
    )

    train_utils.train_lyapunov_with_buffer(
        derivative_lyaloss=derivative_lyaloss,
        positivity_lyaloss=None,
        observer_loss=None,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        grid_size=grid_size,
        learning_rate=float(args.lr),
        lr_controller=float(args.lr),
        weight_decay=0.0,
        max_iter=int(args.max_iter),
        enable_wandb=False,
        derivative_ibp_ratio=0.0,
        derivative_sample_ratio=1.0,
        positivity_ibp_ratio=0.0,
        positivity_sample_ratio=0.0,
        save_best_model=str(save_path),
        pgd_steps=int(args.pgd_steps),
        buffer_size=int(args.buffer_size),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        samples_per_iter=int(args.samples_per_iter),
        l1_reg=float(args.l1_reg),
        num_samples_per_boundary=int(args.boundary_samples),
        V_decrease_within_roa=True,
        Vmin_x_boundary_weight=0.0,
        Vmax_x_boundary_weight=0.0,
        candidate_roa_states=candidate_roa_states,
        candidate_roa_states_weight=float(args.candidate_roa_weight),
        always_candidate_roa_regulizer=False,
    )

    with torch.no_grad():
        rho = derivative_lyaloss.get_rho().item()
        x_test = (torch.rand((args.eval_samples, 4), device=device, dtype=dtype) - 0.5) * (2.0 * limit)
        loss_test = derivative_lyaloss(x_test)
        frac_viol = (loss_test < 0).float().mean().item()

    summary_path = out_dir / "cartpole_std_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                "Reference CartPole Standard Pipeline",
                f"device={device}",
                f"rho={rho:.8f}",
                f"fraction_violation={frac_viol:.8f}",
                f"checkpoint={save_path}",
            ]
        ),
        encoding="utf-8",
    )
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reference CartPole standard training pipeline")
    parser.add_argument("--output-dir", type=str, default="outputs/cartpole_std")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--u-max", type=float, default=30.0)
    parser.add_argument("--limit", type=float, default=1.0)
    parser.add_argument("--rho-multiplier", type=float, default=0.9)
    parser.add_argument("--kappa", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--samples-per-iter", type=int, default=2048)
    parser.add_argument("--pgd-steps", type=int, default=60)
    parser.add_argument("--boundary-samples", type=int, default=512)
    parser.add_argument("--candidate-roa-weight", type=float, default=0.2)
    parser.add_argument("--l1-reg", type=float, default=1e-3)
    parser.add_argument("--eval-samples", type=int, default=20000)
    run(parser.parse_args())
