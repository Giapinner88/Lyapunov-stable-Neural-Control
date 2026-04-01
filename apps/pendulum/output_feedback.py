import os
import sys
import warnings
from pathlib import Path

# Allow running this file directly without requiring PYTHONPATH setup.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Suppress non-critical wandb warnings to keep training logs clean.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"pkg_resources is deprecated as an API.*",
)
warnings.filterwarnings(
    "ignore",
    category=ResourceWarning,
    message=r"Implicitly cleaning up <TemporaryDirectory '.*wandb.*'>",
)
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.simplefilter("ignore")

import hydra
import logging
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.pendulum as pendulum
import neural_lyapunov_training.output_train_utils as output_train_utils
import itertools
import scipy.linalg

import neural_lyapunov_training.train_utils as train_utils

device = torch.device("cuda")
dtype = torch.float


def linearize_pendulum(pendulum_continuous: pendulum.PendulumDynamics):
    x = torch.tensor([[0.0, 0.0]])
    x.requires_grad = True
    u = torch.tensor([[0.0]])
    u.requires_grad = True
    qddot = pendulum_continuous.forward(x, u)
    A = torch.empty((2, 2))
    B = torch.empty((2, 1))
    A[0, 0] = 0
    A[0, 1] = 1
    B[0, 0] = 0
    A[1], B[1] = torch.autograd.grad(qddot[0, 0], [x, u])
    return A, B


def compute_lqr(pendulum_continuous: pendulum.PendulumDynamics):
    A, B = linearize_pendulum(pendulum_continuous)
    A_np, B_np = A.detach().numpy(), B.detach().numpy()
    Q = np.eye(2)
    R = np.eye(1) * 100
    S = scipy.linalg.solve_continuous_are(A_np, B_np, Q, R)
    K = -np.linalg.solve(R, B_np.T @ S)
    return K, S


@hydra.main(version_base=None, config_path="./config", config_name="output_feedback")
def main(cfg: DictConfig):
    run_output_dir = Path(HydraConfig.get().runtime.output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(run_output_dir / "config.yaml"))

    train_utils.set_seed(cfg.seed)

    grid_size = torch.tensor([10, 10, 5, 5], device=device)

    dt = 0.01
    pendulum_continuous = pendulum.PendulumDynamics(m=0.15, l=0.5, beta=0.1)
    dynamics = dynamical_system.SecondOrderDiscreteTimeSystem(
        pendulum_continuous, dt=dt
    )
    nx = pendulum_continuous.nx
    dynamics.nx = 4
    dynamics.to(device)

    # Reference controller
    u_max = cfg.model.u_max  # mgl = 0.736
    u_nominal = output_train_utils.load_sos_controller(
        str(PROJECT_ROOT / "data/pendulum/output_feedback/sos_controller.pkl"),
        lambda x: x,
        2,
    )
    controller_nominal = lambda x: torch.clamp(u_nominal(x), min=-u_max, max=u_max)

    controller = controllers.NeuralNetworkController(
        nlayer=4,
        in_dim=3,
        out_dim=1,
        hidden_dim=8,
        clip_output="clamp",
        u_lo=torch.tensor([-u_max]),
        u_up=torch.tensor([u_max]),
        x_equilibrium=torch.zeros(3, dtype=dtype),
        u_equilibrium=pendulum_continuous.u_equilibrium,
    )
    controller.to(device)
    controller.load_state_dict(
        torch.load(
            str(PROJECT_ROOT / "data/pendulum/output_feedback/controller_[8, 8, 8].pth")
        )
    )
    controller.eval()

    h = lambda x: pendulum_continuous.h(x)
    # Reference EKF observer
    ekf_observer = controllers.EKFObserver(
        dynamics, h, gamma=0, delta=1e-3, lam=0.1, alpha=1.05
    )
    observer = controllers.NeuralNetworkLuenbergerObserver(
        nx,
        pendulum_continuous.ny,
        dynamics,
        h,
        torch.zeros(1, pendulum_continuous.ny),
        fc_hidden_dim=[8, 8],
    )
    observer.to(device)
    observer.load_state_dict(
        torch.load(
            str(PROJECT_ROOT / "data/pendulum/output_feedback/observer_[8, 8].pth")
        )
    )
    observer.eval()

    K, S = compute_lqr(pendulum_continuous)
    K_torch = torch.from_numpy(K).type(dtype).to(device)
    S_torch = torch.from_numpy(S).type(dtype).to(device)
    V_lqr = lambda x: torch.sum(x * (x @ S_torch), axis=1, keepdim=True)

    if cfg.model.lyapunov.quadratic:
        S_cl = torch.cat(
            (
                torch.cat((S_torch / 10, torch.zeros(nx, nx, device=device)), dim=1),
                torch.cat(
                    (
                        torch.zeros(nx, nx, device=device),
                        torch.linalg.inv(ekf_observer.P0.to(device)) / 50,
                    ),
                    dim=1,
                ),
            ),
            dim=0,
        )
        R = torch.linalg.cholesky(S_cl)
        lyapunov_nn = lyapunov.NeuralNetworkQuadraticLyapunov(
            goal_state=torch.zeros(4, dtype=dtype).to(device),
            x_dim=4,
            R_rows=4,
            eps=0.01,
            R=R,
        )

    else:
        lyapunov_hidden_widths = cfg.model.lyapunov.hidden_widths
        lyapunov_nn = lyapunov.NeuralNetworkLyapunov(
            goal_state=torch.tensor([0.0, 0.0, 0.0, 0.0]),
            hidden_widths=lyapunov_hidden_widths,
            x_dim=4,
            R_rows=2,
            absolute_output=True,
            eps=0.01,
            activation=nn.LeakyReLU,
            V_psd_form=cfg.model.V_psd_form,
        )
        lyapunov_nn.load_state_dict(
            torch.load(
                str(PROJECT_ROOT / "data/pendulum/output_feedback/lyapunov_init.pth")
            )
        )
        lyapunov_nn.eval()
    lyapunov_nn.to(device)

    def lyapunov_target(xe, r=0.01):
        x = xe[:, :nx]
        e = xe[:, nx:]
        Vc = V_lqr(x)
        Vo = (
            torch.einsum(
                "bi, ii, bi->b", e, torch.linalg.inv(ekf_observer.P0.to(e.device)), e
            )
            / 200
        )
        return r * Vc + Vo.unsqueeze(1)

    kappa = cfg.model.kappa
    hard_max = cfg.train.hard_max
    # Placeholder for loading models
    derivative_lyaloss = lyapunov.LyapunovDerivativeDOFLoss(
        dynamics,
        observer,
        controller,
        lyapunov_nn,
        0,
        0,
        1,
        kappa=kappa,
        hard_max=hard_max,
        beta=1,
        loss_weights=torch.tensor([0.5, 1.0, 0.5], device=device),
    )
    observer_loss = lyapunov.ObserverLoss(dynamics, observer, controller, ekf_observer)

    if cfg.model.load_lyaloss is not None:
        load_lyaloss = cfg.model.load_lyaloss
        derivative_lyaloss.load_state_dict(torch.load(load_lyaloss)["state_dict"])

    positivity_lyaloss = None

    save_lyaloss = cfg.model.save_lyaloss
    V_decrease_within_roa = cfg.model.V_decrease_within_roa
    save_lyaloss_path = None
    wandb_run = None
    wandb_enabled = False

    if wandb_enabled:
        import wandb

        wandb_run = wandb.init(
            project=cfg.train.wandb.project,
            entity=cfg.train.wandb.entity,
            name=cfg.train.wandb.name,
        )

    def _as_list(value):
        if OmegaConf.is_list(value):
            return list(value)
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _get_stage_value(value, idx):
        values = _as_list(value)
        return values[idx] if idx < len(values) else values[-1]

    limit_xe = torch.tensor([np.pi, np.pi, np.pi / 4, np.pi / 4], device=device)
    limit_scales = _as_list(cfg.model.limit_scale)
    rho_multipliers = _as_list(cfg.model.rho_multiplier)
    if cfg.train.train_lyaloss:
        permute_array = [[-1.0, 1.0]] * 4
        permute_array_torch = torch.tensor(
            list(itertools.product(*permute_array)), device=device
        )

        for n, limit_scale in enumerate(limit_scales):
            limit = limit_xe * limit_scale
            lower_limit = -limit
            upper_limit = limit
            candidate_roa_states = permute_array_torch * upper_limit
            V_candidate = lyapunov_target(candidate_roa_states)
            V_max = torch.max(V_candidate)
            candidate_roa_states = (
                candidate_roa_states
                / torch.sqrt(V_candidate / V_max)
                * cfg.loss.candidate_scale
            )

            rho_multiplier = _get_stage_value(rho_multipliers, n)
            derivative_lyaloss = lyapunov.LyapunovDerivativeDOFLoss(
                dynamics,
                observer,
                controller,
                lyapunov_nn,
                lower_limit,
                upper_limit,
                rho_multiplier,
                kappa=kappa,
                hard_max=hard_max,
                beta=1,
                loss_weights=torch.tensor([0.5, 1.0, 0.5], device=device),
            )

            save_name = f"lyaloss_{kappa}kappa_{limit_scale}_{u_max}.pth"
            if save_lyaloss:
                save_lyaloss_path = str(run_output_dir / save_name)

            train_utils.train_lyapunov_with_buffer(
                derivative_lyaloss=derivative_lyaloss,
                positivity_lyaloss=positivity_lyaloss,
                observer_loss=observer_loss,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                grid_size=grid_size,
                learning_rate=cfg.train.learning_rate,
                lr_controller=cfg.train.lr_controller,
                weight_decay=0.0,
                max_iter=_get_stage_value(cfg.train.max_iter, n),
                enable_wandb=wandb_enabled,
                derivative_ibp_ratio=cfg.loss.ibp_ratio_derivative,
                derivative_sample_ratio=cfg.loss.sample_ratio_derivative,
                positivity_ibp_ratio=cfg.loss.ibp_ratio_positivity,
                positivity_sample_ratio=cfg.loss.sample_ratio_positivity,
                save_best_model=save_lyaloss_path,
                pgd_steps=cfg.train.pgd_steps,
                buffer_size=cfg.train.buffer_size,
                batch_size=cfg.train.batch_size,
                epochs=cfg.train.epochs,
                samples_per_iter=cfg.train.samples_per_iter,
                l1_reg=_get_stage_value(cfg.loss.l1_reg, n),
                observer_ratio=_get_stage_value(cfg.loss.observer_ratio, n),
                Vmin_x_pgd_buffer_size=cfg.train.Vmin_x_pgd_buffer_size,
                V_decrease_within_roa=V_decrease_within_roa,
                Vmin_x_boundary_weight=cfg.loss.Vmin_x_boundary_weight,
                Vmax_x_boundary_weight=cfg.loss.Vmax_x_boundary_weight,
                candidate_roa_states=candidate_roa_states,
                candidate_roa_states_weight=_get_stage_value(
                    cfg.loss.candidate_roa_states_weight, n
                ),
                lr_scheduler=cfg.train.lr_scheduler,
                hard_max=cfg.train.hard_max,
                always_candidate_roa_regulizer=cfg.loss.always_candidate_roa_regulizer,
            )

    else:
        limit = limit_xe * limit_scales[-1]
        lower_limit = -limit
        upper_limit = limit
        rho_multiplier = rho_multipliers[-1]

    # "Verify" Lyapunov conditions with PGD attack
    derivative_lyaloss_check = lyapunov.LyapunovDerivativeDOFLoss(
        dynamics,
        observer,
        controller,
        lyapunov_nn,
        lower_limit,
        upper_limit,
        rho_multiplier=rho_multiplier,
        kappa=0e-3,
        hard_max=True,
    )
    x_min_boundary = train_utils.calc_V_extreme_on_boundary_pgd(
        lyapunov_nn,
        lower_limit,
        upper_limit,
        num_samples_per_boundary=1000,
        eps=limit,
        steps=100,
        direction="minimize",
    )
    if derivative_lyaloss.x_boundary is not None:
        derivative_lyaloss_check.x_boundary = torch.cat(
            (x_min_boundary, derivative_lyaloss.x_boundary), dim=0
        )
    else:
        derivative_lyaloss_check.x_boundary = x_min_boundary

    for seed in range(50):
        train_utils.set_seed(seed)
        if V_decrease_within_roa:
            x_min_boundary = train_utils.calc_V_extreme_on_boundary_pgd(
                lyapunov_nn,
                lower_limit,
                upper_limit,
                num_samples_per_boundary=1000,
                eps=limit,
                steps=100,
                direction="minimize",
            )
            if derivative_lyaloss.x_boundary is not None:
                derivative_lyaloss_check.x_boundary = torch.cat(
                    (x_min_boundary, derivative_lyaloss.x_boundary), dim=0
                )
            else:
                derivative_lyaloss_check.x_boundary = x_min_boundary
        x_check_start = (
            (
                torch.rand((50000, 4), device=device)
                - torch.full((4,), 0.5, device=device)
            )
            * limit
            * 2
        )
        adv_x = train_utils.pgd_attack(
            x_check_start,
            derivative_lyaloss_check,
            eps=limit,
            steps=cfg.pgd_verifier_steps,
            lower_boundary=lower_limit,
            upper_boundary=upper_limit,
            direction="minimize",
        ).detach()
        adv_lya = derivative_lyaloss_check(adv_x)
        adv_output = torch.clamp(-adv_lya, min=0.0)
        max_adv_violation = adv_output.max().item()
        print(
            f"pgd attack max violation {max_adv_violation}, total violation {adv_output.sum().item()}"
        )

    rho = derivative_lyaloss_check.get_rho().item()
    print("rho = ", rho)

    # Always save a checkpoint for the current run, even in verification-only mode.
    checkpoint_payload = {
        "state_dict": derivative_lyaloss_check.state_dict(),
        "controller_state_dict": controller.state_dict(),
        "observer_state_dict": observer.state_dict(),
        "lyapunov_state_dict": lyapunov_nn.state_dict(),
        "rho": rho,
    }

    model_filename = f"lyapunov_nn_kappa_{kappa}_rho_{rho:.6f}.pth"
    model_path = str(run_output_dir / model_filename)
    torch.save(checkpoint_payload, model_path)
    print(f"Model saved to: {model_path}")

    # Canonical checkpoint path for simulation/inference scripts.
    canonical_model_path = PROJECT_ROOT / "models" / "pendulum_output_feedback.pth"
    torch.save(checkpoint_payload, str(canonical_model_path))
    print(f"Canonical model updated: {canonical_model_path}")

    plot_dir = PROJECT_ROOT / "plots" / "pendulum_output_feedback"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Simulate the system
    x_max = limit[:2]
    e_max = limit[2:]
    x0 = (torch.rand((50, 2), device=device) - 0.5) * 2 * x_max
    e0 = (torch.rand((50, 2), device=device) - 0.5) * 2 * e_max

    # n_grid = 7
    # X1, X2 = torch.meshgrid(torch.linspace(x_min[0], x_max[0], n_grid, device=device),
    #                 torch.linspace(x_min[1], x_max[1], n_grid, device=device))
    # x0 = torch.vstack((X1.flatten(), X2.flatten())).transpose(0, 1)
    # e0 = (torch.rand((x0.shape[0], 2), device=device) - 0.5) * 2 * e_max

    z0 = x0 - e0
    x_traj, z_traj, V_traj = output_train_utils.simulate(
        derivative_lyaloss_check, 1400, x0, z0
    )
    e_traj = x_traj - z_traj
    idx = V_traj[100, :] <= rho
    V_traj = V_traj[:, idx]

    # Plot V trajectories with summary statistics for better readability.
    fig_v, ax_v = plt.subplots(figsize=(10, 6))
    V_traj_np = (
        V_traj.detach().cpu().numpy() if torch.is_tensor(V_traj) else np.asarray(V_traj)
    )
    time_axis = dt * np.arange(V_traj_np.shape[0])
    if V_traj_np.shape[1] > 0:
        ax_v.plot(
            time_axis,
            V_traj_np[:, :: max(1, V_traj_np.shape[1] // 20)],
            alpha=0.2,
        )
        V_mean = V_traj_np.mean(axis=1)
        V_median = np.median(V_traj_np, axis=1)
        ax_v.plot(time_axis, V_mean, color="tab:blue", linewidth=2.0, label="mean V")
        ax_v.plot(
            time_axis,
            V_median,
            color="tab:orange",
            linewidth=2.0,
            linestyle="--",
            label="median V",
        )
    ax_v.axhline(rho, color="tab:red", linestyle=":", linewidth=1.8, label=f"rho={rho:.4g}")
    ax_v.set_xlabel("time (s)")
    ax_v.set_ylabel("V(x)")
    ax_v.set_title("Lyapunov trajectories in simulation")
    ax_v.grid(alpha=0.3)
    ax_v.legend(loc="upper right")
    ax_v.text(
        0.01,
        0.98,
        (
            f"max violation={max_adv_violation:.3e}\n"
            f"total violation={adv_output.sum().item():.3e}\n"
            f"kept trajectories={V_traj_np.shape[1]}"
        ),
        transform=ax_v.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
    )
    fig_v.tight_layout()
    fig_v.savefig(plot_dir / f"V_traj_{kappa}.png", dpi=180)
    plt.close(fig_v)

    # Consistent font scaling (matching plot_phase.py)
    FONT_SCALE = 0.95
    def fs(size):
        return max(6, int(size * FONT_SCALE))

    # Helper to plot 2D Lyapunov slice with gradient expansion, vector field, trajectories
    def plot_lyapunov_2d_slice(
        ax, lyapunov_nn, lower_lim, upper_lim, traj_2d,
        label_indices, rho_val, phase_name, device, grid_sz=280
    ):
        """Plot 2D heatmap + ROA expansion + vector field + trajectories."""
        lim_lo = lower_lim[label_indices].cpu().numpy()
        lim_up = upper_lim[label_indices].cpu().numpy()
        xl, yl = lim_lo[0], lim_lo[1]
        xu, yu = lim_up[0], lim_up[1]

        # Grid and compute V
        x_ticks = torch.linspace(lower_lim[label_indices[0]], upper_lim[label_indices[0]], grid_sz, device=device)
        y_ticks = torch.linspace(lower_lim[label_indices[1]], upper_lim[label_indices[1]], grid_sz, device=device)
        grid_x, grid_y = torch.meshgrid(x_ticks, y_ticks, indexing='ij')
        X_flat = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

        # Reconstruct full 4D state
        X_full = torch.zeros(len(X_flat), 4, dtype=torch.float32, device=device)
        X_full[:, label_indices] = X_flat
        with torch.no_grad():
            V_flat = lyapunov_nn(X_full).squeeze()
        V_grid = V_flat.reshape(grid_sz, grid_sz).cpu().numpy()
        grid_x_np = grid_x.cpu().numpy()
        grid_y_np = grid_y.cpu().numpy()

        # Heatmap + iso-contours
        cf = ax.contourf(grid_x_np, grid_y_np, V_grid, levels=50, cmap='viridis', alpha=0.80)
        ax.contour(grid_x_np, grid_y_np, V_grid, levels=12, colors='white', linewidths=0.3, alpha=0.15)

        # Gradient ROA expansion (plasma colormap)
        rho_max = float(V_grid.max()) * 0.85
        N_LEVELS = 8
        rho_levels = np.linspace(rho_max / N_LEVELS, rho_max, N_LEVELS)
        cmap_exp = plt.cm.plasma
        for idx, r in enumerate(rho_levels):
            t = idx / (N_LEVELS - 1)
            color = cmap_exp(t)
            lw = 0.8 + t * 1.0
            alpha = 0.50 + t * 0.30
            ax.contour(grid_x_np, grid_y_np, V_grid, levels=[r], colors=[color],
                      linewidths=lw, alpha=alpha)

        # Selected rho boundary
        cs = ax.contour(grid_x_np, grid_y_np, V_grid, levels=[rho_val], colors='#ff4444',
                       linewidths=2.5, linestyles='--')
        ax.clabel(cs, fmt=rf'$\rho$ = {rho_val:.3f}', fontsize=fs(8),
                 inline=True, inline_spacing=3, colors='#ff4444')

        # Vector field (dynamics with u=0)
        # Note: dynamics is 2D (state), but we're in 4D space (state+error)
        # Only x-part evolves; error part converges exponentially
        n_arr = 12
        x_pts = np.linspace(xl, xu, n_arr)
        y_pts = np.linspace(yl, yu, n_arr)
        X_arr_full = []
        for xv in x_pts:
            for yv in y_pts:
                x_pt = torch.zeros(4, dtype=torch.float32, device=device)
                x_pt[label_indices] = torch.tensor([xv, yv], dtype=torch.float32, device=device)
                X_arr_full.append(x_pt)
        X_arr_full = torch.stack(X_arr_full)

        # Extract only the state part (first 2 dims for pendulum)
        X_state_only = X_arr_full[:, :2]
        u_zero = torch.zeros(len(X_arr_full), 1, dtype=torch.float32, device=device)
        with torch.no_grad():
            X_next_state = dynamics(X_state_only, u_zero)
        
        # For error part, assume exponential convergence (simplified)
        # Reconstruct full next state
        X_next_full = torch.zeros_like(X_arr_full)
        X_next_full[:, :2] = X_next_state
        X_next_full[:, 2:] = X_arr_full[:, 2:] * 0.95  # Simple decay for visualization
        
        dX = (X_next_full - X_arr_full)[:, label_indices].cpu().numpy()
        pts = X_arr_full[:, label_indices].cpu().numpy()
        mag = np.hypot(dX[:, 0], dX[:, 1]) + 1e-8
        ax.quiver(
            pts[:, 0], pts[:, 1],
            dX[:, 0] / mag * (xu - xl) / n_arr * 0.40,
            dX[:, 1] / mag * (yu - yl) / n_arr * 0.40,
            color='white', alpha=0.40, width=0.0025,
            headwidth=3.5, headlength=3.5, headaxislength=3.0,
            scale=1, scale_units='xy', angles='xy', zorder=4
        )

        # Overlay trajectories
        if traj_2d.shape[1] > 1:
            n_traj_show = min(20, traj_2d.shape[1])
            stride = max(1, traj_2d.shape[1] // n_traj_show)
            for i in range(0, traj_2d.shape[1], stride):
                traj_pt = traj_2d[:, i, :]
                traj_np = traj_pt.cpu().numpy() if torch.is_tensor(traj_pt) else np.asarray(traj_pt)
                ax.plot(traj_np[:, 0], traj_np[:, 1], color="#ffffff", linewidth=0.8, alpha=0.35, zorder=5)
                ax.plot(traj_np[0, 0], traj_np[0, 1], 'o', color="#D9FF00", markersize=2.5, alpha=0.5, zorder=6)

        # Equilibrium
        ax.plot(0, 0, '*', color='#ff3333', markersize=12, zorder=7,
               markeredgecolor='white', markeredgewidth=0.6)

        # Colorbar
        cbar = plt.colorbar(cf, ax=ax, pad=0.02, shrink=0.90)
        cbar.set_label(r'$V$', fontsize=fs(10))
        cbar.ax.tick_params(labelsize=fs(8))

        # Labels and title
        ax.set_xlabel(labels[label_indices[0]], fontsize=fs(11))
        ax.set_ylabel(labels[label_indices[1]], fontsize=fs(11))
        ax.set_title(
            rf"Lyapunov {phase_name} | $\rho$ = {rho_val:.4g} | $\kappa$ = {kappa}",
            fontsize=fs(12)
        )
        ax.set_xlim(xl, xu)
        ax.set_ylim(yl, yu)
        ax.grid(alpha=0.2, linestyle=':')
        ax.set_aspect('equal', adjustable='box')

    x_boundary = x_min_boundary[torch.argmin(lyapunov_nn(x_min_boundary))]
    labels = [r"$\theta$", r"$\dot \theta$", r"$e_\theta$", r"$e_{\dot \theta}$"]

    # Extract 2D trajectories
    x_traj_2d = x_traj[:, :, :2]  # theta, theta_dot
    e_traj_2d = e_traj[:, :, :2]  # e_theta, e_theta_dot

    for plot_idx in [[0, 1], [2, 3]]:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111)
        if plot_idx == [0, 1]:
            plot_lyapunov_2d_slice(ax, lyapunov_nn, lower_limit, upper_limit,
                                  x_traj_2d, plot_idx, rho, "state space", device, grid_sz=280)
        else:
            plot_lyapunov_2d_slice(ax, lyapunov_nn, lower_limit, upper_limit,
                                  e_traj_2d, plot_idx, rho, "error space", device, grid_sz=280)
        fig.tight_layout()
        fig.savefig(plot_dir / f"V_{kappa}_{str(plot_idx)}.png", dpi=180)
        plt.close(fig)

    print(f"Plots saved to {plot_dir}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
