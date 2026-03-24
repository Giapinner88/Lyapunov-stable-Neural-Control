"""
Post-training verification script for CartPole Lyapunov-stable control.

Usage:
    python verify.py --system cartpole --controller checkpoints/cartpole/cartpole_controller.pth --lyapunov checkpoints/cartpole/cartpole_lyapunov.pth
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
from typing import Dict

from core.dynamics import CartpoleDynamics, PendulumDynamics
from core.models import NeuralController, NeuralLyapunov
from core.verification import BisectionVerifier, create_cartpole_verification_result
from core.roa_utils import compute_rho_boundary, estimate_roa_size
from core.training_config import get_default_config


def load_models(
    controller_path: str,
    lyapunov_path: str,
    system_name: str = "cartpole",
    device: torch.device = torch.device('cpu'),
):
    """Load pre-trained controller and Lyapunov function."""
    
    # Get configuration
    config = get_default_config(system_name)
    
    # Build dynamics
    if system_name == "cartpole":
        dynamics = CartpoleDynamics().to(device)
    elif system_name == "pendulum":
        dynamics = PendulumDynamics().to(device)
    else:
        raise ValueError(f"Unknown system: {system_name}")
    
    # Build models
    controller = NeuralController(
        nx=config.model.nx,
        nu=config.model.nu,
        u_bound=config.model.u_bound,
        state_limits=config.model.state_limits,
    ).to(device)
    
    lyapunov = NeuralLyapunov(
        nx=config.model.nx,
        state_limits=config.model.state_limits,
    ).to(device)
    
    # Load weights
    controller.load_state_dict(torch.load(controller_path, map_location=device))
    lyapunov.load_state_dict(torch.load(lyapunov_path, map_location=device))
    
    controller.eval()
    lyapunov.eval()
    dynamics.eval()
    
    return controller, lyapunov, dynamics, config


def verify_cartpole_roa(
    controller_path: str,
    lyapunov_path: str,
    output_dir: str = "./verification_results",
    alpha_lyap: float = 0.05,
    verbose: bool = True,
) -> Dict:
    """
    Verify CartPole Lyapunov-stable controller and find the maximum certified ROA.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[Verify] Device: {device}")
    
    # Load models
    controller, lyapunov, dynamics, config = load_models(
        controller_path,
        lyapunov_path,
        system_name="cartpole",
        device=device,
    )
    
    x_min = torch.tensor(config.box.x_min, device=device, dtype=torch.float32)
    x_max = torch.tensor(config.box.x_max, device=device, dtype=torch.float32)
    
    if verbose:
        print(f"[Verify] Box constraints: x ∈ [{x_min.tolist()}, {x_max.tolist()}]")
    
    # Step 1: Estimate empirical ρ from boundary sampling (heuristic)
    if verbose:
        print(f"\n[Step 1] Computing empirical ρ from boundary...")
    
    rho_empirical, _ = compute_rho_boundary(
        lyapunov,
        dynamics,
        controller,
        x_min,
        x_max,
        n_boundary_samples=1000,
        n_pgd_steps=50,
        gamma=0.9,
        device=device,
        verbose=verbose,
    )
    
    # Step 2: Run bisection verification to find certified ρ
    if verbose:
        print(f"\n[Step 2] Running bisection verification...")
    
    verifier = BisectionVerifier(
        controller,
        lyapunov,
        dynamics,
        alpha_lyap=alpha_lyap,
        device=device,
    )
    
    rho_certified, bisection_result = verifier.bisection_search(
        x_min,
        x_max,
        rho_min=1e-6,
        rho_max=rho_empirical * 2.0,  # Use empirical as upper bound
        max_iterations=10,
        n_samples=10000,
        verbose=verbose,
    )
    
    # Step 3: Estimate final ROA size
    if verbose:
        print(f"\n[Step 3] Computing final ROA statistics...")
    
    roa_volume, roa_ratio = estimate_roa_size(
        lyapunov,
        rho_certified,
        x_min,
        x_max,
        n_samples=10000,
        device=device,
    )
    
    # Create result
    result = create_cartpole_verification_result(
        rho_certified,
        bisection_result["history"],
        x_min,
        x_max,
    )
    
    result["rho_empirical"] = rho_empirical
    result["roa_volume"] = roa_volume
    result["roa_ratio"] = roa_ratio
    
    if verbose:
        print(f"\n[Results Summary]")
        print(f"  Empirical ρ: {rho_empirical:.6f}")
        print(f"  Verified ρ:  {rho_certified:.6f}")
        print(f"  ROA Ratio in Box: {roa_ratio:.2%}")
        print(f"  Estimated ROA Volume: {roa_volume:.4f}")
    
    # Save results
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Save summary
    summary_path = Path(output_dir) / "verification_summary.txt"
    with open(summary_path, "w") as f:
        f.write("="*60 + "\n")
        f.write("CartPole Lyapunov-Stable Controller Verification\n")
        f.write("="*60 + "\n\n")
        f.write(f"Empirical ρ: {rho_empirical:.6f}\n")
        f.write(f"Verified ρ:  {rho_certified:.6f}\n")
        f.write(f"ROA Ratio in Box: {roa_ratio:.2%}\n")
        f.write(f"Estimated ROA Volume: {roa_volume:.4f}\n")
        f.write(f"Box Limits: x ∈ [{x_min.tolist()}, {x_max.tolist()}]\n")
    
    if verbose:
        print(f"\n[Output] Summary saved to: {summary_path}")
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify CartPole Lyapunov-stable controller"
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="checkpoints/cartpole/cartpole_controller.pth",
        help="Path to controller weights",
    )
    parser.add_argument(
        "--lyapunov",
        type=str,
        default="checkpoints/cartpole/cartpole_lyapunov.pth",
        help="Path to Lyapunov function weights",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./verification_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--alpha-lyap",
        type=float,
        default=0.05,
        help="Lyapunov decrease rate",
    )
    
    args = parser.parse_args()
    
    result = verify_cartpole_roa(
        args.controller,
        args.lyapunov,
        output_dir=args.output_dir,
        alpha_lyap=args.alpha_lyap,
        verbose=True,
    )
    
    print("\n[✓] Verification complete!")
