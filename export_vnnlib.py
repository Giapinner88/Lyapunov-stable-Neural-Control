#!/usr/bin/env python3
"""
Export CartPole Lyapunov verification problem to VNNLIB format for alpha-beta-CROWN.

This script:
1. Loads trained controller and Lyapunov checkpoints
2. Exports model to ONNX (single-output: ΔV - M*ReLU(V-ρ))
3. Generates VNNLIB spec with input domain + unsafe set
4. Creates YAML config for abcrown.py invocation
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.onnx
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.models import NeuralLyapunov, NeuralController
from core.dynamics import CartpoleDynamics
from core.verification import CartpoleLevelsetImplicationGraph


def export_vnnlib(
    controller_path: Path,
    lyapunov_path: Path,
    output_dir: Path,
    alpha_lyap: float = 0.01,
    rho: float = None,
    implication_m: float = 100.0,
    x_min: np.ndarray = None,
    x_max: np.ndarray = None,
):
    """
    Export CartPole verification to ONNX + VNNLIB + YAML.
    
    Args:
        controller_path: Path to trained controller checkpoint
        lyapunov_path: Path to trained Lyapunov checkpoint
        output_dir: Where to save ONNX/VNNLIB/YAML
        alpha_lyap: Exponential decay rate (default 0.01)
        rho: Level-set radius (if None, estimated from forward pass)
        implication_m: Big-M coefficient for implication encoding
        x_min, x_max: Input domain bounds (if None, uses cartpole defaults)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default CartPole bounds
    if x_min is None:
        x_min = np.array([-2.4, -8.0, -(np.pi), -8.0], dtype=np.float32)
    if x_max is None:
        x_max = np.array([2.4, 8.0, np.pi, 8.0], dtype=np.float32)
    
    x_min = np.asarray(x_min, dtype=np.float32)
    x_max = np.asarray(x_max, dtype=np.float32)
    nx = len(x_min)
    
    print(f"[Export] Loading controller from {controller_path}")
    controller_ckpt = torch.load(controller_path, map_location="cpu")
    controller = NeuralController(nx=4, nu=1, hidden_sizes=[64, 64], u_bound=6.0)
    controller.load_state_dict(controller_ckpt)
    controller.eval()
    
    print(f"[Export] Loading lyapunov from {lyapunov_path}")
    lyapunov_ckpt = torch.load(lyapunov_path, map_location="cpu")
    lyapunov = NeuralLyapunov(nx=4, hidden_sizes=[64, 64], eps=0.01)
    lyapunov.load_state_dict(lyapunov_ckpt)
    lyapunov.eval()
    
    # Create dynamics
    dynamics = CartpoleDynamics(
        dt=0.05,
        max_force=6.0,
        position_integration="midpoint"
    )
    
    # Estimate rho from center + small samples if not provided
    if rho is None:
        print("[Export] Estimating rho from forward pass...")
        with torch.no_grad():
            x_samples = np.random.uniform(x_min * 0.1, x_max * 0.1, size=(100, nx)).astype(np.float32)
            x_t = torch.from_numpy(x_samples)
            v_vals = lyapunov(x_t)
            rho = float(v_vals.max().item()) * 1.5
        print(f"[Export] Estimated rho = {rho:.6f}")
    
    # Create verification graph
    graph = CartpoleLevelsetImplicationGraph(
        controller=controller,
        lyapunov=lyapunov,
        dynamics=dynamics,
        alpha_lyap=alpha_lyap,
        rho=rho,
        implication_m=implication_m
    )
    graph.eval()
    
    # Export ONNX
    dummy_x = torch.randn(1, nx)
    onnx_path = output_dir / "cartpole_verification.onnx"
    print(f"[Export] Exporting ONNX to {onnx_path}")
    torch.onnx.export(
        graph, dummy_x,
        str(onnx_path),
        input_names=["X"],
        output_names=["Y"],
        opset_version=12,
        do_constant_folding=True
    )
    
    # Generate VNNLIB
    vnnlib_path = output_dir / "cartpole_verification.vnnlib"
    print(f"[Export] Generating VNNLIB to {vnnlib_path}")
    
    vnnlib_content = []
    vnnlib_content.append("; CartPole Lyapunov Verification (Implication Formulation)")
    vnnlib_content.append("; V(x) level-set with implication: ΔV(x) - M*ReLU(V(x)-ρ) < 0")
    vnnlib_content.append(f"; rho={rho:.6f}, M={implication_m:.1f}, alpha={alpha_lyap:.4f}")
    vnnlib_content.append("")
    
    # Declare inputs
    for i in range(nx):
        vnnlib_content.append(f"(declare-const X_{i} Real)")
    
    # Declare output
    vnnlib_content.append(f"(declare-const Y_0 Real)")
    vnnlib_content.append("")
    
    # Input constraints
    vnnlib_content.append("; Input domain constraints")
    for i in range(nx):
        vnnlib_content.append(f"(assert (>= X_{i} {x_min[i]:.6f}))")
        vnnlib_content.append(f"(assert (<= X_{i} {x_max[i]:.6f}))")
    
    vnnlib_content.append("")
    vnnlib_content.append("; Unsafe: Y_0 >= 0")
    vnnlib_content.append("; (We want to prove: Y_0 < 0 for all inputs in domain)")
    vnnlib_content.append("(assert (>= Y_0 0.0))")
    
    with open(vnnlib_path, "w") as f:
        f.write("\n".join(vnnlib_content))
    
        # Use existing cartpole_verification.yaml if available
        existing_yaml = PROJECT_ROOT / "cartpole_verification.yaml"
        if existing_yaml.exists():
                print(f"[Export] Found existing YAML config at {existing_yaml}")
                yaml_path = existing_yaml
        else:
                yaml_path = output_dir / "cartpole_config.yaml"
                print(f"[Export] Generating YAML config to {yaml_path}")
                yaml_content = f"""# alpha-beta-CROWN configuration for CartPole Lyapunov verification
model:
    onnx_model_path: {onnx_path.name}

specification:
    vnnlib_path: {vnnlib_path.name}

solver:
    batch_size: 64
    auto_enlarge_batch_size: true
    max_subproblems_num: 5000000

bounds:
    init_method: ibp
    method: alpha-beta-CROWN

# Early verification
verification:
    max_iterations: 100
    timeout: 1000
  
# BaB pruning
pruning:
    method: auto
"""
                with open(yaml_path, "w") as f:
                        f.write(yaml_content)
    
    print("")
    print("=" * 80)
    print("EXPORT COMPLETE")
    print("=" * 80)
    print(f"ONNX model:       {onnx_path}")
    print(f"VNNLIB spec:      {vnnlib_path}")
    print(f"YAML config:      {yaml_path}")
    print(f"Input domain:     [{x_min}, {x_max}]")
    print(f"Level-set rho:    {rho:.6f}")
    print(f"Implication M:    {implication_m:.1f}")
    print(f"Decay rate α:     {alpha_lyap:.4f}")
    print("")
    print("Next step: Run alpha-beta-CROWN")
    print(f"  cd {output_dir.parent / 'alpha-beta-CROWN' / 'complete_verifier'}")
    print(f"  python abcrown.py --config {yaml_path.name}")
    print("=" * 80)
    
    return {
        "onnx_path": onnx_path,
        "vnnlib_path": vnnlib_path,
        "yaml_path": yaml_path,
        "rho": rho,
        "implication_m": implication_m,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CartPole verification to VNNLIB for BaB")
    parser.add_argument("--controller", type=Path, default=Path("checkpoints/cartpole/cartpole_controller.pth"))
    parser.add_argument("--lyapunov", type=Path, default=Path("checkpoints/cartpole/cartpole_lyapunov.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("verification_results/bab_artifacts"))
    parser.add_argument("--alpha-lyap", type=float, default=0.01)
    parser.add_argument("--rho", type=float, default=None, help="Level-set radius (auto-estimate if not provided)")
    parser.add_argument("--implication-m", type=float, default=100.0)
    
    args = parser.parse_args()
    
    export_vnnlib(
        controller_path=args.controller,
        lyapunov_path=args.lyapunov,
        output_dir=args.output_dir,
        alpha_lyap=args.alpha_lyap,
        rho=args.rho,
        implication_m=args.implication_m,
    )
