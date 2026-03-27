"""
Evaluation and closed-loop testing for CartPole controller.
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path
import sys
from typing import Tuple, List

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.dynamics import CartpoleDynamics
from core.runtime_utils import box_tensors, choose_device, load_trained_system


def test_closed_loop_trajectory(
    controller: nn.Module,
    dynamics: nn.Module,
    x_init: torch.Tensor,
    n_steps: int = 200,
    device: torch.device = torch.device('cpu'),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a closed-loop trajectory from initial state x_init.
    
    Returns:
        (trajectory, controls) - Arrays of states and actions over time
    """
    controller.eval()
    dynamics.eval()
    
    trajectory = [x_init.cpu().numpy().flatten()]
    controls = []
    
    x_current = x_init.unsqueeze(0).to(device)
    
    for step in range(n_steps):
        with torch.no_grad():
            u = controller(x_current)
            x_next = dynamics.step(x_current, u)
        
        controls.append(u.cpu().numpy().flatten())
        trajectory.append(x_next.cpu().numpy().flatten())
        x_current = x_next
    
    return np.array(trajectory), np.array(controls)


def evaluate_convergence(
    controller: nn.Module,
    dynamics: nn.Module,
    lyapunov: nn.Module,
    x_init: torch.Tensor,
    n_steps: int = 200,
    device: torch.device = torch.device('cpu'),
    verbose: bool = False,
) -> dict:
    """
    Evaluate if controller converges from initial state x_init.
    """
    
    controller.eval()
    dynamics.eval()
    lyapunov.eval()
    
    trajectory, controls = test_closed_loop_trajectory(
        controller, dynamics, x_init, n_steps, device
    )
    
    # Compute Lyapunov values along trajectory
    v_values = []
    with torch.no_grad():
        for x in trajectory:
            x_torch = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
            v = lyapunov(x_torch).item()
            v_values.append(v)
    
    v_values = np.array(v_values)
    
    # Check convergence
    final_state = trajectory[-1]
    final_distance = np.linalg.norm(final_state)
    
    # Check Lyapunov is decreasing
    v_is_decreasing = np.all(np.diff(v_values) <= 1e-3)  # Small tolerance for numerical errors
    
    # Check stabilization
    state_at_end = trajectory[-10:]
    is_stabilized = np.std(state_at_end) < 0.05
    
    if verbose:
        print(f"  Initial state: {x_init.cpu().numpy().flatten()}")
        print(f"  Final state: {final_state}")
        print(f"  Final distance (L2): {final_distance:.6f}")
        print(f"  V(x_init): {v_values[0]:.6f}, V(x_final): {v_values[-1]:.6f}")
        print(f"  V is decreasing: {v_is_decreasing}")
        print(f"  Stabilized: {is_stabilized}")
    
    return {
        "trajectory": trajectory,
        "controls": controls,
        "v_values": v_values,
        "final_distance": final_distance,
        "v_decreasing": v_is_decreasing,
        "stabilized": is_stabilized,
        "x_final": final_state,
    }


def batch_evaluation(
    controller: nn.Module,
    dynamics: nn.Module,
    lyapunov: nn.Module,
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    n_tests: int = 100,
    device: torch.device = torch.device('cpu'),
    verbose: bool = False,
) -> dict:
    """
    Evaluate controller over a batch of random initial conditions.
    """
    
    # Sample initial conditions
    x_inits = x_min + torch.rand((n_tests, x_min.shape[0]), device=device) * (x_max - x_min)
    
    if verbose:
        print(f"\n[Evaluating] {n_tests} random initial conditions...")
    
    convergence_count = 0
    lyap_decrease_count = 0
    stabilize_count = 0
    
    results_list = []
    
    for i, x_init in enumerate(x_inits):
        result = evaluate_convergence(
            controller, dynamics, lyapunov, x_init, n_steps=200, device=device
        )
        
        results_list.append(result)
        
        if result["v_decreasing"]:
            lyap_decrease_count += 1
        if result["stabilized"]:
            stabilize_count += 1
        if result["v_decreasing"] and result["stabilized"]:
            convergence_count += 1
        
        if verbose and (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{n_tests}")
    
    return {
        "total_tests": n_tests,
        "convergence_count": convergence_count,
        "convergence_rate": convergence_count / n_tests,
        "lyap_decrease_count": lyap_decrease_count,
        "lyap_decrease_rate": lyap_decrease_count / n_tests,
        "stabilize_count": stabilize_count,
        "stabilize_rate": stabilize_count / n_tests,
        "results": results_list,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CartPole controller")
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
        "--n-tests",
        type=int,
        default=100,
        help="Number of test trajectories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    device = choose_device("auto")
    print(f"[Evaluate] Device: {device}")

    bundle = load_trained_system(
        args.controller,
        args.lyapunov,
        system_name="cartpole",
        device=device,
    )
    controller = bundle.controller
    lyapunov = bundle.lyapunov
    dynamics = bundle.dynamics
    config = bundle.config

    x_min, x_max = box_tensors(config, device=device, dtype=torch.float32)
    
    # Run evaluation
    print(f"\n[CartPole Evaluation]")
    print(f"Box constraints: x ∈ [{config.box.x_min}, {config.box.x_max}]")
    
    results = batch_evaluation(
        controller,
        dynamics,
        lyapunov,
        x_min,
        x_max,
        n_tests=args.n_tests,
        device=device,
        verbose=True,
    )
    
    # Print summary
    print(f"\n[Summary]")
    print(f"  Convergence rate: {results['convergence_rate']:.2%}")
    print(f"  Lyapunov decrease rate: {results['lyap_decrease_rate']:.2%}")
    print(f"  Stabilization rate: {results['stabilize_rate']:.2%}")
    
    # Save results
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    summary_path = Path(args.output_dir) / "eval_summary.txt"
    with open(summary_path, "w") as f:
        f.write("CartPole Controller Evaluation Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total tests: {results['total_tests']}\n")
        f.write(f"Convergence rate: {results['convergence_rate']:.2%}\n")
        f.write(f"Lyapunov decrease rate: {results['lyap_decrease_rate']:.2%}\n")
        f.write(f"Stabilization rate: {results['stabilize_rate']:.2%}\n")
    
    print(f"\n[Output] Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
