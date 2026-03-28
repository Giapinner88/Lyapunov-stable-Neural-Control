"""
Evaluation and closed-loop testing for CartPole controller.
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path
import sys
from typing import Tuple

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from neural_lyapunov_training.runtime_utils import box_tensors, choose_device, load_trained_system


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
    lyap_step_tol: float = 2e-3,
    lyap_pass_ratio: float = 0.95,
    min_v_drop_ratio: float = 0.15,
    final_state_tol: tuple[float, ...] = (0.10, 0.20, 0.15, 0.25),
    end_window_std_tol: tuple[float, ...] = (0.04, 0.08, 0.05, 0.10),
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
    
    dv = np.diff(v_values)
    v_decrease_ratio = float(np.mean(dv <= lyap_step_tol))
    v_is_decreasing = v_decrease_ratio >= lyap_pass_ratio
    v_drop_target = (1.0 - min_v_drop_ratio) * max(float(v_values[0]), 1e-8)
    v_drop_ok = float(v_values[-1]) <= v_drop_target

    # Stabilization for cartpole should be checked per-state, not by one scalar std over all dims.
    state_at_end = trajectory[-20:]
    end_std = np.std(state_at_end, axis=0)
    final_abs = np.abs(final_state)

    final_state_tol_arr = np.asarray(final_state_tol, dtype=np.float64)
    end_window_std_tol_arr = np.asarray(end_window_std_tol, dtype=np.float64)
    if final_state_tol_arr.shape[0] != final_state.shape[0]:
        raise ValueError("final_state_tol must have same dimension as state")
    if end_window_std_tol_arr.shape[0] != final_state.shape[0]:
        raise ValueError("end_window_std_tol must have same dimension as state")

    final_state_ok = bool(np.all(final_abs <= final_state_tol_arr))
    end_window_stable = bool(np.all(end_std <= end_window_std_tol_arr))
    is_stabilized = final_state_ok and end_window_stable
    is_converged = bool(v_is_decreasing and v_drop_ok and is_stabilized)
    
    if verbose:
        print(f"  Initial state: {x_init.cpu().numpy().flatten()}")
        print(f"  Final state: {final_state}")
        print(f"  Final distance (L2): {final_distance:.6f}")
        print(f"  V(x_init): {v_values[0]:.6f}, V(x_final): {v_values[-1]:.6f}")
        print(f"  V decrease ratio: {v_decrease_ratio:.3f}")
        print(f"  V is decreasing: {v_is_decreasing}")
        print(f"  V drop target met: {v_drop_ok}")
        print(f"  Stabilized: {is_stabilized}")
    
    return {
        "trajectory": trajectory,
        "controls": controls,
        "v_values": v_values,
        "final_distance": final_distance,
        "v_decrease_ratio": v_decrease_ratio,
        "v_decreasing": v_is_decreasing,
        "v_drop_ok": v_drop_ok,
        "is_converged": is_converged,
        "stabilized": is_stabilized,
        "final_state_ok": final_state_ok,
        "end_window_stable": end_window_stable,
        "x_final": final_state,
    }


def sample_initial_conditions(
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    n_tests: int,
    device: torch.device,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = float(scale)
    if not (0.0 < scale <= 1.0):
        raise ValueError("eval scale must be in (0, 1]")

    center = 0.5 * (x_min + x_max)
    span = (x_max - x_min) * scale
    lo = center - 0.5 * span
    hi = center + 0.5 * span
    x_inits = lo + torch.rand((n_tests, x_min.shape[0]), device=device) * (hi - lo)
    return x_inits, lo, hi


def batch_evaluation(
    controller: nn.Module,
    dynamics: nn.Module,
    lyapunov: nn.Module,
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    n_tests: int = 100,
    eval_scale: float = 0.4,
    device: torch.device = torch.device('cpu'),
    verbose: bool = False,
) -> dict:
    """
    Evaluate controller over a batch of random initial conditions.
    """
    
    x_inits, eval_lo, eval_hi = sample_initial_conditions(
        x_min=x_min,
        x_max=x_max,
        n_tests=n_tests,
        device=device,
        scale=eval_scale,
    )
    
    if verbose:
        print(f"\n[Evaluating] {n_tests} random initial conditions...")
    
    convergence_count = 0
    lyap_decrease_count = 0
    v_drop_count = 0
    stabilize_count = 0
    
    results_list = []
    
    for i, x_init in enumerate(x_inits):
        result = evaluate_convergence(
            controller, dynamics, lyapunov, x_init, n_steps=200, device=device
        )
        
        results_list.append(result)
        
        if result["v_decreasing"]:
            lyap_decrease_count += 1
        if result["v_drop_ok"]:
            v_drop_count += 1
        if result["stabilized"]:
            stabilize_count += 1
        if result["is_converged"]:
            convergence_count += 1
        
        if verbose and (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{n_tests}")
    
    return {
        "total_tests": n_tests,
        "eval_scale": float(eval_scale),
        "eval_lo": eval_lo.detach().cpu().tolist(),
        "eval_hi": eval_hi.detach().cpu().tolist(),
        "convergence_count": convergence_count,
        "convergence_rate": convergence_count / n_tests,
        "lyap_decrease_count": lyap_decrease_count,
        "lyap_decrease_rate": lyap_decrease_count / n_tests,
        "v_drop_count": v_drop_count,
        "v_drop_rate": v_drop_count / n_tests,
        "stabilize_count": stabilize_count,
        "stabilize_rate": stabilize_count / n_tests,
        "mean_v_decrease_ratio": float(np.mean([r["v_decrease_ratio"] for r in results_list])) if results_list else 0.0,
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
        "--eval-scale",
        type=float,
        default=0.4,
        help="Evaluate on centered box scaled from training box (0,1], default 0.4",
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
        eval_scale=args.eval_scale,
        device=device,
        verbose=True,
    )
    
    # Print summary
    print(f"\n[Summary]")
    print(f"  Eval scale: {results['eval_scale']:.2f}")
    print(f"  Eval range: [{results['eval_lo']}, {results['eval_hi']}]")
    print(f"  Convergence rate: {results['convergence_rate']:.2%}")
    print(f"  Lyapunov decrease rate: {results['lyap_decrease_rate']:.2%}")
    print(f"  Lyapunov drop rate: {results['v_drop_rate']:.2%}")
    print(f"  Stabilization rate: {results['stabilize_rate']:.2%}")
    print(f"  Mean Lyapunov decrease-step ratio: {results['mean_v_decrease_ratio']:.3f}")
    
    # Save results
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    summary_path = Path(args.output_dir) / "eval_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("CartPole Controller Evaluation Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total tests: {results['total_tests']}\n")
        f.write(f"Eval scale: {results['eval_scale']:.2f}\n")
        f.write(f"Eval range lo: {results['eval_lo']}\n")
        f.write(f"Eval range hi: {results['eval_hi']}\n")
        f.write(f"Convergence rate: {results['convergence_rate']:.2%}\n")
        f.write(f"Lyapunov decrease rate: {results['lyap_decrease_rate']:.2%}\n")
        f.write(f"Lyapunov drop rate: {results['v_drop_rate']:.2%}\n")
        f.write(f"Stabilization rate: {results['stabilize_rate']:.2%}\n")
        f.write(f"Mean Lyapunov decrease-step ratio: {results['mean_v_decrease_ratio']:.3f}\n")
    
    print(f"\n[Output] Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
