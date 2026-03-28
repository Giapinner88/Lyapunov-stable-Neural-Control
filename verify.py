"""
Post-training verification script for CartPole Lyapunov-stable control.

Usage:
    python verify.py --system cartpole --controller checkpoints/cartpole/cartpole_controller.pth --lyapunov checkpoints/cartpole/cartpole_lyapunov.pth
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
import subprocess
import sys
from typing import Dict

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from neural_lyapunov_training.runtime_utils import box_tensors, choose_device, load_trained_system
from neural_lyapunov_training.verification import (
    BisectionVerifier,
    CartpoleLyapunovLevelsetGraph,
    CrownRadiusVerifier,
    create_cartpole_verification_result,
)
from neural_lyapunov_training.roa_utils import compute_rho_boundary, estimate_roa_size


def format_ratio_percent(ratio: float) -> str:
    pct = ratio * 100.0
    if pct >= 0.01:
        return f"{pct:.2f}%"
    if pct >= 1e-6:
        return f"{pct:.6f}%"
    return f"{pct:.3e}%"


def format_volume(volume: float) -> str:
    if abs(volume) >= 1e-4:
        return f"{volume:.6f}"
    return f"{volume:.3e}"


def ratio_display_with_sampling_limit(ratio: float, n_samples: int) -> str:
    if ratio > 0.0:
        return format_ratio_percent(ratio)
    min_detectable_pct = 100.0 / max(1, int(n_samples))
    return f"<{min_detectable_pct:.6f}% (0/{n_samples} hits)"


def volume_display_with_sampling_limit(volume: float, box_volume: float, ratio: float, n_samples: int) -> str:
    if ratio > 0.0:
        return format_volume(volume)
    min_detectable_volume = box_volume / max(1, int(n_samples))
    if min_detectable_volume >= 1e-4:
        return f"<{min_detectable_volume:.6f}"
    return f"<{min_detectable_volume:.3e}"


def build_bab_artifacts(
    controller: nn.Module,
    lyapunov: nn.Module,
    dynamics: nn.Module,
    alpha_lyap: float,
    rho: float,
    tolerance: float,
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    output_dir: Path,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = CartpoleLyapunovLevelsetGraph(
        controller=controller,
        lyapunov=lyapunov,
        dynamics=dynamics,
        alpha_lyap=alpha_lyap,
    ).to(torch.device("cpu"))
    graph.eval()

    onnx_path = output_dir / "cartpole_lyapunov_in_levelset.onnx"
    vnnlib_path = output_dir / "cartpole_lyapunov_in_levelset.vnnlib"
    yaml_path = output_dir / "cartpole_lyapunov_in_levelset.yaml"

    dummy_x = torch.zeros(1, dynamics.nx, dtype=torch.float32)
    torch.onnx.export(
        graph,
        dummy_x,
        str(onnx_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["X"],
        output_names=["Y"],
        dynamic_axes={"X": {0: "batch_size"}, "Y": {0: "batch_size"}},
    )

    with open(vnnlib_path, "w", encoding="utf-8") as f:
        for i in range(dynamics.nx):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("(declare-const Y_0 Real)\n")
        f.write("(declare-const Y_1 Real)\n")
        for i in range(dynamics.nx):
            f.write(f"(declare-const Y_{2 + i} Real)\n")
        f.write("\n")

        for i in range(dynamics.nx):
            f.write(f"(assert (>= X_{i} {float(x_min[i]):.10f}))\n")
            f.write(f"(assert (<= X_{i} {float(x_max[i]):.10f}))\n")

        f.write("\n")
        f.write("(assert (or\n")
        f.write(f"  (and (<= Y_0 -{tolerance:.10f}))\n")
        for i in range(dynamics.nx):
            f.write(f"  (and (<= Y_{2 + i} {float(x_min[i]) - tolerance:.10f}))\n")
            f.write(f"  (and (>= Y_{2 + i} {float(x_max[i]) + tolerance:.10f}))\n")
        f.write("))\n")
        f.write(f"(assert (<= Y_1 {rho:.10f}))\n")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("model:\n")
        f.write(f"  onnx_model_path: {onnx_path.resolve()}\n")
        f.write("specification:\n")
        f.write(f"  vnnlib_path: {vnnlib_path.resolve()}\n")
        f.write("solver:\n")
        f.write("  batch_size: 1024\n")

    return {
        "onnx_path": str(onnx_path.resolve()),
        "vnnlib_path": str(vnnlib_path.resolve()),
        "yaml_path": str(yaml_path.resolve()),
    }


def run_bab_complete_verifier(config_path: str) -> Dict[str, str | int]:
    verifier_dir = Path("alpha-beta-CROWN/complete_verifier").resolve()
    abcrown_path = verifier_dir / "abcrown.py"
    if not abcrown_path.exists():
        raise RuntimeError(f"abcrown.py not found at {abcrown_path}")

    cmd = [sys.executable, str(abcrown_path), "--config", config_path]
    proc = subprocess.run(
        cmd,
        cwd=str(verifier_dir),
        capture_output=True,
        text=True,
        check=False,
    )

    return {
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def verify_cartpole_roa(
    controller_path: str,
    lyapunov_path: str,
    output_dir: str = "./verification_results",
    alpha_lyap: float = 0.01,
    run_crown: bool = True,
    crown_eps_max: float = 1.0,
    crown_method: str = "CROWN",
    run_bab: bool = False,
    bab_rho: float | None = None,
    bab_tolerance: float = 1e-6,
    verbose: bool = True,
) -> Dict:
    """
    Verify CartPole Lyapunov-stable controller and find the maximum certified ROA.
    """
    
    device = choose_device("auto")
    if verbose:
        print(f"[Verify] Device: {device}")

    bundle = load_trained_system(
        controller_path,
        lyapunov_path,
        system_name="cartpole",
        device=device,
    )
    controller = bundle.controller
    lyapunov = bundle.lyapunov
    dynamics = bundle.dynamics
    config = bundle.config

    x_min, x_max = box_tensors(config, device=device, dtype=torch.float32)
    
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
    
    roa_samples = 10000
    roa_volume, roa_ratio = estimate_roa_size(
        lyapunov,
        rho_certified,
        x_min,
        x_max,
        n_samples=roa_samples,
        device=device,
    )
    box_volume = torch.prod(x_max - x_min).item()
    
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

    # Step 4: Formal local certificate with CROWN (optional)
    if run_crown:
        if verbose:
            print("\n[Step 4] Running CROWN local-radius verification...")
        try:
            crown_verifier = CrownRadiusVerifier(
                controller,
                lyapunov,
                dynamics,
                alpha_lyap=alpha_lyap,
                device=device,
            )
            certified_eps, crown_info = crown_verifier.bisection_search(
                eps_min=1e-4,
                eps_max=crown_eps_max,
                max_iterations=12,
                method=crown_method,
                verbose=verbose,
            )
            result["crown_local_certified_eps"] = float(certified_eps)
            result["crown_method"] = crown_method
            result["crown_history"] = crown_info.get("history", [])
        except RuntimeError as exc:
            result["crown_error"] = str(exc)
            if verbose:
                print(f"[Step 4] Skip CROWN: {exc}")

    # Step 5: Complete verifier (alpha-beta-CROWN) with Branch-and-Bound
    if run_bab:
        if verbose:
            print("\n[Step 5] Running alpha-beta-CROWN complete verifier (BaB)...")
        try:
            rho_for_bab = float(rho_certified if bab_rho is None else bab_rho)
            artifact_dir = Path(output_dir) / "bab_artifacts"
            artifacts = build_bab_artifacts(
                controller=controller,
                lyapunov=lyapunov,
                dynamics=dynamics,
                alpha_lyap=alpha_lyap,
                rho=rho_for_bab,
                tolerance=float(bab_tolerance),
                x_min=x_min,
                x_max=x_max,
                output_dir=artifact_dir,
            )
            bab_run = run_bab_complete_verifier(artifacts["yaml_path"])
            result["bab_artifacts"] = artifacts
            result["bab_rho"] = rho_for_bab
            result["bab_tolerance"] = float(bab_tolerance)
            result["bab_check_x_next"] = True
            result["bab_command"] = bab_run["command"]
            result["bab_returncode"] = bab_run["returncode"]
            result["bab_stdout_tail"] = bab_run["stdout"][-4000:]
            result["bab_stderr_tail"] = bab_run["stderr"][-4000:]
            if verbose:
                status = "success" if bab_run["returncode"] == 0 else "failed"
                print(f"[Step 5] complete_verifier finished with status={status}")
                print(f"[Step 5] artifacts: {artifacts}")
        except Exception as exc:
            result["bab_error"] = str(exc)
            if verbose:
                print(f"[Step 5] Skip BaB: {exc}")
    
    if verbose:
        print(f"\n[Results Summary]")
        print(f"  Empirical ρ: {rho_empirical:.6f}")
        print(f"  Verified ρ:  {rho_certified:.6f}")
        print(
            f"  ROA Ratio in Box: {ratio_display_with_sampling_limit(roa_ratio, roa_samples)} "
            f"(raw={roa_ratio:.8e})"
        )
        print(
            "  Estimated ROA Volume: "
            f"{volume_display_with_sampling_limit(roa_volume, box_volume, roa_ratio, roa_samples)}"
        )
        if "crown_local_certified_eps" in result:
            print(f"  CROWN Certified Local Radius (L_inf): {result['crown_local_certified_eps']:.6f}")
        elif "crown_error" in result:
            print(f"  CROWN status: {result['crown_error']}")

        if "bab_returncode" in result:
            print(f"  BaB return code: {result['bab_returncode']}")
        elif "bab_error" in result:
            print(f"  BaB status: {result['bab_error']}")
    
    # Save results
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Save summary
    summary_path = Path(output_dir) / "verification_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("CartPole Lyapunov-Stable Controller Verification\n")
        f.write("="*60 + "\n\n")
        f.write(f"Empirical rho: {rho_empirical:.6f}\n")
        f.write(f"Verified rho:  {rho_certified:.6f}\n")
        f.write(f"ROA Ratio in Box: {ratio_display_with_sampling_limit(roa_ratio, roa_samples)}\n")
        f.write(f"ROA Ratio Raw: {roa_ratio:.8e}\n")
        f.write(
            f"Estimated ROA Volume: "
            f"{volume_display_with_sampling_limit(roa_volume, box_volume, roa_ratio, roa_samples)}\n"
        )
        f.write(f"Estimated ROA Volume Raw: {roa_volume:.8e}\n")
        f.write(f"Box Limits: x in [{x_min.tolist()}, {x_max.tolist()}]\n")
        if "crown_local_certified_eps" in result:
            f.write(
                f"CROWN Local Certified Radius (L_inf): {result['crown_local_certified_eps']:.6f}"
                f" (method={result.get('crown_method', 'CROWN')})\n"
            )
        elif "crown_error" in result:
            f.write(f"CROWN status: {result['crown_error']}\n")

        if "bab_returncode" in result:
            f.write(f"BaB return code: {result['bab_returncode']}\n")
            f.write(f"BaB rho: {result.get('bab_rho', 0.0):.6f}\n")
            f.write(f"BaB tolerance: {result.get('bab_tolerance', 0.0):.3e}\n")
            f.write(f"BaB check_x_next: {result.get('bab_check_x_next', True)}\n")
            if "bab_artifacts" in result:
                f.write(f"BaB artifacts: {result['bab_artifacts']}\n")
        elif "bab_error" in result:
            f.write(f"BaB status: {result['bab_error']}\n")
    
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
        default=0.01,
        help="Lyapunov decrease rate",
    )
    parser.add_argument(
        "--skip-crown",
        action="store_true",
        help="Skip CROWN local-radius verification",
    )
    parser.add_argument(
        "--crown-eps-max",
        type=float,
        default=1.0,
        help="Upper radius bound for CROWN bisection",
    )
    parser.add_argument(
        "--crown-method",
        type=str,
        default="CROWN",
        choices=["CROWN", "alpha-CROWN"],
        help="Bound propagation method for CROWN local verification",
    )
    parser.add_argument(
        "--run-bab",
        action="store_true",
        help="Run alpha-beta-CROWN complete verifier with branch-and-bound",
    )
    parser.add_argument(
        "--bab-rho",
        type=float,
        default=None,
        help="Level-set rho for BaB implication spec (default: use certified rho from step 2)",
    )
    parser.add_argument("--bab-tolerance", type=float, default=1e-6)

    args = parser.parse_args()
    
    result = verify_cartpole_roa(
        args.controller,
        args.lyapunov,
        output_dir=args.output_dir,
        alpha_lyap=args.alpha_lyap,
        run_crown=not args.skip_crown,
        crown_eps_max=args.crown_eps_max,
        crown_method=args.crown_method,
        run_bab=args.run_bab,
        bab_rho=args.bab_rho,
        bab_tolerance=args.bab_tolerance,
        verbose=True,
    )
    
    print("\n[✓] Verification complete!")
