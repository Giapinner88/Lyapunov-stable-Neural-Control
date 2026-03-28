#!/usr/bin/env python3
"""
Complete Lyapunov verification pipeline: Training → Export → BaB Verification

This script orchestrates the full workflow:
1. Waits for training to complete (monitors train.py process)
2. Exports trained models to VNNLIB format
3. Runs alpha-beta-CROWN complete_verifier (BaB) for formal ROA certification
"""

import sys
import subprocess
import time
from pathlib import Path
import shutil
import os

PROJECT_ROOT = Path(__file__).parent
ALPHA_BETA_CROWN_DIR = PROJECT_ROOT / "alpha-beta-CROWN" / "complete_verifier"


def wait_for_training(poll_interval_sec=60, max_wait_sec=None):
    """
    Poll until train.py process completes.
    
    Args:
        poll_interval_sec: Check every N seconds
        max_wait_sec: Timeout (None = wait forever)
    """
    start_time = time.time()
    poll_count = 0
    
    print("[Pipeline] Waiting for training process to complete...")
    print(f"[Pipeline] Poll interval: {poll_interval_sec}s")
    if max_wait_sec:
        print(f"[Pipeline] Timeout: {max_wait_sec}s ({max_wait_sec/3600:.1f} hours)")
    
    while True:
        result = subprocess.run(
            ["pgrep", "-f", "python train.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            # Process not found = training complete
            print(f"[Pipeline] Training process terminated after {poll_count} polls")
            print("[Pipeline] ✓ Training completed!")
            return True
        
        elapsed = time.time() - start_time
        if max_wait_sec and elapsed > max_wait_sec:
            print(f"[Pipeline] ✗ Timeout: training did not complete within {max_wait_sec}s")
            return False
        
        hours = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)
        secs = int(elapsed % 60)
        poll_count += 1
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Poll #{poll_count} | Elapsed: {hours}h {mins}m {secs}s | Training in progress...")
        
        time.sleep(poll_interval_sec)


def export_vnnlib():
    """Run export_vnnlib.py to generate ONNX + VNNLIB + YAML"""
    print("[Pipeline] ========================================")
    print("[Pipeline] Step 1: Export VNNLIB")
    print("[Pipeline] ========================================")
    
    export_script = PROJECT_ROOT / "export_vnnlib.py"
    if not export_script.exists():
        print(f"[Pipeline] ✗ export_vnnlib.py not found at {export_script}")
        return False
    
    cmd = [
        sys.executable,
        str(export_script),
        "--alpha-lyap", "0.01",
        "--tolerance", "1e-6"
    ]
    
    print(f"[Pipeline] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode == 0:
        print("[Pipeline] ✓ VNNLIB export successful")
        return True
    else:
        print(f"[Pipeline] ✗ VNNLIB export failed (returncode={result.returncode})")
        return False


def run_abcrown_bab(run_bab=True, timeout_sec=3600):
    """
    Run alpha-beta-CROWN complete_verifier for formal BaB verification.
    
    Args:
        run_bab: Whether to actually run abcrown.py
        timeout_sec: Timeout for BaB execution
    """
    if not run_bab:
        print("[Pipeline] Skipping BaB (--skip-bab not set)")
        return None
    
    print("[Pipeline] ========================================")
    print("[Pipeline] Step 2: Run alpha-beta-CROWN BaB")
    print("[Pipeline] ========================================")
    
    # Use cartpole_verification.yaml from project root
    yaml_config = PROJECT_ROOT / "cartpole_verification.yaml"
    if not yaml_config.exists():
        print(f"[Pipeline] ✗ Config not found: {yaml_config}")
        return False
    
    abcrown_py = ALPHA_BETA_CROWN_DIR / "abcrown.py"
    if not abcrown_py.exists():
        print(f"[Pipeline] ✗ abcrown.py not found: {abcrown_py}")
        return False
    
    # Change to abcrown directory for proper import resolution
    cmd = [
        sys.executable,
        "abcrown.py",
        "--config", str(yaml_config)
    ]
    
    print(f"[Pipeline] Running from: {ALPHA_BETA_CROWN_DIR}")
    print(f"[Pipeline] Command: {' '.join(cmd)}")
    print(f"[Pipeline] Timeout: {timeout_sec}s ({timeout_sec/3600:.1f} hours)")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ALPHA_BETA_CROWN_DIR),
            timeout=timeout_sec,
            capture_output=False
        )
        
        if result.returncode == 0:
            print("[Pipeline] ✓ BaB verification completed")
            return True
        else:
            print(f"[Pipeline] ✗ BaB returned non-zero code: {result.returncode}")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"[Pipeline] ✗ BaB timeout after {timeout_sec}s")
        return False
    except Exception as e:
        print(f"[Pipeline] ✗ BaB error: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete Lyapunov verification pipeline")
    parser.add_argument("--no-wait", action="store_true", help="Skip training wait (assume training done)")
    parser.add_argument("--skip-export", action="store_true", help="Skip VNNLIB export")
    parser.add_argument("--skip-bab", action="store_true", help="Skip BaB verification")
    parser.add_argument("--bab-timeout", type=int, default=3600, help="BaB timeout in seconds")
    parser.add_argument("--poll-interval", type=int, default=60, help="Training poll interval (seconds)")
    parser.add_argument("--max-wait", type=int, default=None, help="Max training wait time (seconds)")
    
    args = parser.parse_args()
    
    print("")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  CARTPOLE LYAPUNOV VERIFICATION PIPELINE".center(78) + "║")
    print("║" + "═" * 78 + "║")
    print("║ Training → Export VNNLIB → Run alpha-beta-CROWN BaB Verification".ljust(79) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print("")
    
    # Step 0: Wait for training
    if not args.no_wait:
        if not wait_for_training(
            poll_interval_sec=args.poll_interval,
            max_wait_sec=args.max_wait
        ):
            print("[Pipeline] ✗ Failed to wait for training")
            return 1
        
        print("[Pipeline] Sleeping 5s to ensure all files flushed...")
        time.sleep(5)
    else:
        print("[Pipeline] Skipping training wait (--no-wait)")
    
    # Step 1: Export VNNLIB
    if not args.skip_export:
        if not export_vnnlib():
            print("[Pipeline] ✗ VNNLIB export failed")
            return 1
    else:
        print("[Pipeline] Skipping VNNLIB export (--skip-export)")
    
    # Step 2: Run BaB
    if not args.skip_bab:
        bab_result = run_abcrown_bab(
            run_bab=True,
            timeout_sec=args.bab_timeout
        )
        if bab_result is False:
            print("[Pipeline] ✗ BaB verification failed")
            return 1
    else:
        print("[Pipeline] Skipping BaB verification (--skip-bab)")
    
    print("")
    print("╔" + "═" * 78 + "╗")
    print("║" + "PIPELINE COMPLETE - All steps succeeded!".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
