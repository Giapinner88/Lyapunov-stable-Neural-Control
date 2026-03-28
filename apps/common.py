from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_env() -> dict[str, str]:
    env = dict(os.environ)
    extra = [
        str(ROOT),
        str(ROOT / "alpha-beta-CROWN"),
        str(ROOT / "alpha-beta-CROWN" / "complete_verifier"),
    ]
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join([current, *extra]) if current else ":".join(extra)
    return env


def run_python_file(target: Path, args: list[str]) -> None:
    if not target.exists():
        raise FileNotFoundError(f"Target script not found: {target}")
    subprocess.run([sys.executable, str(target), *args], check=True, cwd=str(ROOT), env=build_env())


def run_abcrown(config_path: Path, args: list[str]) -> None:
    verifier_dir = ROOT / "alpha-beta-CROWN" / "complete_verifier"
    abcrown = verifier_dir / "abcrown.py"
    if not abcrown.exists():
        raise FileNotFoundError(f"abcrown.py not found: {abcrown}")
    env = build_env()
    env["CONFIG_PATH"] = str(ROOT / "verification")
    subprocess.run(
        [sys.executable, str(abcrown), "--config", str(config_path), *args],
        check=True,
        cwd=str(verifier_dir),
        env=env,
    )


def run_generate_vnnlib(default_lower: list[float], default_upper: list[float], hole_size: float, rho: float, output_prefix: str, extra_args: list[str]) -> None:
    out_prefix = ROOT / output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "neural_lyapunov_training.generate_vnnlib",
        "--lower_limit",
        *[str(v) for v in default_lower],
        "--upper_limit",
        *[str(v) for v in default_upper],
        "--hole_size",
        str(hole_size),
        "--value_levelset",
        str(rho),
        str(out_prefix),
        *extra_args,
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=build_env())
