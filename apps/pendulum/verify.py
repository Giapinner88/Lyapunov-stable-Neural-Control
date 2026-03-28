from __future__ import annotations

from pathlib import Path

from apps.common import ROOT, run_abcrown


DEFAULT_CONFIG = "pendulum/pendulum_state_feedback_lyapunov_in_levelset.yaml"


def run(config: str | None, extra_args: list[str]) -> None:
    chosen = config or DEFAULT_CONFIG
    config_path = Path(chosen)
    if not config_path.is_absolute():
        candidate = ROOT / "verification" / chosen
        if candidate.exists():
            config_path = candidate
        else:
            config_path = ROOT / "verification" / "pendulum" / chosen
    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Pendulum verification config not found: {config_path}")
    run_abcrown(config_path, extra_args)
