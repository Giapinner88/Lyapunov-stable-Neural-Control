from __future__ import annotations

from pathlib import Path

from apps.common import ROOT, run_abcrown


def run(config: str | None, extra_args: list[str]) -> None:
    if not config:
        raise ValueError(
            "Cartpole verification requires --config because no default cartpole YAML is bundled in verification/cartpole/."
        )
    config_path = Path(config)
    if not config_path.is_absolute():
        candidate = ROOT / "verification" / config
        if candidate.exists():
            config_path = candidate
        else:
            config_path = ROOT / "verification" / "cartpole" / config
    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Cartpole verification config not found: {config_path}")
    run_abcrown(config_path, extra_args)
