from __future__ import annotations

from pathlib import Path

from apps.common import ROOT, run_python_file


def run(extra_args: list[str]) -> None:
    target = ROOT / "examples" / "cartpole" / "state_training.py"
    run_python_file(Path(target), extra_args)
