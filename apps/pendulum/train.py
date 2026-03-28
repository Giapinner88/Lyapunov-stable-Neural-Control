from __future__ import annotations

from pathlib import Path

from apps.common import ROOT, run_python_file


def run(variant: str, extra_args: list[str]) -> None:
    if variant == "state":
        target = ROOT / "examples" / "pendulum" / "state_training.py"
    elif variant == "output":
        target = ROOT / "examples" / "pendulum" / "output_training.py"
    else:
        raise ValueError("Pendulum variant must be one of: state, output")
    run_python_file(Path(target), extra_args)
