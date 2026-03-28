from __future__ import annotations

from apps.common import run_generate_vnnlib


DEFAULT_LOWER = [-12.0, -12.0]
DEFAULT_UPPER = [12.0, 12.0]
DEFAULT_HOLE_SIZE = 0.001
DEFAULT_OUTPUT_PREFIX = "verification/specs/pendulum/pendulum_state_feedback"


def run(rho: float, output_prefix: str | None, extra_args: list[str]) -> None:
    run_generate_vnnlib(
        default_lower=DEFAULT_LOWER,
        default_upper=DEFAULT_UPPER,
        hole_size=DEFAULT_HOLE_SIZE,
        rho=rho,
        output_prefix=output_prefix or DEFAULT_OUTPUT_PREFIX,
        extra_args=extra_args,
    )
