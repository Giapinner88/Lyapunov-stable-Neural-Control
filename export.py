#!/usr/bin/env python3
"""Root export router with separated pendulum/cartpole app flows."""

from __future__ import annotations

import argparse
from apps.cartpole.export import run as run_cartpole_export
from apps.pendulum.export import run as run_pendulum_export


def _interactive_choice() -> str | None:
    options = {
        "1": "pendulum",
        "2": "cartpole",
        "pendulum": "pendulum",
        "cartpole": "cartpole",
    }

    print("Select model to export specs:")
    print("  1) pendulum")
    print("  2) cartpole")
    print("Type number/name, or q to quit. Default: 1")

    raw = input("> ").strip().lower()
    if raw == "":
        raw = "1"
    if raw in {"q", "quit", "exit"}:
        return None
    return options.get(raw)


def _interactive_rho(model: str) -> float | None:
    defaults = {
        "pendulum": "672",
        "cartpole": "0.13",
    }
    default_str = defaults[model]
    raw = input(f"rho value (default {default_str}): ").strip()
    if raw.lower() in {"q", "quit", "exit"}:
        return None
    if raw == "":
        raw = default_str
    try:
        return float(raw)
    except ValueError:
        print(f"Invalid rho: {raw}")
        return None


def _interactive_output_prefix() -> str | None:
    raw = input("Output prefix (optional, Enter for default): ").strip()
    if raw.lower() in {"q", "quit", "exit"}:
        return "__quit__"
    return raw or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export VNNLIB specs with explicit model separation."
    )
    subparsers = parser.add_subparsers(dest="model")

    p_pendulum = subparsers.add_parser("pendulum", help="Pendulum spec export")
    p_pendulum.add_argument("--rho", type=float, required=True)
    p_pendulum.add_argument("--output-prefix", type=str, default=None)
    p_pendulum.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to generate_vnnlib.",
    )

    p_cartpole = subparsers.add_parser("cartpole", help="Cartpole spec export")
    p_cartpole.add_argument("--rho", type=float, required=True)
    p_cartpole.add_argument("--output-prefix", type=str, default=None)
    p_cartpole.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to generate_vnnlib.",
    )

    args = parser.parse_args()

    if args.model is None:
        model = _interactive_choice()
        if model is None:
            print("Exit without exporting specs.")
            return
        rho = _interactive_rho(model)
        if rho is None:
            print("Exit without exporting specs.")
            return
        output_prefix = _interactive_output_prefix()
        if output_prefix == "__quit__":
            print("Exit without exporting specs.")
            return

        if model == "pendulum":
            run_pendulum_export(rho, output_prefix, [])
            return
        run_cartpole_export(rho, output_prefix, [])
        return

    passthrough = list(args.args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    if args.model == "pendulum":
        run_pendulum_export(args.rho, args.output_prefix, passthrough)
        return

    run_cartpole_export(args.rho, args.output_prefix, passthrough)


if __name__ == "__main__":
    main()
