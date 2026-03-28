#!/usr/bin/env python3
"""Root verification router with separated pendulum/cartpole app flows."""

from __future__ import annotations

import argparse
from apps.cartpole.verify import run as run_cartpole_verify
from apps.pendulum.verify import run as run_pendulum_verify


def _interactive_choice() -> tuple[str, str | None] | None:
    options = {
        "1": "pendulum",
        "2": "cartpole",
        "pendulum": "pendulum",
        "cartpole": "cartpole",
    }

    print("Select model to verify:")
    print("  1) pendulum (default config)")
    print("  2) cartpole (requires config path/name)")
    print("Type number/name, or q to quit. Default: 1")

    raw = input("> ").strip().lower()
    if raw == "":
        raw = "1"
    if raw in {"q", "quit", "exit"}:
        return None

    model = options.get(raw)
    if model is None:
        print("Invalid selection.")
        return None

    if model == "pendulum":
        return model, None

    cfg = input("Cartpole config (in verification/ or absolute path): ").strip()
    if cfg == "":
        print("Cartpole requires a config. Exit without verification.")
        return None
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verification entrypoint with explicit model separation."
    )
    subparsers = parser.add_subparsers(dest="model")

    p_pendulum = subparsers.add_parser("pendulum", help="Pendulum verification")
    p_pendulum.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional pendulum config filename in verification/ or absolute path.",
    )
    p_pendulum.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Extra args passed through to abcrown.py.",
    )

    p_cartpole = subparsers.add_parser("cartpole", help="Cartpole verification")
    p_cartpole.add_argument(
        "--config",
        type=str,
        required=True,
        help="Cartpole config filename in verification/ or absolute path.",
    )
    p_cartpole.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Extra args passed through to abcrown.py.",
    )

    args = parser.parse_args()

    if args.model is None:
        selected = _interactive_choice()
        if selected is None:
            print("Exit without running verification.")
            return
        model, config = selected
        if model == "pendulum":
            run_pendulum_verify(None, [])
            return
        run_cartpole_verify(config, [])
        return

    passthrough = list(args.args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    if args.model == "pendulum":
        run_pendulum_verify(args.config, passthrough)
        return

    run_cartpole_verify(args.config, passthrough)


if __name__ == "__main__":
    main()
