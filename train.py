#!/usr/bin/env python3
"""Root training router with separated pendulum/cartpole app flows."""

from __future__ import annotations

import argparse
from apps.cartpole.train import run as run_cartpole_train
from apps.pendulum.train import run as run_pendulum_train


def _interactive_choice() -> tuple[str, str] | None:
    options = {
        "1": ("pendulum", "state"),
        "2": ("pendulum", "output"),
        "3": ("cartpole", "state"),
        "pendulum_state": ("pendulum", "state"),
        "pendulum-state": ("pendulum", "state"),
        "pendulum_output": ("pendulum", "output"),
        "pendulum-output": ("pendulum", "output"),
        "cartpole": ("cartpole", "state"),
    }

    print("Select model to train:")
    print("  1) pendulum_state")
    print("  2) pendulum_output")
    print("  3) cartpole")
    print("Type number/name, or q to quit. Default: 1")

    raw = input("> ").strip().lower()
    if raw == "":
        raw = "1"
    if raw in {"q", "quit", "exit"}:
        return None
    return options.get(raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train entrypoint with explicit model separation."
    )
    subparsers = parser.add_subparsers(dest="model")

    p_pendulum = subparsers.add_parser("pendulum", help="Pendulum training")
    p_pendulum.add_argument(
        "--variant",
        choices=["state", "output"],
        default="state",
        help="Choose pendulum state-feedback or output-feedback workflow.",
    )
    p_pendulum.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Extra args passed through to pendulum training script.",
    )

    p_cartpole = subparsers.add_parser("cartpole", help="Cartpole training")
    p_cartpole.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Extra args passed through to cartpole training script.",
    )

    args = parser.parse_args()
    if args.model is None:
        selected = _interactive_choice()
        if selected is None:
            print("Exit without running training.")
            return
        model, variant = selected
        if model == "pendulum":
            run_pendulum_train(variant, [])
            return
        run_cartpole_train([])
        return

    passthrough = list(args.args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    if args.model == "pendulum":
        run_pendulum_train(args.variant, passthrough)
        return

    run_cartpole_train(passthrough)


if __name__ == "__main__":
    main()
