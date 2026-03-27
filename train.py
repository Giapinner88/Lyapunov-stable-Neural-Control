import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.training_config import get_default_config
from core.trainer import LyapunovTrainer


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_seed_registry(record: dict) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / "seed_registry.csv"
    json_dir = reports_dir / "run_registry"
    json_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_id",
        "timestamp",
        "system",
        "seed",
        "deterministic",
        "resume",
        "strong_profile",
        "paper_profile",
        "final_mission",
        "pretrain_epochs",
        "cegis_epochs",
        "alpha_lyap",
        "bank_capacity",
        "bank_mode",
        "replay_new_ratio",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: record.get(k) for k in fieldnames})

    (json_dir / f"{record['run_id']}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def apply_cartpole_strong_profile(config) -> None:
    if config.system.name != "cartpole":
        return
    # Stronger profile to improve certified region for cartpole.
    config.loop.learning_rate = 5e-4
    config.loop.learner_updates = 4
    config.loop.batch_size = 1024
    config.loop.train_batch_size = 1024
    config.loop.attack_seed_size = 512
    config.loop.sweep_every = 20
    config.loop.checkpoint_every = 20
    config.loop.lqr_anchor_radius = (0.10, 0.10, 0.10, 0.10)
    config.curriculum.start_scale = 0.5
    config.cegis.local_box_samples = 1024
    config.cegis.local_box_weight = 0.50


def apply_cartpole_paper_profile(config) -> None:
    if config.system.name != "cartpole":
        return
    # Aggressive profile aimed at getting closer to paper-level behavior.
    config.loop.learning_rate = 3e-4
    config.loop.learner_updates = 5
    config.loop.batch_size = 1024
    config.loop.train_batch_size = 1024
    config.loop.attack_seed_size = 640
    config.loop.sweep_every = 20
    config.loop.checkpoint_every = 20
    config.curriculum.start_scale = 0.4
    config.curriculum.end_scale = 1.0

    config.attacker.num_steps = 120
    config.attacker.num_restarts = 8
    config.attacker.step_size = 0.015

    config.cegis.bank_capacity = 500000
    config.cegis.bank_mode = "reservoir"
    config.cegis.replay_new_ratio = 0.20
    config.cegis.violation_margin = 3e-4
    config.cegis.local_box_samples = 1536
    config.cegis.local_box_weight = 0.60


def apply_cartpole_final_mission_profile(config) -> None:
    if config.system.name != "cartpole":
        return
    # Final aggressive profile for pushing closest to paper-level results.
    config.loop.learning_rate = 2e-4
    config.loop.learner_updates = 6
    config.loop.batch_size = 1280
    config.loop.train_batch_size = 1280
    config.loop.attack_seed_size = 768
    config.loop.sweep_every = 10
    config.loop.checkpoint_every = 10
    config.loop.lqr_anchor_radius = (0.08, 0.08, 0.08, 0.08)
    config.curriculum.start_scale = 0.35
    config.curriculum.end_scale = 1.0

    config.attacker.num_steps = 160
    config.attacker.num_restarts = 10
    config.attacker.step_size = 0.012

    config.cegis.bank_capacity = 800000
    config.cegis.bank_mode = "reservoir"
    config.cegis.replay_new_ratio = 0.18
    config.cegis.violation_margin = 2e-4
    config.cegis.local_box_radius = 0.15
    config.cegis.local_box_samples = 2048
    config.cegis.local_box_weight = 0.70
    config.cegis.equilibrium_weight = 0.15


def train(
    system="cartpole",
    pretrain_epochs=150,
    cegis_epochs=350,
    alpha_lyap=0.08,
    resume=False,
    skip_pretrain_if_resumed=True,
    strong_profile=False,
    paper_profile=False,
    final_mission=False,
    bank_capacity=None,
    replay_new_ratio=None,
    bank_mode=None,
    curriculum_start_scale=None,
    curriculum_end_scale=None,
    local_box_samples=None,
    local_box_weight=None,
    attacker_steps=None,
    attacker_restarts=None,
    seed=None,
    deterministic=False,
):
    if seed is None:
        seed = int(datetime.now().timestamp() * 1000) % (2**31 - 1)
    seed = int(seed)
    set_global_seed(seed, deterministic=bool(deterministic))

    config = get_default_config(system)

    if strong_profile:
        apply_cartpole_strong_profile(config)
    if paper_profile:
        apply_cartpole_paper_profile(config)
    if final_mission:
        apply_cartpole_final_mission_profile(config)

    config.loop.pretrain_epochs = int(pretrain_epochs)
    config.loop.cegis_epochs = int(cegis_epochs)
    config.loop.alpha_lyap = float(alpha_lyap)

    if bank_capacity is not None:
        config.cegis.bank_capacity = int(bank_capacity)
    if replay_new_ratio is not None:
        config.cegis.replay_new_ratio = float(replay_new_ratio)
    if bank_mode is not None:
        config.cegis.bank_mode = str(bank_mode).lower()
    if curriculum_start_scale is not None:
        config.curriculum.start_scale = float(curriculum_start_scale)
    if curriculum_end_scale is not None:
        config.curriculum.end_scale = float(curriculum_end_scale)
    if local_box_samples is not None:
        config.cegis.local_box_samples = int(local_box_samples)
    if local_box_weight is not None:
        config.cegis.local_box_weight = float(local_box_weight)
    if attacker_steps is not None:
        config.attacker.num_steps = int(attacker_steps)
    if attacker_restarts is not None:
        config.attacker.num_restarts = int(attacker_restarts)

    print("[Config]", {
        "system": config.system.name,
        "seed": seed,
        "deterministic": bool(deterministic),
        "pretrain_epochs": config.loop.pretrain_epochs,
        "cegis_epochs": config.loop.cegis_epochs,
        "alpha_lyap": config.loop.alpha_lyap,
        "bank_capacity": config.cegis.bank_capacity,
        "bank_mode": config.cegis.bank_mode,
        "replay_new_ratio": config.cegis.replay_new_ratio,
        "curriculum": (config.curriculum.start_scale, config.curriculum.end_scale),
        "attacker_steps": config.attacker.num_steps,
        "attacker_restarts": config.attacker.num_restarts,
    })

    run_id = f"{config.system.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_seed_registry(
        {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "system": config.system.name,
            "seed": seed,
            "deterministic": bool(deterministic),
            "resume": bool(resume),
            "strong_profile": bool(strong_profile),
            "paper_profile": bool(paper_profile),
            "final_mission": bool(final_mission),
            "pretrain_epochs": int(config.loop.pretrain_epochs),
            "cegis_epochs": int(config.loop.cegis_epochs),
            "alpha_lyap": float(config.loop.alpha_lyap),
            "bank_capacity": int(config.cegis.bank_capacity),
            "bank_mode": str(config.cegis.bank_mode),
            "replay_new_ratio": float(config.cegis.replay_new_ratio),
        }
    )

    LyapunovTrainer(config).run(
        resume=bool(resume),
        skip_pretrain_if_resumed=bool(skip_pretrain_if_resumed),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Lyapunov-stable controller")
    parser.add_argument("--system", type=str, default="cartpole", choices=["pendulum", "cartpole"])
    parser.add_argument("--pretrain-epochs", type=int, default=150)
    parser.add_argument("--cegis-epochs", type=int, default=350)
    parser.add_argument("--alpha-lyap", type=float, default=0.08)
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoints if available")
    parser.add_argument(
        "--no-skip-pretrain",
        action="store_true",
        help="When resuming, still run pre-training before CEGIS",
    )
    parser.add_argument(
        "--strong-profile",
        action="store_true",
        help="Use stronger cartpole-focused hyperparameters",
    )
    parser.add_argument(
        "--paper-profile",
        action="store_true",
        help="Use aggressive paper-oriented cartpole profile",
    )
    parser.add_argument(
        "--final-mission",
        action="store_true",
        help="Use strongest paper-chasing cartpole profile",
    )
    parser.add_argument("--bank-capacity", type=int, default=None)
    parser.add_argument("--replay-new-ratio", type=float, default=None)
    parser.add_argument("--bank-mode", type=str, default=None, choices=["fifo", "reservoir"])
    parser.add_argument("--curriculum-start-scale", type=float, default=None)
    parser.add_argument("--curriculum-end-scale", type=float, default=None)
    parser.add_argument("--local-box-samples", type=int, default=None)
    parser.add_argument("--local-box-weight", type=float, default=None)
    parser.add_argument("--attacker-steps", type=int, default=None)
    parser.add_argument("--attacker-restarts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible training")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic PyTorch mode (may reduce speed)",
    )
    args = parser.parse_args()

    train(
        system=args.system,
        pretrain_epochs=args.pretrain_epochs,
        cegis_epochs=args.cegis_epochs,
        alpha_lyap=args.alpha_lyap,
        resume=args.resume,
        skip_pretrain_if_resumed=not args.no_skip_pretrain,
        strong_profile=args.strong_profile,
        paper_profile=args.paper_profile,
        final_mission=args.final_mission,
        bank_capacity=args.bank_capacity,
        replay_new_ratio=args.replay_new_ratio,
        bank_mode=args.bank_mode,
        curriculum_start_scale=args.curriculum_start_scale,
        curriculum_end_scale=args.curriculum_end_scale,
        local_box_samples=args.local_box_samples,
        local_box_weight=args.local_box_weight,
        attacker_steps=args.attacker_steps,
        attacker_restarts=args.attacker_restarts,
        seed=args.seed,
        deterministic=args.deterministic,
    )