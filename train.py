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

from neural_lyapunov_training.training_config import get_default_config
from neural_lyapunov_training.trainer import LyapunovTrainer


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
        "paper_locked",
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


def apply_cartpole_paper_profile(config) -> None:
    if config.system.name != "cartpole":
        return
    # Aggressive profile aimed at getting closer to paper-level behavior.
    config.loop.learning_rate = 1e-4
    config.loop.learner_updates = 2
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
    config.cegis.violation_margin = 1e-4
    config.cegis.local_box_weight = 0.30
    config.cegis.equilibrium_weight = 0.05
    config.cegis.lqr_anchor_weight = 0.05
    config.cegis.candidate_roa_weight = 0.2
    config.cegis.candidate_roa_num_samples = 256
    config.cegis.candidate_roa_scale = 0.4
    config.cegis.candidate_roa_rho = None
    config.cegis.candidate_roa_rho_quantile = 0.9
    config.cegis.candidate_roa_always = False


def train(
    system="cartpole",
    pretrain_epochs=150,
    cegis_epochs=350,
    alpha_lyap=0.01,
    resume=False,
    skip_pretrain_if_resumed=True,
    bank_capacity=None,
    replay_new_ratio=None,
    bank_mode=None,
    curriculum_start_scale=None,
    curriculum_end_scale=None,
    local_box_samples=None,
    local_box_weight=None,
    equilibrium_weight=None,
    attacker_steps=None,
    attacker_restarts=None,
    lyapunov_phi_dim=None,
    lyapunov_absolute_output=None,
    ibp_ratio=None,
    ibp_eps=None,
    candidate_roa_weight=None,
    candidate_roa_num_samples=None,
    candidate_roa_scale=None,
    candidate_roa_rho=None,
    candidate_roa_rho_quantile=None,
    candidate_roa_always=None,
    seed=None,
    deterministic=False,
):
    if seed is None:
        seed = int(datetime.now().timestamp() * 1000) % (2**31 - 1)
    seed = int(seed)
    set_global_seed(seed, deterministic=bool(deterministic))

    config = get_default_config(system)

    if config.system.name == "cartpole":
        apply_cartpole_paper_profile(config)

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
    if equilibrium_weight is not None:
        config.cegis.equilibrium_weight = float(equilibrium_weight)
    if attacker_steps is not None:
        config.attacker.num_steps = int(attacker_steps)
    if attacker_restarts is not None:
        config.attacker.num_restarts = int(attacker_restarts)
    if lyapunov_phi_dim is not None:
        config.model.lyapunov_phi_dim = int(lyapunov_phi_dim)
    if lyapunov_absolute_output is not None:
        config.model.lyapunov_absolute_output = bool(lyapunov_absolute_output)
    if ibp_ratio is not None:
        config.cegis.ibp_ratio = float(ibp_ratio)
    if ibp_eps is not None:
        config.cegis.ibp_eps = float(ibp_eps)
    if candidate_roa_weight is not None:
        config.cegis.candidate_roa_weight = float(candidate_roa_weight)
    if candidate_roa_num_samples is not None:
        config.cegis.candidate_roa_num_samples = int(candidate_roa_num_samples)
    if candidate_roa_scale is not None:
        config.cegis.candidate_roa_scale = float(candidate_roa_scale)
    if candidate_roa_rho is not None:
        config.cegis.candidate_roa_rho = float(candidate_roa_rho)
    if candidate_roa_rho_quantile is not None:
        config.cegis.candidate_roa_rho_quantile = float(candidate_roa_rho_quantile)
    if candidate_roa_always is not None:
        config.cegis.candidate_roa_always = bool(candidate_roa_always)

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
        "lyapunov_phi_dim": config.model.lyapunov_phi_dim,
        "lyapunov_absolute_output": config.model.lyapunov_absolute_output,
        "ibp_ratio": config.cegis.ibp_ratio,
        "ibp_eps": config.cegis.ibp_eps,
        "candidate_roa_weight": config.cegis.candidate_roa_weight,
        "candidate_roa_num_samples": config.cegis.candidate_roa_num_samples,
        "candidate_roa_scale": config.cegis.candidate_roa_scale,
        "candidate_roa_rho": config.cegis.candidate_roa_rho,
        "candidate_roa_rho_quantile": config.cegis.candidate_roa_rho_quantile,
        "candidate_roa_always": config.cegis.candidate_roa_always,
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
            "paper_locked": True,
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
    parser.add_argument("--alpha-lyap", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoints if available")
    parser.add_argument(
        "--no-skip-pretrain",
        action="store_true",
        help="When resuming, still run pre-training before CEGIS",
    )
    parser.add_argument("--bank-capacity", type=int, default=None)
    parser.add_argument("--replay-new-ratio", type=float, default=None)
    parser.add_argument("--bank-mode", type=str, default=None, choices=["fifo", "reservoir"])
    parser.add_argument("--curriculum-start-scale", type=float, default=None)
    parser.add_argument("--curriculum-end-scale", type=float, default=None)
    parser.add_argument("--local-box-samples", type=int, default=None)
    parser.add_argument("--local-box-weight", type=float, default=None)
    parser.add_argument("--equilibrium-weight", type=float, default=None)
    parser.add_argument("--attacker-steps", type=int, default=None)
    parser.add_argument("--attacker-restarts", type=int, default=None)
    parser.add_argument("--lyapunov-phi-dim", type=int, default=None)
    parser.add_argument("--lyapunov-absolute-output", type=int, choices=[0, 1], default=None)
    parser.add_argument("--ibp-ratio", type=float, default=None)
    parser.add_argument("--ibp-eps", type=float, default=None)
    parser.add_argument("--candidate-roa-weight", type=float, default=None)
    parser.add_argument("--candidate-roa-num-samples", type=int, default=None)
    parser.add_argument("--candidate-roa-scale", type=float, default=None)
    parser.add_argument("--candidate-roa-rho", type=float, default=None)
    parser.add_argument("--candidate-roa-rho-quantile", type=float, default=None)
    parser.add_argument("--candidate-roa-always", type=int, choices=[0, 1], default=None)
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
        bank_capacity=args.bank_capacity,
        replay_new_ratio=args.replay_new_ratio,
        bank_mode=args.bank_mode,
        curriculum_start_scale=args.curriculum_start_scale,
        curriculum_end_scale=args.curriculum_end_scale,
        local_box_samples=args.local_box_samples,
        local_box_weight=args.local_box_weight,
        equilibrium_weight=args.equilibrium_weight,
        attacker_steps=args.attacker_steps,
        attacker_restarts=args.attacker_restarts,
        lyapunov_phi_dim=args.lyapunov_phi_dim,
        lyapunov_absolute_output=(None if args.lyapunov_absolute_output is None else bool(args.lyapunov_absolute_output)),
        ibp_ratio=args.ibp_ratio,
        ibp_eps=args.ibp_eps,
        candidate_roa_weight=args.candidate_roa_weight,
        candidate_roa_num_samples=args.candidate_roa_num_samples,
        candidate_roa_scale=args.candidate_roa_scale,
        candidate_roa_rho=args.candidate_roa_rho,
        candidate_roa_rho_quantile=args.candidate_roa_rho_quantile,
        candidate_roa_always=(None if args.candidate_roa_always is None else bool(args.candidate_roa_always)),
        seed=args.seed,
        deterministic=args.deterministic,
    )