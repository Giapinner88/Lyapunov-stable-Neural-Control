import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.training_config import get_default_config
from core.trainer import LyapunovTrainer


def train(system="pendulum", pretrain_epochs=150, cegis_epochs=350, alpha_lyap=0.08):
    config = get_default_config(system)
    config.loop.pretrain_epochs = int(pretrain_epochs)
    config.loop.cegis_epochs = int(cegis_epochs)
    config.loop.alpha_lyap = float(alpha_lyap)
    LyapunovTrainer(config).run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Lyapunov-stable controller")
    parser.add_argument("--system", type=str, default="pendulum", choices=["pendulum", "cartpole"])
    parser.add_argument("--pretrain-epochs", type=int, default=150)
    parser.add_argument("--cegis-epochs", type=int, default=350)
    parser.add_argument("--alpha-lyap", type=float, default=0.08)
    args = parser.parse_args()

    train(
        system=args.system,
        pretrain_epochs=args.pretrain_epochs,
        cegis_epochs=args.cegis_epochs,
        alpha_lyap=args.alpha_lyap,
    )