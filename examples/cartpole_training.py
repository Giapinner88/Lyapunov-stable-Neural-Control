from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import train


if __name__ == "__main__":
    train(system="cartpole")
