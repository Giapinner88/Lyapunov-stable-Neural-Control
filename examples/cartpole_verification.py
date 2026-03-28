from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verify import verify_cartpole_roa


if __name__ == "__main__":
    verify_cartpole_roa(
        controller_path="checkpoints/cartpole/cartpole_controller.pth",
        lyapunov_path="checkpoints/cartpole/cartpole_lyapunov.pth",
        output_dir="verification_results",
        alpha_lyap=0.01,
        run_crown=False,
        verbose=True,
    )
