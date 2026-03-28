import argparse
from pathlib import Path
import sys

import torch

if __package__ is None or __package__ == "":
    # Allow running this file directly: python core/export.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_lyapunov_training.runtime_utils import load_trained_system
from neural_lyapunov_training.verification import CartpoleLyapunovLevelsetGraph


def export_to_onnx(
    controller_path: str = "checkpoints/cartpole/cartpole_controller.pth",
    lyapunov_path: str = "checkpoints/cartpole/cartpole_lyapunov.pth",
    output_onnx: str = "verification_results/bab_artifacts/cartpole_lyapunov_in_levelset.onnx",
    alpha_lyap: float = 0.01,
) -> None:
    device = torch.device("cpu")
    bundle = load_trained_system(
        controller_path=controller_path,
        lyapunov_path=lyapunov_path,
        system_name="cartpole",
        device=device,
    )

    graph = CartpoleLyapunovLevelsetGraph(
        controller=bundle.controller,
        lyapunov=bundle.lyapunov,
        dynamics=bundle.dynamics,
        alpha_lyap=alpha_lyap,
    ).to(device)
    graph.eval()

    output_path = Path(output_onnx)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_x = torch.zeros((1, bundle.config.model.nx), dtype=torch.float32)

    torch.onnx.export(
        graph,
        dummy_x,
        str(output_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["X"],
        output_names=["Y"],
        dynamic_axes={"X": {0: "batch_size"}, "Y": {0: "batch_size"}},
    )
    print(f"Exported cartpole verification graph to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CartPole verification ONNX graph")
    parser.add_argument("--controller", type=str, default="checkpoints/cartpole/cartpole_controller.pth")
    parser.add_argument("--lyapunov", type=str, default="checkpoints/cartpole/cartpole_lyapunov.pth")
    parser.add_argument(
        "--output-onnx",
        type=str,
        default="verification_results/bab_artifacts/cartpole_lyapunov_in_levelset.onnx",
    )
    parser.add_argument("--alpha-lyap", type=float, default=0.01)
    args = parser.parse_args()

    export_to_onnx(
        controller_path=args.controller,
        lyapunov_path=args.lyapunov,
        output_onnx=args.output_onnx,
        alpha_lyap=args.alpha_lyap,
    )


if __name__ == "__main__":
    main()
