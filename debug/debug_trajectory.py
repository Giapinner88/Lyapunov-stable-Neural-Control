import sys
import os
import csv
import mujoco
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neural_lyapunov_training.pendulum import PendulumDynamics
from neural_lyapunov_training.dynamical_system import SecondOrderDiscreteTimeSystem
from neural_lyapunov_training.controllers import NeuralNetworkController, NeuralNetworkLuenbergerObserver


def load_neural_modules(device):
    pendulum_continuous = PendulumDynamics(m=0.15, l=0.5, beta=0.1)
    dynamics = SecondOrderDiscreteTimeSystem(pendulum_continuous, dt=0.01)

    controller = NeuralNetworkController(
        nlayer=4,
        in_dim=3,
        out_dim=1,
        hidden_dim=8,
        clip_output="clamp",
        u_lo=torch.tensor([-0.25]),
        u_up=torch.tensor([0.25]),
        x_equilibrium=torch.zeros(3),
        u_equilibrium=torch.zeros(1),
    ).to(device)

    h = lambda x: pendulum_continuous.h(x)
    observer = NeuralNetworkLuenbergerObserver(
        z_dim=2,
        y_dim=1,
        dynamics=dynamics,
        h=h,
        zero_obs_error=torch.zeros(1, 1),
        fc_hidden_dim=[8, 8],
    ).to(device)

    model_path = PROJECT_ROOT / "models" / "pendulum_output_feedback.pth"
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        if "controller_state_dict" in ckpt and "observer_state_dict" in ckpt:
            controller_state_dict = ckpt["controller_state_dict"]
            observer_state_dict = ckpt["observer_state_dict"]
        elif "state_dict" in ckpt:
            controller_state_dict = {
                k.replace("controller.", ""): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("controller.")
            }
            observer_state_dict = {
                k.replace("observer.", ""): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("observer.")
            }
        else:
            raise RuntimeError(f"Checkpoint format khong hop le: {model_path}")

        controller.load_state_dict(controller_state_dict)
        observer.load_state_dict(observer_state_dict)

    controller.eval()
    observer.eval()
    return controller, observer

def main():
    device = torch.device("cpu")
    model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "assets" / "pendulum.xml"))
    data = mujoco.MjData(model)

    nn_controller, nn_observer = load_neural_modules(device)

    # Test BOTH starting conditions
    test_configs = [
        (np.pi + 0.01, "START NEAR UPRIGHT (θ=π+0.01)"),
        (0.0, "START AT DOWNWARD (θ=0)"),
    ]

    for init_qpos, label in test_configs:
        sep = "="*80
        print(f"\n{sep}")
        print(f"  {label}")
        print(f"{sep}")

        data.qpos[0] = init_qpos
        data.qvel[0] = 0.0
        mujoco.mj_forward(model, data)

        m, l, g, b = 0.15, 0.5, 9.81, 0.1
        I = m * (l**2)
        
        control_dt = 0.01
        last_ctrl_time = -control_dt
        tau_applied = 0.0
        theta_upright_raw = np.pi
        
        init_theta_norm = (data.qpos[0] - theta_upright_raw + np.pi) % (2 * np.pi) - np.pi
        x_hat = torch.tensor([[init_theta_norm, 0.0]], dtype=torch.float32, device=device)

        print(f"θ_raw init: {data.qpos[0]:.4f} rad ({np.degrees(data.qpos[0]):.1f}°)")
        print(f"θ_norm init: {init_theta_norm:.4f} rad ({np.degrees(init_theta_norm):.1f}°)")
        print()
        print(f"{'Time':>8} | {'θ_raw':>10} | {'θ_norm':>10} | {'θ̇':>10} | {'u':>8}")
        print("-" * 80)

        for step in range(3001):
            theta_raw = data.sensor("sens_theta").data[0]
            theta_dot = data.sensor("sens_theta_dot").data[0]
            theta_norm = (theta_raw - theta_upright_raw + np.pi) % (2 * np.pi) - np.pi

            if data.time - last_ctrl_time >= control_dt - 1e-6:
                y_obs = torch.tensor([[theta_norm]], dtype=torch.float32, device=device)
                
                with torch.no_grad():
                    nn_input = torch.cat([x_hat, y_obs], dim=1)
                    u_tensor = nn_controller(nn_input)
                    tau_applied = u_tensor.item()
                    x_hat = nn_observer(x_hat, u_tensor, y_obs)

                last_ctrl_time = data.time
                
                # Print at key time points
                if step < 5 or step % 600 == 0 or step > 2995:
                    print(f"{data.time:8.2f} | {theta_raw:10.4f} | {theta_norm:10.4f} | {theta_dot:10.4f} | {tau_applied:8.4f}")

            data.ctrl[0] = tau_applied
            mujoco.mj_step(model, data)

        # Final state
        print(f"\nFINAL STATE | {theta_raw:10.4f} | {theta_norm:10.4f} | {theta_dot:10.4f} | {tau_applied:8.4f}")
        print()

if __name__ == "__main__":
    main()
