import sys
import os
import csv
import mujoco
import mujoco.viewer
import time
import numpy as np
import torch

# Đảm bảo import được framework từ thư mục gốc của repo
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neural_lyapunov_training.pendulum import PendulumDynamics
from neural_lyapunov_training.dynamical_system import SecondOrderDiscreteTimeSystem
from neural_lyapunov_training.controllers import NeuralNetworkController, NeuralNetworkLuenbergerObserver


def load_neural_modules(device):
    """Khởi tạo Controller và Observer với cấu trúc Hidden Dim [8, 8] từ bài báo"""
    pendulum_continuous = PendulumDynamics(m=0.15, l=0.5, beta=0.1)
    dynamics = SecondOrderDiscreteTimeSystem(pendulum_continuous, dt=0.01)

    # Controller output-feedback luôn bị clamp theo giới hạn đã train (±0.25)
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
            raise RuntimeError(
                f"Checkpoint format khong hop le: {model_path}. "
                "Can it nhat 'state_dict' hoac cap keys 'controller_state_dict'/'observer_state_dict'."
            )

        if len(controller_state_dict) == 0 or len(observer_state_dict) == 0:
            raise RuntimeError(
                f"Checkpoint tai {model_path} khong chua trong so controller/observer. "
                "Hay chay lai apps/pendulum/output_feedback.py de tao checkpoint day du."
            )

        controller.load_state_dict(controller_state_dict)
        observer.load_state_dict(observer_state_dict)
        print(f"✓ Đã nạp trọng số từ {model_path}")
    else:
        print(f"⚠ Cảnh báo: Model file không tìm thấy tại {model_path}")

    controller.eval()
    observer.eval()
    return controller, observer

def main():
    device = torch.device("cpu")
    # XML hiện tại: raw q=0 gần buông thõng, raw q=pi gần upright.
    # Quy về frame train bằng theta_norm = wrap_to_pi(theta_raw - pi), để 0 là upright.
    model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "assets" / "pendulum.xml"))
    data = mujoco.MjData(model)

    nn_controller, nn_observer = load_neural_modules(device)

    # --- Trạng thái ban đầu: Gần vị trí buông thõng trong frame raw của XML ---
    data.qpos[0] = 0.3
    data.qvel[0] = 0.0
    mujoco.mj_forward(model, data)

    # --- Thông số vật lý chuẩn ---
    m, l, g, b = 0.15, 0.5, 9.81, 0.1
    I = m * (l**2) # 0.0375
    E_desired = m * g * l # Năng lượng tại θ=0 (upright)
    swingup_gain = 0.9 # Bơm năng lượng mạnh hơn một chút để dễ vào ROA.
    damping_gain = 0.10 # Giảm damping ngoài ROA để tránh bị kéo ngược quá sớm.
    recover_k_theta = 0.35 # Giảm lực kéo vị trí để không triệt swing-up.
    recover_k_omega = 0.30 # Giảm lực hãm vận tốc để bớt "phanh gấp".
    bleed_k = 0.22 # Hệ số xả bớt năng lượng khi gần upright và dư năng lượng.
    bleed_theta_band = 0.20
    bleed_energy_margin = 0.03
    u_swing_max = 3.0 # Giới hạn torque tối đa khi swing-up, có thể điều chỉnh để tránh quá đà hoặc không đủ lực.
    # Hysteresis ROA: vào ROA khi |θ|<0.25 và |ω|<0.6, ra ROA khi |θ|<0.5 và |ω|<1.0. Điều chỉnh để tránh bật/tắt mode liên tục gần ranh giới ROA.
    roa_enter_theta, roa_enter_omega = 0.30, 0.70
    roa_exit_theta, roa_exit_omega = 0.55, 1.30
    
    control_dt = 0.01
    last_ctrl_time = -control_dt
    tau_applied = 0.0
    theta_upright_raw = np.pi
    init_theta_norm = (data.qpos[0] - theta_upright_raw + np.pi) % (2 * np.pi) - np.pi
    x_hat = torch.tensor([[init_theta_norm, 0.0]], dtype=torch.float32, device=device)
    in_stabilization_mode = False

    log = {'time': [], 'theta': [], 'theta_dot': [], 'E': [], 'e_theta': [], 'u': [], 'in_roa': []}

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < 30.0:
            step_start = time.time()

            # 1. Đọc sensor raw rồi quy đổi về frame train (0=upright)
            theta_raw = data.sensor("sens_theta").data[0]
            theta_dot = data.sensor("sens_theta_dot").data[0]

            theta_norm = (theta_raw - theta_upright_raw + np.pi) % (2 * np.pi) - np.pi

            # 2. Điều khiển tần số 100Hz (khớp với dt huấn luyện)
            if data.time - last_ctrl_time >= control_dt - 1e-6:
                y_obs = torch.tensor([[theta_norm]], dtype=torch.float32, device=device)
                current_E = 0.5 * I * (theta_dot**2) + m * g * l * np.cos(theta_norm)
                
                # Hysteresis ROA để tránh bật/tắt mode liên tục.
                in_roa_enter = abs(theta_norm) < roa_enter_theta and abs(theta_dot) < roa_enter_omega
                in_roa_exit = abs(theta_norm) < roa_exit_theta and abs(theta_dot) < roa_exit_omega

                if in_stabilization_mode and not in_roa_exit:
                    in_stabilization_mode = False
                elif (not in_stabilization_mode) and in_roa_enter:
                    in_stabilization_mode = True

                with torch.no_grad():
                    if in_stabilization_mode:
                        # NN Output Feedback: u = π(x_hat, y)
                        nn_input = torch.cat([x_hat, y_obs], dim=1)
                        u_tensor = nn_controller(nn_input)
                        tau_applied = u_tensor.item()
                        
                        # Cập nhật Observer: x_hat+ = f(x_hat, u, y)
                        x_hat = nn_observer(x_hat, u_tensor, y_obs)
                    else:
                        # Ngoài ROA: swing-up + damping + recover để giảm quá đà.
                        u_energy = swingup_gain * theta_dot * (E_desired - current_E)
                        u_damping = -damping_gain * theta_dot

                        near_upright = abs(theta_norm) < 0.75
                        excess_energy = current_E - E_desired
                        moving_away_from_upright = (theta_norm * theta_dot) > 0.0
                        need_recover = near_upright and moving_away_from_upright and (
                            excess_energy > 0.12 or abs(theta_dot) > 1.3
                        )

                        if need_recover:
                            u_recover = -recover_k_theta * theta_norm - recover_k_omega * theta_dot
                            tau_applied = 0.35 * u_energy + u_damping + u_recover
                        else:
                            tau_applied = u_energy + 0.5 * u_damping

                        # Energy bleed nhỏ gần upright để tránh lặp vượt năng lượng rồi bật ra khỏi ROA.
                        if abs(theta_norm) < bleed_theta_band and excess_energy > bleed_energy_margin:
                            u_bleed = -bleed_k * theta_dot
                            tau_applied += u_bleed

                        tau_applied = np.clip(tau_applied, -u_swing_max, u_swing_max)
                        
                        # Bám đuổi trạng thái thực khi ngoài ROA
                        x_hat = torch.tensor([[theta_norm, theta_dot]], dtype=torch.float32, device=device)

                last_ctrl_time = data.time
                log['time'].append(data.time)
                log['theta'].append(theta_norm)
                log['theta_dot'].append(theta_dot)
                log['E'].append(current_E)
                log['u'].append(tau_applied)
                log['in_roa'].append(1 if in_stabilization_mode else 0)
                log['e_theta'].append(x_hat[0, 0].item() - theta_norm)

            data.ctrl[0] = tau_applied
            mujoco.mj_step(model, data)
            
            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

    _save_to_csv(log)

def _save_to_csv(log):
    """Xuất dữ liệu mô phỏng ra file CSV để debug"""
    import os
    output_dir = PROJECT_ROOT / "output"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = output_dir / "pendulum_simulation.csv"
    
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['time', 'theta', 'theta_dot', 'E', 'u', 'in_roa', 'e_theta']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i in range(len(log['time'])):
                writer.writerow({
                    'time': f"{log['time'][i]:.4f}",
                    'theta': f"{log['theta'][i]:.6f}",
                    'theta_dot': f"{log['theta_dot'][i]:.6f}",
                    'E': f"{log['E'][i]:.6f}",
                    'u': f"{log['u'][i]:.6f}",
                    'in_roa': int(log['in_roa'][i]),
                    'e_theta': f"{log['e_theta'][i]:.6f}"
                })
        print(f"\n✓ Đã xuất dữ liệu ra CSV: {csv_path}")
        print(f"  Số sample: {len(log['time'])}")
        
    except Exception as e:
        print(f"\n✗ Lỗi khi xuất CSV: {e}")



if __name__ == "__main__":
    main()