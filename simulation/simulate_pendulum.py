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

    # Controller mujoco-feedback clamp theo giới hạn đồng bộ với MuJoCo XML (±1).
    controller = NeuralNetworkController(
        nlayer=4,
        in_dim=3,
        out_dim=1,
        hidden_dim=8,
        clip_output="clamp",
        u_lo=torch.tensor([-1.0]),
        u_up=torch.tensor([1.0]),
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

    # Try to load observer trained specifically for MuJoCo surrogate dynamics
    observer_mujoco_path = PROJECT_ROOT / "models" / "mujoco_feedback" / "observer_mujoco_surrogate_[8,8].pth"
    observer_legacy_path = PROJECT_ROOT / "models" / "mujoco_feedback" / "pendulum_mujoco_feedback.pth"
    
    observer_loaded = False
    if observer_mujoco_path.exists():
        print(f"✓ Loading observer trained for MuJoCo surrogate: {observer_mujoco_path}")
        ckpt = torch.load(observer_mujoco_path, map_location=device, weights_only=False)
        if "observer_state_dict" in ckpt:
            observer.load_state_dict(ckpt["observer_state_dict"])
            observer_loaded = True
    
    if not observer_loaded and observer_legacy_path.exists():
        print(f"⚠ Observer-specific checkpoint not found, using legacy checkpoint: {observer_legacy_path}")
        ckpt = torch.load(observer_legacy_path, map_location=device, weights_only=False)
        if "observer_state_dict" in ckpt:
            observer.load_state_dict(ckpt["observer_state_dict"])
            observer_loaded = True
    
    if not observer_loaded:
        print(f"⚠ No observer checkpoint found. Using random initialization.")
        print(f"   Recommendation: Train observer with 'python apps/pendulum/mujoco_feedback.py train_observer'")
    
    observer.eval()
    return controller, observer

def main():
    device = torch.device("cpu")
    # Quy về frame train bằng theta_norm = wrap_to_pi(theta_raw - pi), để 0 là upright.
    model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "assets" / "pendulum.xml"))
    data = mujoco.MjData(model)

    nn_controller, nn_observer = load_neural_modules(device)

    # BALANCE ONLY (upright): khởi tạo gần vị trí thẳng đứng.
    data.qpos[0] = np.pi + 0.01  # Đặt gần upright một chút để tránh trường hợp PD bị deadzone.
    data.qvel[0] = 0.0
    mujoco.mj_forward(model, data)

    # Balance-only quanh upright (không swing-up): dùng PD chắc chắn hội tụ trong MuJoCo.
    # Có thể bật lại NN bằng cách đặt USE_NEURAL_CONTROLLER=True để so sánh.
    USE_NEURAL_CONTROLLER = True
    k_p = 2.0
    k_d = 0.8
    u_min = float(model.actuator_ctrlrange[0, 0])
    u_max = float(model.actuator_ctrlrange[0, 1])
    
    control_dt = 0.01
    last_ctrl_time = -control_dt
    tau_applied = 0.0
    theta_upright_raw = np.pi
    init_theta_norm = (data.qpos[0] - theta_upright_raw + np.pi) % (2 * np.pi) - np.pi
    x_hat = torch.tensor([[init_theta_norm, 0.0]], dtype=torch.float32, device=device)
    in_stabilization_mode = False

    log = {'time': [], 'theta': [], 'theta_dot': [], 'E': [], 'e_theta': [], 'u': [], 'in_roa': []}

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wide_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wide_fixed")
        if wide_cam_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = wide_cam_id

        while viewer.is_running() and data.time < 30.0:
            step_start = time.time()

            # 1. Đọc sensor raw rồi quy đổi về frame train (0=upright)
            theta_raw = data.sensor("sens_theta").data[0]
            theta_dot = data.sensor("sens_theta_dot").data[0]

            theta_norm = (theta_raw - theta_upright_raw + np.pi) % (2 * np.pi) - np.pi

            # 2. Điều khiển tần số 100Hz (khớp với dt huấn luyện)
            if data.time - last_ctrl_time >= control_dt - 1e-6:
                # Năng lượng quanh upright (chỉ để log/debug)
                m, l, g = 0.15, 0.5, 9.81
                I = m * (l**2)
                current_E = 0.5 * I * (theta_dot**2) + m * g * l * np.cos(theta_norm)

                if USE_NEURAL_CONTROLLER:
                    # Đường NN giữ nguyên cho mục đích so sánh/ablation.
                    y_obs = torch.tensor([[theta_norm]], dtype=torch.float32, device=device)
                    with torch.no_grad():
                        nn_input = torch.cat([x_hat, y_obs], dim=1)
                        u_tensor = nn_controller(nn_input)
                        tau_applied = float(np.clip(u_tensor.item(), u_min, u_max))
                        x_hat = nn_observer(
                            x_hat,
                            torch.tensor([[tau_applied]], dtype=torch.float32, device=device),
                            y_obs,
                        )
                    in_stabilization_mode = True
                else:
                    # Balance-only upright: PD quanh theta_norm=0.
                    tau_applied = np.clip(-k_p * theta_norm - k_d * theta_dot, u_min, u_max)
                    x_hat = torch.tensor([[theta_norm, theta_dot]], dtype=torch.float32, device=device)
                    in_stabilization_mode = True

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