import sys
import os
import mujoco
import numpy as np
from pathlib import Path

# PROJECT_ROOT should be /home/giapinner88/Project/Lyapunov-stable-Neural-Control
PROJECT_ROOT = Path(__file__).resolve().parent

model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "assets" / "pendulum.xml"))
data = mujoco.MjData(model)

print("Testing coordinate system...")
print()

# Test 1: Start at θ=0 (downward in MuJoCo convention)
data.qpos[0] = 0.0
mujoco.mj_forward(model, data)
theta_raw_down = data.sensor("sens_theta").data[0]
print(f"1. qpos=0.0 (downward) → sens_theta={theta_raw_down:.4f} rad ({np.degrees(theta_raw_down):.1f}°)")

# Test 2: Start at θ=π (upward in MuJoCo convention)
data.qpos[0] = np.pi
mujoco.mj_forward(model, data)
theta_raw_up = data.sensor("sens_theta").data[0]
print(f"2. qpos=π → sens_theta={theta_raw_up:.4f} rad ({np.degrees(theta_raw_up):.1f}°)")

# Test 3: Small angle around π
data.qpos[0] = np.pi + 0.1
mujoco.mj_forward(model, data)
theta_raw = data.sensor("sens_theta").data[0]
print(f"3. qpos=π+0.1 → sens_theta={theta_raw:.4f} rad ({np.degrees(theta_raw):.1f}°)")

print()
print("Normalization logic:")
theta_upright_raw = np.pi

for test_qpos, label in [(0.0, "downward"), (np.pi, "upright"), (np.pi+0.1, "upright+0.1")]:
    data.qpos[0] = test_qpos
    mujoco.mj_forward(model, data)
    theta_raw = data.sensor("sens_theta").data[0]
    theta_norm = (theta_raw - theta_upright_raw + np.pi) % (2 * np.pi) - np.pi
    print(f"qpos={test_qpos:.4f} ({label:12s}) → raw={theta_raw:.4f} → norm={theta_norm:.4f} ({np.degrees(theta_norm):.1f}°)")
