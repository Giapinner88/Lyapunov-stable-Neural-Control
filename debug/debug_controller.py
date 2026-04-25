import sys
import os
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neural_lyapunov_training.pendulum import PendulumDynamics
from neural_lyapunov_training.dynamical_system import SecondOrderDiscreteTimeSystem
from neural_lyapunov_training.controllers import NeuralNetworkController, NeuralNetworkLuenbergerObserver

device = torch.device("cpu")

# Load controller
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
ckpt = torch.load(model_path, map_location=device, weights_only=False)

if "controller_state_dict" in ckpt:
    print("✓ Found 'controller_state_dict' in checkpoint")
if "state_dict" in ckpt:
    print("✓ Found 'state_dict' in checkpoint")
    print(f"\nState dict keys:")
    for k, v in ckpt["state_dict"].items():
        if k.startswith("controller."):
            print(f"  {k}: {v.shape}")

print(f"\n\nNN Controller architecture:")
print(controller)
print(f"\nController model input dimension: {controller.in_dim}")
print(f"Controller model output dimension: {controller.out_dim}")

print(f"\n\nNN Observer architecture:")
print(observer)

# Test with different input dimensions
print("\n" + "="*70)
print("Testing NN controller with different input dimensions:")
print("="*70)

test_inputs = [
    (torch.randn(1, 3), "3D input [theta, dtheta, y]"),
    (torch.randn(1, 4), "4D input [theta, dtheta, observer_error_theta, observer_error_dtheta]"),
]

for test_input, description in test_inputs:
    try:
        with torch.no_grad():
            output = controller(test_input)
        print(f"✓ {description}: input shape {test_input.shape} → output shape {output.shape}")
    except Exception as e:
        print(f"✗ {description}: FAILED - {str(e)[:60]}")
