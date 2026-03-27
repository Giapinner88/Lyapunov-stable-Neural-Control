import torch
import torch.nn as nn
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    # Allow running this file directly: python core/export.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models import NeuralController, NeuralLyapunov
from core.dynamics import PendulumDynamics

class CompleteVerifierGraph(nn.Module):
    def __init__(self, controller, lyapunov, dynamics, rho=0.01):
        super().__init__()
        self.controller = controller
        self.lyapunov = lyapunov
        self.dynamics = dynamics
        self.rho = rho

    def forward(self, x):
        # 1. Tính V(x) -> Tương đương Y_1
        v_t = self.lyapunov(x)
        
        # 2. Tính x_next qua Euler step
        u_t = self.controller(x)
        x_next = self.dynamics.step(x, u_t)
        
        # 3. Tính V(x_next)
        v_next = self.lyapunov(x_next)
        
        # 4. Tính Delta V và Y_0
        delta_v = v_next - v_t + self.rho * v_t
        y_0 = -delta_v # Y_0 phải là âm của Delta V để khớp logic SMT
        
        # Gom lại thành vector đầu ra [batch_size, 2]
        return torch.cat([y_0, v_t], dim=1)

def export_to_onnx(
    controller_path="checkpoints/pendulum/pendulum_controller.pth",
    lyapunov_path="checkpoints/pendulum/pendulum_lyapunov.pth",
    output_onnx="models/pendulum_system.onnx",
):
    device = torch.device("cpu") # ONNX export nên chạy trên CPU
    
    net_c = NeuralController(nx=2, nu=1, u_bound=6.0)
    net_v = NeuralLyapunov(nx=2)

    try:
        net_c.load_state_dict(torch.load(controller_path, map_location=device))
        net_v.load_state_dict(torch.load(lyapunov_path, map_location=device))
        print("Đã nạp thành công trọng số controller/lyapunov.")
    except FileNotFoundError as e:
        print(f"Không tìm thấy checkpoint: {e}")
        return
    
    dynamics = PendulumDynamics()
    graph = CompleteVerifierGraph(net_c, net_v, dynamics)
    graph.eval()

    # Tạo dummy tensor (Con lắc có 2 state)
    dummy_x = torch.zeros(1, 2)
    
    print("Đang xuất đồ thị ra định dạng ONNX...")
    torch.onnx.export(
        graph, 
        dummy_x, 
        output_onnx, 
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['X'],
        output_names=['Y'],
        dynamic_axes={'X': {0: 'batch_size'}, 'Y': {0: 'batch_size'}}
    )
    print(f"Hoàn tất! File lưu tại: {output_onnx}")

if __name__ == "__main__":
    export_to_onnx()