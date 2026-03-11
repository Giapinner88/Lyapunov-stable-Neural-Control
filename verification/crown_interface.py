import torch
import torch.nn as nn
import os

class FusedLyapunovGraph(nn.Module):
    def __init__(self, system, controller, lyapunov, kappa=0.1):
        super().__init__()
        self.system = system
        self.controller = controller
        self.lyapunov = lyapunov
        self.kappa = kappa

    def forward(self, x):
        u = self.controller(x)
        x_next = self.system(x, u)
        
        v_curr = self.lyapunov(x)
        v_next = self.lyapunov(x_next)
        
        # KHỬ FLOAT64
        factor = torch.tensor(1.0 - self.kappa, dtype=torch.float32, device=x.device)
        f_x = v_next - factor * v_curr
        
        return torch.cat([f_x, v_curr], dim=1)

def export_crown_artifacts(system, controller, lyapunov, rho, 
                           onnx_path="verification/specs/model.onnx", 
                           vnnlib_path="verification/specs/condition.vnnlib"):
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    fused_model = FusedLyapunovGraph(system, controller, lyapunov)
    fused_model.eval()
    
    dummy_input = torch.zeros(1, system.x_dim, dtype=torch.float32)
    
    torch.onnx.export(
        fused_model, dummy_input, onnx_path,
        input_names=['x'], output_names=['y'],
        opset_version=12,  # Đưa về 12 do đồ thị đã thuần float32
        do_constant_folding=True
    )
    
    with open(vnnlib_path, 'w') as f:
        f.write("; Khai báo biến đầu vào X (Trạng thái)\n")
        f.write("(declare-const X_0 Real)\n") 
        f.write("(declare-const X_1 Real)\n") 
        f.write("\n; Khai báo biến đầu ra Y\n")
        f.write("(declare-const Y_0 Real)\n") 
        f.write("(declare-const Y_1 Real)\n") 
        f.write("\n; Giới hạn vật lý của trạng thái (Preconditions)\n")
        x_bounds = system.x_bounds.numpy()
        f.write(f"(assert (>= X_0 {x_bounds[0][0]}))\n")
        f.write(f"(assert (<= X_0 {x_bounds[0][1]}))\n")
        f.write(f"(assert (>= X_1 {x_bounds[1][0]}))\n")
        f.write(f"(assert (<= X_1 {x_bounds[1][1]}))\n")
        f.write("\n; Điều kiện Vi phạm (Postconditions / Target)\n")
        f.write(f"(assert (<= Y_1 {rho}))\n")
        f.write("(assert (>= Y_0 0.0))\n")

    return onnx_path, vnnlib_path