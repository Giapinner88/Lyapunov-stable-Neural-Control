import torch
import torch.nn as nn
import os
import math

class FusedLyapunovGraph(nn.Module):
    def __init__(self, system, controller, lyapunov, kappa=0.1):
        super().__init__()
        self.system = system
        self.controller = controller
        self.lyapunov = lyapunov
        self.kappa = kappa

    def forward(self, x):
        """
        Đầu vào: x (batch_size, 2)
        Đầu ra: [F(x), V(x)] (batch_size, 2)
        Y_0 = F(x), Y_1 = V(x)
        """
        u = self.controller(x)
        x_next = self.system(x, u)
        
        v_curr = self.lyapunov(x)
        v_next = self.lyapunov(x_next)
        
        # F(x) = V(x_next) - (1 - kappa) * V(x)
        f_x = v_next - (1 - self.kappa) * v_curr
        
        return torch.cat([f_x, v_curr], dim=1)

def export_crown_artifacts(system, controller, lyapunov, rho, 
                           onnx_path="verification/specs/model.onnx", 
                           vnnlib_path="verification/specs/condition.vnnlib"):
    """Xuất mô hình ONNX và sinh file đặc tả VNNLIB cho CROWN."""
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # 1. Xuất ONNX
    fused_model = FusedLyapunovGraph(system, controller, lyapunov)
    fused_model.eval()
    dummy_input = torch.zeros(1, system.x_dim)
    
    torch.onnx.export(
        fused_model, dummy_input, onnx_path,
        input_names=['x'], output_names=['y'],
        opset_version=12 # Opset 12 thường ổn định nhất với CROWN
    )
    
    # 2. Sinh VNNLIB
    # Mục tiêu của CROWN là tìm PHẢN VÍ DỤ. 
    # Do đó ta yêu cầu CROWN tìm x sao cho: (x trong giới hạn vật lý) VÀ (V(x) <= rho) VÀ (F(x) > 0)
    with open(vnnlib_path, 'w') as f:
        f.write("; Khai báo biến đầu vào X (Trạng thái)\n")
        f.write("(declare-const X_0 Real)\n") # theta
        f.write("(declare-const X_1 Real)\n") # dot_theta
        
        f.write("\n; Khai báo biến đầu ra Y\n")
        f.write("(declare-const Y_0 Real)\n") # F(x)
        f.write("(declare-const Y_1 Real)\n") # V(x)
        
        f.write("\n; Giới hạn vật lý của trạng thái (Preconditions)\n")
        x_bounds = system.x_bounds.numpy()
        f.write(f"(assert (>= X_0 {x_bounds[0][0]}))\n")
        f.write(f"(assert (<= X_0 {x_bounds[0][1]}))\n")
        f.write(f"(assert (>= X_1 {x_bounds[1][0]}))\n")
        f.write(f"(assert (<= X_1 {x_bounds[1][1]}))\n")
        
        f.write("\n; Điều kiện Vi phạm (Postconditions / Target)\n")
        f.write(f"; 1. Nằm trong vùng ROA: V(x) <= {rho}\n")
        f.write(f"(assert (<= Y_1 {rho}))\n")
        f.write(f"; 2. F(x) dương (Mất ổn định)\n")
        f.write("(assert (>= Y_0 0.0))\n") # Nếu CROWN chứng minh được điều này là UNSAT, hệ thống an toàn!

    return onnx_path, vnnlib_path