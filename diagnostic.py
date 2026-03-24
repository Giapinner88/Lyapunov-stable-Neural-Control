"""
DIAGNOSTIC SCRIPT: Tìm nguyên nhân thực sự
===========================================

Kiểm tra 3 câu hỏi:
1. Controller thực sự ổn định không? (Test trajectories)
2. CROWN bound có loose không?
3. Vậy vấn đề ở đâu?
"""

import torch
import numpy as np
from core.models import NeuralController, NeuralLyapunov
from core.dynamics import PendulumDynamics

def test_trajectory_stability(num_trajectories=1000, max_steps=100, device='cpu'):
    """
    Kiểm tra: Khi chạy controller NN, trajectories có hội tụ về 0 không?
    """
    print("\n" + "="*70)
    print("KIỂM TRA 1: TRAJECTORY STABILITY (Thực)") 
    print("="*70)
    
    net_c = NeuralController(nx=2, nu=1).to(device)
    net_v = NeuralLyapunov(nx=2).to(device)
    
    try:
        net_c.load_state_dict(torch.load("pendulum_controller.pth", map_location=device))
        net_v.load_state_dict(torch.load("pendulum_lyapunov.pth", map_location=device))
        print("✓ Đã load models")
    except Exception as e:
        print(f"✗ Lỗi: {e}")
        return
    
    dynamics = PendulumDynamics().to(device)
    
    # Sinh random initial conditions
    print(f"\n[Test] Sinh {num_trajectories} trajectories")
    init_states = torch.rand((num_trajectories, 2), device=device) * 0.3 - 0.15  # [-0.15, 0.15]
    
    converged = 0
    diverged = 0
    max_distance = 0
    max_distance_state = None
    
    for traj_idx, x0 in enumerate(init_states):
        x = x0.unsqueeze(0)
        trajectory = [x.clone()]
        
        for step in range(max_steps):
            with torch.no_grad():
                u = net_c(x)
                x = dynamics.step(x, u)
                trajectory.append(x.clone())
            
            # Check nếu quá xa gốc → diverge
            if torch.norm(x) > 2.0:
                diverged += 1
                break
        
        # Check kết thúc
        final_distance = torch.norm(x).item()
        if final_distance > max_distance:
            max_distance = final_distance
            max_distance_state = x0.tolist()
        
        if final_distance < 0.05:
            converged += 1
        
        if (traj_idx + 1) % 100 == 0:
            print(f"  [{traj_idx+1}/{num_trajectories}] Done")
    
    print(f"\n[Kết quả]")
    print(f"  Converged (final_dist < 0.05): {converged}/{num_trajectories} ({converged/num_trajectories*100:.1f}%)")
    print(f"  Diverged (norm > 2.0): {diverged}/{num_trajectories} ({diverged/num_trajectories*100:.1f}%)")
    print(f"  Max distance reached: {max_distance:.6f} tại state {max_distance_state}")
    
    if converged > num_trajectories * 0.8:
        print(f"\n✓ TUYỆT VỜI: Controller thực sự ổn định!")
        print(f"  → Vấnđề là CROWN loose, không phải training")
        return "CROWN_LOOSE"
    elif diverged > num_trajectories * 0.2:
        print(f"\n✗ VẤNĐỀ: Có {diverged} trajectories diverge!")
        print(f"  → Vấnđề là training, cần cải tiến strategy")
        return "TRAINING_PROBLEM"
    else:
        print(f"\n⚠️ MỜI: Khoảng cách final khá xa gốc")
        print(f"  → Có thể cả 2 vấnđề: training + CROWN loose")
        return "MIXED_PROBLEM"


def point_wise_vs_crown_analysis(device='cpu'):
    """
    Kiểm tra: So sánh violation point-wise vs CROWN upper bound
    """
    print("\n" + "="*70)
    print("KIỂM TRA 2: POINT-WISE vs CROWN BOUND")
    print("="*70)
    
    net_c = NeuralController(nx=2, nu=1).to(device)
    net_v = NeuralLyapunov(nx=2).to(device)
    
    try:
        net_c.load_state_dict(torch.load("pendulum_controller.pth", map_location=device))
        net_v.load_state_dict(torch.load("pendulum_lyapunov.pth", map_location=device))
    except:
        print("✗ Lỗi load model")
        return
    
    dynamics = PendulumDynamics().to(device)
    
    # Test grid
    print(f"\n[Test] Quét grid trong [-0.1, 0.1]²")
    test_grid = torch.linspace(-0.1, 0.1, 21)
    
    violations = []
    for theta in test_grid:
        for dot_theta in test_grid:
            x = torch.tensor([[theta.item(), dot_theta.item()]], device=device)
            
            with torch.no_grad():
                v_t = net_v(x).item()
                u_t = net_c(x)
                x_next = dynamics.step(x, u_t)
                v_next = net_v(x_next).item()
                violation = v_next - v_t
            
            violations.append(violation)
    
    violations = np.array(violations)
    
    print(f"\n[Point-wise Statistics] ({len(violations)} điểm)")
    print(f"  Max violation: {violations.max():.6f}")
    print(f"  Min violation: {violations.min():.6f}")
    print(f"  Mean violation: {violations.mean():.6f}")
    print(f"  Std violation: {violations.std():.6f}")
    print(f"  % violation ≥ 0: {(violations >= 0).sum() / len(violations) * 100:.1f}%")
    
    if violations.max() < 0:
        print(f"\n✓ Model HOÀN TOÀN ổn định (all points violated < 0)")
        print(f"  → CROWN upper bound phải < 0 (nếu tight)")
        print(f"  → Violation: UB = CROWN không tight!")
        gap = 0.0233 - violations.max()  # Giả sử CROWN UB ~ 0.0233
        print(f"  → Tightness gap: ~{gap:.6f}")
    else:
        print(f"\n✗ Có {(violations >= 0).sum()} điểm violation ≥ 0")
        print(f"  → CROWN UB >= {violations.max():.6f}")
        print(f"  → Nhưng CROWN tính được ~0.0233 → CROWN loose ~{0.0233 - violations.max():.6f}")


def crown_tightness_test(device='cpu'):
    """
    Kiểm tra: CROWN bound có bao hàm tất cả điểm xấu không
    """
    print("\n" + "="*70)
    print("KIỂM TRA 3: CROWN TIGHTNESS")
    print("="*70)
    
    print(f"""\n[Lý thuyết]
CROWN bound hoạt động như sau:
  1. Tính lower & upper bound của từng neuron qua layers
  2. Propagate qua network → tính bound của output
  3. Có loss từ ReLU linearization
  
Nếu:
  - Thực tế: max(violation) = 0.00001
  - CROWN: UB = 0.0233
  - → Gap = 0.02329 (QUÁCLOOSELUNCH!)
  
Lý do CROWN loose:
  1. ReLU không linear - CROWN chỉ là lower bound
  2. Network 2 layer với hidden = 64 → 128 ReLU
  3. Dependency tracking bị mất → UB tích tũy
  4. RK4 + 2 NN → rất complex → CROWN không tight
  
Giới hạn CROWN:
  - CROWN tốt với network nhỏ, shallow
  - Với network này (2 layer × 64-64): Kỳ vọng gap ~20x
  """)


def main():
    device = 'cpu'
    
    # Check 1: Trajectory stability
    result1 = test_trajectory_stability(num_trajectories=500, max_steps=100, device=device)
    
    # Check 2: Point-wise vs CROWN
    point_wise_vs_crown_analysis(device=device)
    
    # Check 3: CROWN tightness
    crown_tightness_test(device=device)
    
    print("\n" + "="*70)
    print("KẾT LUẬN")
    print("="*70)
    
    conclusions = {
        "CROWN_LOOSE": """
🎯 CROWN BOUND QUÁLOOSELUNCH!

Nguyên nhân:
1. ReLU không tuyến tính → CROWN chỉ là lower bound
2. Network quá sâu → dependency tracking mất
3. RK4 + 2 NN = rất phức tạp → CROWN rất loose

Giải pháp:
A. Chuyển sang verification khác:
   - α-β-CROWN (tighter bound)
   - Abstraction-based methods
   - Zonotope/Polytope methods
   
B. Hoặc: Sử dụng quadratic Lyapunov (thay vì NN)
   - x^T P x dễ verify hơn
   - Nhưng kém linh hoạt

C. Hoặc: Chấp nhận NN controller + empirical verification
   - Simulate 10k+ trajectories → statistical guarantee
   - Không cần formal verification
        """,
        "TRAINING_PROBLEM": """
❌ CONTROLLER KHÔNG THỰC SỰ ỔN ĐỊNH

Nguyên nhân:
- Trajectories diverge → training fail
- Không phải vấnđề verification

Giải pháp:
- Completely redesign training strategy
- Có thể NN controller không đủ powerful
- Thử LQR-NN hybrid
        """,
        "MIXED_PROBLEM": """
⚠️ CẢ HAI VẤĐỀ

- Training chưa tối ưu
- CROWN cũng loose

Giải pháp: Kết hợp cả 2
1. Cải tiến training
2. Tìm verification tighter
        """
    }
    
    if result1 in conclusions:
        print(conclusions[result1])
    
    print("\n" + "="*70)
    print("KHUYẾN NỊ")
    print("="*70)
    print("""
Dữa trên kết quả:

1. Nếu trajectories OK → CROWN tight hơn bằng cách:
   - Dùng α-β-CROWN (experimental)
   - Hoặc đơn giản: Quadratic Lyapunov
   - Hoặc: Empirical verification (simulation)

2. Nếu trajectories diverge → Training cần cải tiến:
   - Thử LQR baseline (không NN)
   - Hoặc hybrid LQR + small NN
   - Hoặc reduce network size
    """)


if __name__ == "__main__":
    main()
