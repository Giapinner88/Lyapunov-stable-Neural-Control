import torch
import argparse
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
# Import các module cốt lõi của bạn (giả sử tên hàm và class như sau)
from core.models import NeuralController, NeuralLyapunov
from core.dynamics import PendulumDynamics
from core.verifier import SystemViolationGraph # Lớp ta vừa thảo luận ở bước trước

def build_bounded_model(device='cpu'):
    # 1. Khởi tạo các module như Phase 1/2
    net_c = NeuralController(nx=2, nu=1).to(device)
    net_v = NeuralLyapunov(nx=2).to(device)

    # Nạp tri thức đã học từ quá trình huấn luyện
    try:
        net_c.load_state_dict(torch.load("pendulum_controller.pth", map_location=device))
        net_v.load_state_dict(torch.load("pendulum_lyapunov.pth", map_location=device))
        print("-> Đã nạp thành công trọng số mô hình.")
    except FileNotFoundError:
        print("-> CẢNH BÁO: Chưa tìm thấy file trọng số. Đang verify mạng ngẫu nhiên!")

    dynamics = PendulumDynamics().to(device)

    # 2. Khởi tạo đồ thị vi phạm
    model = SystemViolationGraph(net_c, net_v, dynamics).to(device)
    model.eval()

    # 3. Tạo dummy input để auto_LiRPA tracing đồ thị
    dummy_x = torch.zeros(1, 2, device=device)

    print("[1/2] Đang bọc đồ thị bằng BoundedModule...")
    try:
        bounded_model = BoundedModule(model, dummy_x, bound_opts={'relu': 'adaptive'})
        print("-> Tracing thành công! Đồ thị RK4 hoàn toàn tương thích.")
    except Exception as e:
        print(f"-> THẤT BẠI khi Tracing. Lỗi: {e}")
        return None

    return bounded_model, dummy_x


def compute_bounds_for_eps(bounded_model, dummy_x, eps):
    ptb = PerturbationLpNorm(norm=torch.inf, eps=eps)
    bounded_x = BoundedTensor(dummy_x, ptb)
    with torch.no_grad():
        lb, ub = bounded_model.compute_bounds(x=(bounded_x,), method='CROWN')
    return lb.item(), ub.item()


def test_graph_tracing(eps=0.1):
    device = 'cpu'
    result = build_bounded_model(device=device)
    if result is None:
        return
    bounded_model, dummy_x = result

    print("[2/2] Đang lan truyền ranh giới (Bound Propagation)...")
    try:
        lb, ub = compute_bounds_for_eps(bounded_model, dummy_x, eps)
        print(f"-> Phân tích thành công! Ranh giới Violation: Lower Bound={lb:.4f}, Upper Bound={ub:.4f}")
    except Exception as e:
        print(f"-> THẤT BẠI khi tính toán ranh giới. Lỗi: {e}")


def sweep_eps(eps_max=0.2, eps_min=0.005, steps=15):
    if eps_max <= eps_min:
        raise ValueError("eps_max phải lớn hơn eps_min")
    if steps < 2:
        raise ValueError("steps phải >= 2")

    result = build_bounded_model(device='cpu')
    if result is None:
        return
    bounded_model, dummy_x = result

    eps_values = torch.linspace(eps_max, eps_min, steps).tolist()
    first_negative = None

    print("[2/2] Quét eps để tìm Upper Bound âm...")
    for eps in eps_values:
        lb, ub = compute_bounds_for_eps(bounded_model, dummy_x, eps)
        print(f"eps={eps:.6f} -> LB={lb:.6f}, UB={ub:.6f}")
        if ub < 0 and first_negative is None:
            first_negative = (eps, lb, ub)

    if first_negative is None:
        print("=> Chưa tìm thấy UB < 0 trong dải eps đã quét.")
    else:
        eps, lb, ub = first_negative
        print(f"=> eps đầu tiên cho UB < 0: eps={eps:.6f} (LB={lb:.6f}, UB={ub:.6f})")


def parse_args():
    parser = argparse.ArgumentParser(description="Kiểm tra tương thích RK4 và quét eps cho CROWN bound")
    parser.add_argument("--eps", type=float, default=0.1, help="eps dùng cho run đơn")
    parser.add_argument("--sweep", action="store_true", help="Bật chế độ quét eps")
    parser.add_argument("--eps-max", type=float, default=0.2, help="eps lớn nhất khi quét")
    parser.add_argument("--eps-min", type=float, default=0.005, help="eps nhỏ nhất khi quét")
    parser.add_argument("--steps", type=int, default=15, help="số điểm quét")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.sweep:
        sweep_eps(eps_max=args.eps_max, eps_min=args.eps_min, steps=args.steps)
    else:
        test_graph_tracing(eps=args.eps)