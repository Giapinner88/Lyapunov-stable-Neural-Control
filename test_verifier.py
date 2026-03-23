import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
# Import các module cốt lõi của bạn (giả sử tên hàm và class như sau)
from core.models import NeuralController, NeuralLyapunov
from core.dynamics import PendulumDynamics
from core.verifier import SystemViolationGraph # Lớp ta vừa thảo luận ở bước trước

def test_graph_tracing():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Khởi tạo các module như Phase 1/2
    # Thay giá trị của nx và nu bằng kích thước thực tế của hệ thống bạn đang xét
    net_c = NeuralController(nx=2, nu=1).to('cpu') # Chạy trên CPU để dễ dàng debug tracing
    net_v = NeuralLyapunov(nx=2).to('cpu')
    dynamics = PendulumDynamics()
    
    # 2. Khởi tạo đồ thị vi phạm
    model = SystemViolationGraph(net_c, net_v, dynamics).to('cpu')
    model.eval() # Bắt buộc phải là eval() để bỏ qua Dropout/BatchNorm nếu có
    
    # 3. Tạo một dummy input để auto_LiRPA dò đồ thị (Tracing)
    dummy_x = torch.zeros(1, 2, device='cpu') # Con lắc có 2 state: [theta, theta_dot]
    
    print("[1/2] Đang bọc đồ thị bằng BoundedModule...")
    try:
        # adaptive: Tự động tối ưu hóa các đường kẹp tuyến tính (CROWN)
        bounded_model = BoundedModule(model, dummy_x, bound_opts={'relu': 'adaptive'})
        print("-> Tracing thành công! Đồ thị RK4 hoàn toàn tương thích.")
    except Exception as e:
        print(f"-> THẤT BẠI khi Tracing. Lỗi: {e}")
        return

    # 4. Tạo một không gian nhiễu (Ví dụ: hộp bán kính 0.1 quanh gốc tọa độ)
    print("[2/2] Đang lan truyền ranh giới (Bound Propagation)...")
    ptb = PerturbationLpNorm(norm=torch.inf, eps=0.1)
    bounded_x = BoundedTensor(dummy_x, ptb)
    
    # 5. Khởi chạy CROWN
    try:
        with torch.no_grad():
            lb, ub = bounded_model.compute_bounds(x=(bounded_x,), method='CROWN')
        print(f"-> Phân tích thành công! Ranh giới Violation: Lower Bound={lb.item():.4f}, Upper Bound={ub.item():.4f}")
    except Exception as e:
        print(f"-> THẤT BẠI khi tính toán ranh giới. Lỗi: {e}")

if __name__ == "__main__":
    test_graph_tracing()