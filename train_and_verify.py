import torch
import os

# Import các module từ kiến trúc hệ thống
from core.systems import InvertedPendulum
from core.models import Controller, LyapunovNetwork
from synthesis.attacks import PGDAttacker
from synthesis.trainer import CEGISTrainer
from verification.crown_interface import VerificationWrapper
from verification.bisection import ROABisector

def main():
    print("="*60)
    print("HỆ THỐNG ĐIỀU KHIỂN NƠ-RON ỔN ĐỊNH LYAPUNOV (CEGIS PIPELINE)")
    print("="*60)

    # 1. Cấu hình Siêu tham số (Hyperparameters Configuration)
    config = {
        'x_dim': 2,
        'u_dim': 1,
        'kappa': 0.1,             # Tốc độ giảm năng lượng bắt buộc
        'rho_init': 0.05,         # ROA mục tiêu ban đầu (rất nhỏ)
        'epochs_per_cycle': 200,  # Số epoch huấn luyện trước mỗi lần xác minh
        'batch_size': 2000,
        'pgd_steps': 15,
        'pgd_step_size': 0.05,
        'lambda_margin': 100.0,
        'max_cegis_iters': 5      # Số lần lặp vòng lặp lớn (Train -> Verify -> Bisection)
    }

    # Định nghĩa giới hạn không gian vật lý B (thay đổi tùy hệ thống)
    # Pendulum: theta thuộc [-pi, pi], dot_theta thuộc [-8, 8]
    import math
    x_bounds = [(-math.pi, math.pi), (-8.0, 8.0)]
    config['state_bounds'] = torch.tensor(x_bounds)

    # 2. Khởi tạo Đồ thị Tính toán (Instantiating the Computational Graph)
    print("\n[*] Đang khởi tạo mô hình toán học...")
    system = InvertedPendulum(dt=0.05)
    controller = Controller(x_dim=config['x_dim'], u_dim=config['u_dim'])
    lyapunov = LyapunovNetwork(x_dim=config['x_dim'])

    # Khởi tạo các module chuyên trách
    attacker = PGDAttacker(system, controller, lyapunov, config)
    trainer = CEGISTrainer(system, controller, lyapunov, attacker, config)
    
    # Module đóng gói cho CROWN
    verifier_wrapper = VerificationWrapper(system, controller, lyapunov, kappa=config['kappa'])
    bisector = ROABisector(verifier_wrapper, x_bounds, crown_config_path="verification/crown_config.yaml")

    # 3. Vòng lặp CEGIS Cấp cao (Outer CEGIS Loop)
    current_rho = config['rho_init']

    for cegis_iter in range(1, config['max_cegis_iters'] + 1):
        print(f"\n{'-'*40}")
        print(f"VÒNG LẶP CEGIS #{cegis_iter} | MỤC TIÊU ROA (\u03C1) = {current_rho:.4f}")
        print(f"{'-'*40}")

        # GIAI ĐOẠN A: TỔNG HỢP (SYNTHESIS)
        print("[+] Giai đoạn 1: Huấn luyện (Synthesis) bằng PGD Attack...")
        # Cập nhật mức rho mục tiêu cho Trainer
        trainer.rho = current_rho 
        trainer.train(epochs=config['epochs_per_cycle'])

        # Lưu trọng số tạm thời
        os.makedirs("results", exist_ok=True)
        torch.save(controller.state_dict(), f"results/controller_iter_{cegis_iter}.pth")
        torch.save(lyapunov.state_dict(), f"results/lyapunov_iter_{cegis_iter}.pth")

        # GIAI ĐOẠN B: XÁC MINH VÀ MỞ RỘNG (VERIFICATION & BISECTION)
        print("\n[+] Giai đoạn 2: Chứng minh Hình thức (Formal Verification)...")
        # Sử dụng Bisection để tìm ROA tối đa được chứng nhận cho bộ trọng số hiện tại
        verified_rho = bisector.find_max_roa(rho_min=0.01, rho_max=current_rho * 2.0)

        if verified_rho >= current_rho:
            print(f"[*] THÀNH CÔNG: Mạng nơ-ron đã ổn định toàn bộ vùng \u03C1 = {current_rho:.4f}.")
            # Chiến lược nhồi (Curriculum Learning): Tăng rho mục tiêu cho vòng lặp tiếp theo
            current_rho = verified_rho + 0.1 
        else:
            print(f"[*] CẢNH BÁO: Chỉ chứng minh được đến \u03C1 = {verified_rho:.4f}. Chưa đạt mục tiêu.")
            # Giữ nguyên hoặc giảm nhẹ rho mục tiêu để mạng học kỹ hơn vùng không gian này
            current_rho = max(verified_rho, 0.05) + 0.05

    print("\n" + "="*60)
    print("HOÀN TẤT QUÁ TRÌNH HUẤN LUYỆN VÀ CHỨNG MINH.")
    print(f"Kích thước ROA (\u03C1) cuối cùng đạt được: {verified_rho:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()