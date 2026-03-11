import torch
import logging
from core.systems import InvertedPendulum
from core.models import Controller, LyapunovNetwork
from synthesis.attacks import PGDAttacker
from synthesis.trainer import CEGISTrainer
from verification.bisection import find_maximum_rho

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    # 1. Khởi tạo Không gian và Động học
    system = InvertedPendulum()
    controller = Controller()
    lyapunov = LyapunovNetwork()
    
    # 2. Cấu hình Siêu tham số
    config = {
        'rho': 0.1,             # Giá trị ROA khởi điểm
        'kappa': 0.1,           # Tốc độ giảm năng lượng tối thiểu
        'epochs': 50,           # Số epoch huấn luyện trong 1 vòng lặp CEGIS
        'batch_size': 5000,
        'lambda_margin': 100.0
    }
    
    attacker = PGDAttacker(system, controller, lyapunov, config)
    trainer = CEGISTrainer(system, controller, lyapunov, attacker, config)
    
    MAX_CEGIS_ITERS = 20
    target_rho = config['rho']
    
    for iteration in range(MAX_CEGIS_ITERS):
        logging.info(f"\n{'='*20} VÒNG LẶP CEGIS {iteration + 1} {'='*20}")
        logging.info(f"Mục tiêu Huấn luyện: Mở rộng ROA tới rho = {target_rho:.4f}")
        
        # --- GIAI ĐOẠN 1: SYNTHESIS (HUẤN LUYỆN) ---
        trainer.rho = target_rho
        trainer.train()  # Gọi hàm train() bạn đã viết trong trainer.py
        
        # --- GIAI ĐOẠN 2: VERIFICATION (KIỂM CHỨNG HÌNH THỨC) ---
        logging.info("Kích hoạt α,β-CROWN để bảo chứng miền ROA hiện tại...")
        controller.eval()
        lyapunov.eval()
        
        # Tìm kiếm giới hạn an toàn thực tế của hệ thống. 
        # Cận trên (rho_max) được đặt lớn hơn target_rho để kiểm tra xem mạng có học vượt kỳ vọng không.
        verified_rho = find_maximum_rho(
            system, controller, lyapunov, 
            rho_min=0.0, rho_max=target_rho * 1.5, 
            tolerance=0.05
        )
        
        logging.info(f"-> ROA lớn nhất được CHỨNG NHẬN: rho = {verified_rho:.4f}")
        
        # --- GIAI ĐOẠN 3: CẬP NHẬT TRẠNG THÁI (CEGIS FEEDBACK) ---
        if verified_rho >= target_rho:
            # Thuật toán thành công trong việc bảo chứng mức rho hiện tại.
            # Tiến hành mở rộng giới hạn ROA cho vòng lặp tiếp theo.
            logging.info("Chứng minh thành công! Tăng mục tiêu ROA.")
            target_rho = verified_rho + 0.2
        else:
            # CROWN tìm thấy điểm vi phạm ở target_rho hiện tại.
            # Mạng cần tiếp tục củng cố vùng hiện tại thay vì mở rộng.
            logging.info("Chưa đạt mục tiêu. Hệ thống sẽ củng cố trọng số trong vòng lặp tới.")
            
            # (Toán học cốt lõi): Tại đây, CROWN đã tìm thấy phản ví dụ. 
            # Cần bổ sung logic trích xuất các điểm X này từ CROWN để đưa vào bộ đệm của PGDAttacker.

if __name__ == "__main__":
    main()