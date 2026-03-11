# train_and_verify.py

import torch
import logging
from argparse import Namespace
from core.systems import InvertedPendulum
from core.models import Controller, LyapunovNetwork
from synthesis.attacks import PGDAttacker
from synthesis.trainer import CEGISTrainer
from verification.bisection import find_maximum_rho

# Thiết lập hệ thống log để theo dõi tiến trình CEGIS
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    # 1. Khởi tạo Không gian và Động học
    system = InvertedPendulum()
    controller = Controller()
    lyapunov = LyapunovNetwork()
    
    # 2. Cấu hình Siêu tham số (Sử dụng Namespace để hỗ trợ truy cập bằng dấu chấm)
    config = Namespace(
        rho=0.1,                    # Giá trị ROA khởi điểm
        kappa=0.1,                  # Tốc độ giảm năng lượng tối thiểu
        epochs=50,                  # Số epoch huấn luyện trong 1 vòng lặp CEGIS
        batch_size=5000,
        lambda_margin=100.0,
        
        # Tham số chuyên biệt cho PGD Attacker
        state_bounds=system.x_bounds, # Trích xuất giới hạn vật lý từ hệ thống
        pgd_steps=10,                 # Số bước lặp tối ưu cục bộ của PGD
        pgd_step_size=0.01            # Bước nhảy alpha
    )
    
    # 3. Khởi tạo các module tổng hợp
    attacker = PGDAttacker(system, controller, lyapunov, config)
    trainer = CEGISTrainer(system, controller, lyapunov, attacker, config)
    
    MAX_CEGIS_ITERS = 20
    target_rho = config.rho
    
    for iteration in range(MAX_CEGIS_ITERS):
        logging.info(f"\n{'='*20} VÒNG LẶP CEGIS {iteration + 1} {'='*20}")
        logging.info(f"Mục tiêu Huấn luyện: Mở rộng ROA tới rho = {target_rho:.4f}")
        
        # --- GIAI ĐOẠN 1: SYNTHESIS (HUẤN LUYỆN) ---
        trainer.rho = target_rho
        # Mạng nơ-ron học cách ổn định hệ thống và loại bỏ vi phạm F(x)
        trainer.train()  
        
        # --- GIAI ĐOẠN 2: VERIFICATION (KIỂM CHỨNG HÌNH THỨC) ---
        logging.info("Kích hoạt α,β-CROWN để bảo chứng miền ROA hiện tại...")
        controller.eval()
        lyapunov.eval()
        
        # Tìm kiếm giới hạn an toàn thực tế của hệ thống. 
        # Cận trên (rho_max) được đặt ở target_rho * 1.5 để kiểm tra xem mạng có học vượt mức không.
        verified_rho, formal_ce = find_maximum_rho(
            system, controller, lyapunov, 
            rho_min=0.0, rho_max=target_rho * 1.5, 
            tolerance=0.05
        )
        
        logging.info(f"-> ROA lớn nhất được CHỨNG NHẬN: rho = {verified_rho:.4f}")
        
        # --- GIAI ĐOẠN 3: CẬP NHẬT TRẠNG THÁI (CEGIS FEEDBACK) ---
        if verified_rho >= target_rho:
            # Thuật toán thành công trong việc bảo chứng mức rho hiện tại.
            logging.info("Chứng minh thành công! Tăng mục tiêu ROA cho vòng lặp tiếp theo.")
            target_rho = verified_rho + 0.2
        else:
            # CROWN tìm thấy điểm vi phạm ở target_rho hiện tại.
            logging.info("Chưa đạt mục tiêu. Hệ thống sẽ củng cố trọng số tại các điểm mù hình thức.")
            
            # Đưa phản ví dụ (Counter-Example) từ CROWN vào Replay Buffer của Attacker
            if formal_ce is not None:
                attacker.add_formal_counter_examples(formal_ce)
                logging.info(f"Đã thêm điểm mù {formal_ce.numpy()} vào Replay Buffer.")

if __name__ == "__main__":
    main()