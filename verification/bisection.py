# verification/bisection.py
import os
import subprocess
import re
import logging
import torch
from verification.crown_interface import export_crown_artifacts

def extract_counter_example_from_crown(stdout_log, x_dim=2):
    """
    Trích xuất phản ví dụ từ log của tiến trình alpha,beta-CROWN.
    Tìm kiếm chuỗi định dạng giả định: "Counterexample found: [ 0.1234, -1.5678 ]"
    """
    pattern = r"Counterexample found:\s*\[(.*?)\]"
    match = re.search(pattern, stdout_log)
    if match:
        values_str = match.group(1).split(',')
        try:
            values = [float(v.strip()) for v in values_str]
            if len(values) == x_dim:
                return torch.tensor([values], dtype=torch.float32)
        except ValueError:
            logging.warning("Lỗi chuyển đổi kiểu dữ liệu khi parse phản ví dụ từ CROWN.")
            pass
            
    # Dự phòng: In ra 1000 ký tự cuối của log để debug nếu CROWN báo UNSAFE nhưng không tìm thấy mẫu
    logging.debug(f"Không thể trích xuất tọa độ CE từ log CROWN. Log cuối:\n{stdout_log[-1000:]}")
    return None

def verify_rho_with_crown(onnx_path: str, vnnlib_path: str, config_path: str, timeout_sec: int = 150) -> tuple:
    """
    Kích hoạt tiến trình alpha,beta-CROWN độc lập thông qua subprocess.
    """

    my_env = os.environ.copy()
    crown_repo_path = os.path.abspath("verification/complete_verifier")
    if "PYTHONPATH" in my_env:
        my_env["PYTHONPATH"] = f"{crown_repo_path}:{my_env['PYTHONPATH']}"
    else:
        my_env["PYTHONPATH"] = crown_repo_path

    command = [
        "python", "-m", "complete_verifier.abcrown",
        "--config", config_path,
        "--onnx_path", onnx_path,
        "--vnnlib_path", vnnlib_path
    ]
    
    try:
        # Thực thi command line và thu thập toàn bộ log đầu ra
        result = subprocess.run(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            timeout=timeout_sec
        )
        output = result.stdout
        
    except subprocess.TimeoutExpired:
        logging.warning(f"[CROWN] Quá thời gian quy định ({timeout_sec}s). Bộ giải không hội tụ kịp.")
        return "timeout", None
        
    # Phân tích trạng thái bài toán chứng minh
    if re.search(r"Result:\s*safe", output, re.IGNORECASE) or re.search(r"Result:\s*unsat", output, re.IGNORECASE):
        return "safe", None
        
    elif re.search(r"Result:\s*unsafe", output, re.IGNORECASE) or re.search(r"Result:\s*sat", output, re.IGNORECASE):
        ce_tensor = extract_counter_example_from_crown(output)
        return "unsafe", ce_tensor
        
    elif re.search(r"Result:\s*timeout", output, re.IGNORECASE):
        return "timeout", None
        
    else:
        # ---- SỬA ĐỔI QUAN TRỌNG Ở ĐÂY ----
        # Ghi toàn bộ log của CROWN ra một file text để chúng ta dễ dàng debug
        with open("crown_crash_log.txt", "w") as f:
            f.write(output)
            
        # Lấy 15 dòng cuối cùng của log CROWN để in thẳng ra terminal
        lines = output.strip().split('\n')
        tail_log = '\n'.join(lines[-15:]) if len(lines) > 0 else "Không có log đầu ra."
        
        logging.error(f"Tiến trình CROWN bị crash! Đã lưu toàn bộ log vào 'crown_crash_log.txt'.\n"
                      f"--- TRÍCH XUẤT 15 DÒNG LỖI CUỐI CÙNG TỪ CROWN ---\n"
                      f"{tail_log}\n"
                      f"--------------------------------------------------")
        return "unknown", None

def find_maximum_rho(system, controller, lyapunov, 
                     rho_min: float = 0.0, rho_max: float = 2.0, 
                     tolerance: float = 0.05) -> tuple:
    """
    Sử dụng thuật toán tìm kiếm nhị phân (Bisection) để xác định giới hạn ROA lớn nhất (rho)
    mà bộ giải CROWN có thể chứng minh là an toàn nghiêm ngặt (UNSAT).
    
    Returns:
        tuple: (Giá trị rho_max an toàn, 
                Tensor phản ví dụ cuối cùng tìm được tại vùng vi phạm gần ranh giới nhất)
    """
    config_path = "verification/crown_config.yaml"
    best_safe_rho = rho_min
    last_counter_example = None
    
    logging.info(f"Bắt đầu Bisection Verification: Miền tìm kiếm [{rho_min:.3f}, {rho_max:.3f}]")
    
    while (rho_max - rho_min) > tolerance:
        rho_mid = (rho_max + rho_min) / 2.0
        logging.info(f"Đang kiểm chứng tính ổn định tại ứng viên rho = {rho_mid:.4f}...")
        
        # 1. Fuse các module PyTorch và xuất ONNX graph tĩnh + file đặc tả VNNLIB
        onnx_path, vnnlib_path = export_crown_artifacts(system, controller, lyapunov, rho_mid)
        
        # 2. Giao phó cho bộ giải CROWN
        result, ce_tensor = verify_rho_with_crown(onnx_path, vnnlib_path, config_path)
        
        # 3. Cập nhật cận Bisection dựa trên chứng minh hình thức
        if result == "safe":
            logging.info(f"[+] rho = {rho_mid:.4f} ĐƯỢC CHỨNG NHẬN AN TOÀN. Mở rộng ranh giới ROA lên trên.")
            best_safe_rho = rho_mid
            rho_min = rho_mid
        else:
            # Nếu unsafe hoặc timeout, hệ thống chưa đủ khả năng bảo chứng tại mức năng lượng này
            logging.info(f"[-] rho = {rho_mid:.4f} KHÔNG THỂ BẢO CHỨNG ({result.upper()}). Thu hẹp ranh giới ROA xuống dưới.")
            rho_max = rho_mid
            
            # Cập nhật phản ví dụ để đưa về Replay Buffer của mạng nơ-ron
            if ce_tensor is not None:
                last_counter_example = ce_tensor
                
    logging.info(f"Bisection hoàn tất. ROA lớn nhất được chứng nhận: rho = {best_safe_rho:.4f}")
    
    # Chỉ trả về một lần ở cuối hàm, đảm bảo toàn bộ luồng logic đã thực thi
    return best_safe_rho, last_counter_example