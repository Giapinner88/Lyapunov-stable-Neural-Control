import subprocess
import re
import logging
from verification.crown_interface import export_crown_artifacts
import re
import torch

def extract_counter_example_from_crown(stdout_log, x_dim=2):
    """
    Trích xuất phản ví dụ từ log của alpha,beta-CROWN.
    Đầu ra giả định của CROWN: "Counterexample found: [ 0.1234, -1.5678 ]"
    """
    pattern = r"Counterexample found:\s*\[(.*?)\]"
    match = re.search(pattern, stdout_log)
    if match:
        values_str = match.group(1).split(',')
        values = [float(v.strip()) for v in values_str]
        if len(values) == x_dim:
            return torch.tensor([values], dtype=torch.float32)
    return None

def verify_rho_with_crown(onnx_path: str, vnnlib_path: str, config_path: str, timeout_sec: int = 150) -> str:
    """Kích hoạt tiến trình alpha,beta-CROWN độc lập."""
    command = [
        "python", "-m", "complete_verifier.abcrown",
        "--config", config_path,
        "--onnx_path", onnx_path,
        "--vnnlib_path", vnnlib_path
    ]
    
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, timeout=timeout_sec
        )
        output = result.stdout
    except subprocess.TimeoutExpired:
        logging.warning(f"[CROWN] Quá thời gian quy định ({timeout_sec}s).")
        return "timeout"
        
    if re.search(r"Result:\s*safe", output, re.IGNORECASE) or re.search(r"Result:\s*unsat", output, re.IGNORECASE):
        # CROWN không tìm thấy vi phạm F(x) > 0 -> HỆ THỐNG AN TOÀN
        return "safe"
    elif re.search(r"Result:\s*unsafe", output, re.IGNORECASE) or re.search(r"Result:\s*sat", output, re.IGNORECASE):
        # CROWN tìm thấy vi phạm -> CÓ LỖI (Cần mở rộng tập huấn luyện)
        return "unsafe"
    elif re.search(r"Result:\s*timeout", output, re.IGNORECASE):
        return "timeout"
    else:
        logging.error("Lỗi phân tích cú pháp từ CROWN log.")
        return "unknown"

def find_maximum_rho(system, controller, lyapunov, 
                     rho_min: float = 0.0, rho_max: float = 2.0, 
                     tolerance: float = 0.05) -> float:
    """
    Thuật toán Bisection tìm ROA lớn nhất.
    """
    config_path = "verification/crown_config.yaml" # Tệp YAML tôi đã đề xuất ở trước
    best_safe_rho = rho_min
    
    logging.info(f"Bắt đầu Bisection: [{rho_min:.3f}, {rho_max:.3f}]")
    
    while (rho_max - rho_min) > tolerance:
        rho_mid = (rho_max + rho_min) / 2.0
        logging.info(f"Đang kiểm chứng ứng viên rho = {rho_mid:.4f}...")
        
        # 1. Xuất mô hình & VNNLIB động
        onnx_path, vnnlib_path = export_crown_artifacts(system, controller, lyapunov, rho_mid)
        
        # 2. Chạy CROWN
        result = verify_rho_with_crown(onnx_path, vnnlib_path, config_path)
        
        # 3. Cập nhật Bisection
        if result == "safe":
            logging.info(f"[+] rho = {rho_mid:.4f} AN TOÀN (UNSAT). Mở rộng ROA.")
            best_safe_rho = rho_mid
            rho_min = rho_mid
        else:
            # unsafe hoặc timeout đều coi là không bảo chứng được
            logging.info(f"[-] rho = {rho_mid:.4f} KHÔNG AN TOÀN ({result.upper()}). Thu hẹp ROA.")
            rho_max = rho_mid
            
    logging.info(f"Bisection hoàn tất. ROA lớn nhất được chứng nhận: rho = {best_safe_rho:.4f}")
    return best_safe_rho