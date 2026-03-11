import os
import subprocess

class ROABisector:
    def __init__(self, verification_wrapper, x_bounds, crown_config_path):
        self.wrapper = verification_wrapper
        self.x_bounds = x_bounds
        self.crown_config_path = crown_config_path
        
    def call_crown(self, rho):
        """
        Gọi tiến trình alpha-beta-CROWN bên ngoài.
        (Đây là mô phỏng luồng gọi thực tế của bài báo).
        """
        vnnlib_path = "temp_spec.vnnlib"
        # 1. Sinh file spec cho mức rho hiện tại
        from verification.crown_interface import generate_vnnlib_spec
        generate_vnnlib_spec(vnnlib_path, self.x_bounds, rho)
        
        # 2. Export mạng NN hiện tại sang định dạng ONNX
        import torch
        dummy_input = torch.zeros(1, len(self.x_bounds))
        self.wrapper.eval()
        torch.onnx.export(self.wrapper, dummy_input, "temp_model.onnx")
        
        # 3. Chạy CROWN (Giả mã luồng thực thi subprocess)
        print(f"[*] Đang chạy Verifier CROWN với rho = {rho:.4f}...")
        try:
            # Gọi CROWN qua command line
            result = subprocess.run(
                ["python", "verification/complete_verifier/complete_verifier/abcrown.py", 
                "--config", self.crown_config_path],
                capture_output=True, text=True, timeout=300 # Giới hạn 5 phút cho mỗi mức rho
            )
            
            # Phân tích log của CROWN
            if "result: unsat" in result.stdout:
                return True   # Chứng minh thành công
            elif "result: sat" in result.stdout:
                return False  # Có lỗi, chứng minh thất bại
            else:
                print("[-] CROWN không đưa ra được kết luận (Timeout hoặc Lỗi bộ giải).")
                return False
                
        except subprocess.TimeoutExpired:
            print("[-] CROWN Timeout.")
            return False 

    def find_max_roa(self, rho_min=0.01, rho_max=5.0, tolerance=0.05):
        """
        Tìm kiếm nhị phân (Binary Search) để xác định rho_max.
        """
        print("\n--- BẮT ĐẦU TÌM KIẾM NHỊ PHÂN (BISECTION) ---")
        best_certified_rho = 0.0
        
        low = rho_min
        high = rho_max
        
        while (high - low) > tolerance:
            mid = (low + high) / 2.0
            is_certified = self.call_crown(mid)
            
            if is_certified:
                print(f"[+] CHỨNG NHẬN THÀNH CÔNG tại rho = {mid:.4f}")
                best_certified_rho = mid
                low = mid  # Có thể an toàn hơn, thử đẩy rho lên
            else:
                print(f"[-] CHỨNG NHẬN THẤT BẠI tại rho = {mid:.4f} (Tìm thấy phản ví dụ hoặc Timeout)")
                high = mid # Quá rộng, phải thu hẹp lại
                
        print(f"\n=> KẾT QUẢ: ROA được chứng nhận lớn nhất rho_max = {best_certified_rho:.4f}")
        return best_certified_rho