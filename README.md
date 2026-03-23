# Lyapunov-stable Neural Control: Pendulum & Cart-Pole

## 🎯 Mục tiêu Dự án
Dự án này tái hiện và mở rộng kiến trúc từ bài báo **"Lyapunov-stable Neural Control" (Yang et al., 2024)**. 

Thay vì chỉ kiểm chứng trên hệ con lắc ngược (Inverted Pendulum), dự án được cấu trúc lại bằng một bộ khung Hướng đối tượng (OOP) tổng quát cho vòng lặp CEGIS (Counter-Example Guided Inductive Synthesis). Mục đích là để có thể dễ dàng mở rộng (scale-up) thuật toán sang các hệ thống động lực học thiếu dẫn động (underactuated) bậc cao hơn, tiêu biểu là hệ **Cart-Pole** (4D).

**Các mục tiêu cốt lõi:**
1. **Generic CEGIS Pipeline:** Tách biệt hoàn toàn phần thuật toán huấn luyện (Neural Controller & Neural Lyapunov) khỏi phần động lực học vật lý (Dynamics).
2. **Toán học hóa Hàm Lyapunov:** Đảm bảo kiến trúc mạng tuân thủ chặt chẽ công thức toán học nguyên thủy $V(x) = |\phi_V(x) - \phi_V(0)| + \|(\epsilon I + R^TR)x\|_1$ để đảm bảo $V(0) = 0$ tuyệt đối.
3. **Formal Verification:** Tích hợp bộ giải $\alpha,\beta$-CROWN để chứng minh ranh giới an toàn (Region of Attraction - ROA) bằng toán học nghiêm ngặt, khắc phục các điểm mù (counter-examples) mà phương pháp lấy mẫu (PGD) bỏ sót.

## ⚙️ Hướng dẫn Cài đặt (Conda Environment)

Để kiểm soát chặt chẽ các phụ thuộc tuyến tính của công cụ xác minh hình thức (đặc biệt là `auto_LiRPA` và bộ giải CROWN), dự án này bắt buộc phải chạy trong môi trường ảo được cô lập.

**Bước 1: Khởi tạo môi trường ảo với Conda**
Bản chất của Conda không chỉ quản lý các gói Python mà còn quản lý các thư viện C/C++ ngầm định (ví dụ như các driver CUDA hay thư viện toán học cấp thấp) giúp PyTorch tính toán đồ thị tính (computation graph) ổn định hơn.
```bash
conda create -n lyapunov_env python=3.10 -y
conda activate lyapunov_env
```

**Bước 2: Cài đặt các thư viện lõi**
Cài đặt trực tiếp từ file requirements:
```bash
pip install -r requirements.txt
```

**Bước 3: Tích hợp `complete_verifier` (alpha, beta-CROWN)**
Thuật toán Bisection cần gọi module `complete_verifier.abcrown` để xác minh. Bạn cần clone mã nguồn của công cụ này và đưa vào biến môi trường:
```bash
# Tạo thư mục chứa bộ xác minh
mkdir -p verification
cd verification

# Clone mã nguồn alpha, beta-CROWN
git clone [https://github.com/Verified-Intelligence/alpha-beta-CROWN.git](https://github.com/Verified-Intelligence/alpha-beta-CROWN.git) complete_verifier

# Cập nhật PYTHONPATH để Python có thể nhận diện được module này
export PYTHONPATH="$PWD/complete_verifier:$PYTHONPATH"
cd ..
```
*(Lưu ý: Nếu không có bước 3, hệ thống sẽ báo lỗi `ModuleNotFoundError: No module named 'complete_verifier'` khi chạy bisection)*.

---

### 2. File `requirements.txt`

Bản chất của việc huấn luyện hệ thống điều khiển ổn định Lyapunov đòi hỏi các phép tính đạo hàm tự động (Auto-Grad) liên tục và kiểm tra ranh giới phi tuyến. Dưới đây là các thư viện tối giản nhất để hệ thống hoạt động:

```text
# Deep Learning Core
torch>=2.0.0

# Tính toán ma trận và trực quan hóa (vẽ phase portrait, contour 2D/4D)
numpy
matplotlib

# Giải tích ký hiệu (Dùng để tính Jacobian tuyến tính hóa cho LQR baseline)
sympy

# Phân tích cú pháp cấu hình cho bộ giải CROWN
pyyaml

# Core Formal Verification: Phục vụ Linear Relaxation cho các hàm phi tuyến (sin, cos)
auto_LiRPA
```