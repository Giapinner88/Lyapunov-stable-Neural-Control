# 🚀 ROA Expansion - Quick Start Guide

**Bạn muốn mở rộng ROA? Hãy bắt đầu ở đây!**

---

## 📖 3 Tài Liệu Chính

Hướng dẫn được chia thành 3 phần, tùy theo nhu cầu của bạn:

### 1. **EXPAND_ROA_PRACTICAL.md** ⚡ (Bắt Đầu Ngay)
- ✅ Lệnh copy-paste sẵn sàng
- ✅ Không cần đọc toàn bộ lý thuyết
- ✅ 3 kịch bản cụ thể (Basic, Aggressive, Conservative)
- ✅ Workflow toàn bộ từ A→Z
- 📝 **Dùng khi**: Bạn muốn chạy ngay mà không hiểu quá sâu

### 2. **EXPAND_ROA_GUIDE.md** 📚 (Chi Tiết & Lý Thuyết)
- ✅ Giải thích từng chiến lược mở rộng ROA
- ✅ Tại sao mỗi tham số ảnh hưởng ROA?
- ✅ Khắc phục sự cố chi tiết
- ✅ Hơn 14 phần, ~2000 dòng
- 📝 **Dùng khi**: Bạn muốn hiểu vấn đề sâu và tối ưu hóa

### 3. **EXPAND_ROA_MATH.md** 🔬 (Công Thức Toán Học)
- ✅ Công thức Lyapunov, ROA, loss functions
- ✅ Giải thuật Bisection
- ✅ Trade-offs giữa các tham số
- 📝 **Dùng khi**: Bạn cần hiểu toán học chi tiết

---

## ⚡ Bắt Đầu Nhanh (30 Phút)

### Nếu Bạn Vội

**Bước 1**: Copy lệnh này vào terminal

```bash
cd /home/giapinner88/Project/Lyapunov-stable-Neural-Control
conda activate lypen

python apps/pendulum/state_feedback.py \
  model.lyapunov.hidden_widths=[32,32,16] \
  model.kappa=0.001 \
  model.limit_scale=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] \
  train.max_iter=[50,50,40,40,30,30,30,30,20,20] \
  train.pgd_steps=100 \
  --config config.yaml
```

**Bước 2**: Chờ huấn luyện xong (~3-4 giờ)

**Bước 3**: Chạy kiểm chứng

```bash
python neural_lyapunov_training/bisect.py \
  --spec_prefix=specs/test \
  --lower_limit -12 -12 -12 -12 \
  --upper_limit 12 12 12 12 \
  --init_rho=0.001 \
  --config verification/pendulum_state_feedback_lyapunov_in_levelset.yaml \
  --timeout=300
```

**Bước 4**: Xem kết quả

```bash
cat output/*/rho_l  # ROA được chứng minh
```

---

## 📋 Chọn Kịch Bản Phù Hợp

### Scenario A: "Tôi muốn ROA lớn, không quan tâm thời gian"
→ Dùng **Aggressive Strategy**  
→ Xem: `EXPAND_ROA_PRACTICAL.md` → Phần "2️⃣ Chiến Lược Agressive"

### Scenario B: "Tôi muốn kiểm chứng nhanh, chấp nhận ROA nhỏ hơn"
→ Dùng **Conservative Strategy**  
→ Xem: `EXPAND_ROA_PRACTICAL.md` → Phần "3️⃣ Chiến Lược Conservative"

### Scenario C: "Tôi muốn hiểu cách hoạt động của mỗi tham số"
→ Dùng **Detailed Guide**  
→ Xem: `EXPAND_ROA_GUIDE.md` (toàn bộ)

### Scenario D: "Tôi muốn công thức toán học & giải thuật"
→ Dùng **Math Explanation**  
→ Xem: `EXPAND_ROA_MATH.md`

### Scenario E: "Tôi gặp lỗi/sự cố trong quá trình huấn luyện"
→ Tìm trong: `EXPAND_ROA_GUIDE.md` → Phần 12 (Khắc Phục Sự Cố)  
hoặc: `EXPAND_ROA_PRACTICAL.md` → Phần "🐛 Khắc Phục Sự Cố"

---

## 🎯 Mục Tiêu Của Bạn Là Gì?

**Trả lời 3 câu dưới đây để biết nên đọc gì:**

```
Q1: Tôi có hiểu về Lyapunov stability không?
    a) Không, lần đầu nghe → Đọc EXPAND_ROA_GUIDE.md (Phần 1-3)
    b) Có, quen rồi → Skip phần này

Q2: Tôi muốn kết quả hay hiểu vấn đề?
    a) Kết quả ngay → EXPAND_ROA_PRACTICAL.md
    b) Hiểu vấn đề → EXPAND_ROA_GUIDE.md

Q3: Tôi có bao nhiêu thời gian?
    a) 2 giờ → Conservative Strategy
    b) 4 giờ → Basic Strategy
    c) 8+ giờ → Aggressive Strategy
```

---

## 📊 Bảng So Sánh

| Tài Liệu | Độ Khó | Thời Gian Đọc | Khi Nào Dùng |
|----------|--------|----------------|-----------|
| **PRACTICAL** | ⭐ Dễ | 15 phút | Muốn chạy ngay |
| **GUIDE** | ⭐⭐⭐ Khó | 1-2 giờ | Muốn hiểu sâu |
| **MATH** | ⭐⭐⭐⭐ Rất khó | 30 phút - 1h | Muốn công thức |

---

## 🔑 Key Takeaways (Tóm Tắt 30 Giây)

**Vấn đề**: ROA của bạn quá nhỏ (rho = 0.00139)
- Trạng thái ban đầu [0.01, 0, 0, 0] nằm NGOÀI ROA
- Điều khiển không ổn định theo lý thuyết

**Giải pháp** (6 chiến lược chính):

1. **Tăng công suất mạng Lyapunov**
   ```
   hidden_widths: [16,16,8] → [32,32,16]
   ```

2. **Giảm ràng buộc PSD**
   ```
   kappa: 0.1 → 0.001
   ```

3. **Huấn luyện giai đoạn**
   ```
   limit_scale: [0.1, 0.2, ..., 1.0]
   ```

4. **Thêm candidate ROA states**
   ```
   Các trạng thái bạn MUỐN ở trong ROA
   ```

5. **Tăng PGD steps**
   ```
   pgd_steps: 50 → 100
   ```

6. **Kiểm chứng bisection**
   ```
   Tìm ρ_max được chứng minh
   ```

**Thời gian**: ~6-12 giờ cho ROA lớn

**Thành công**: ρ ≥ 0.01 + trạng thái khởi đầu nằm trong ROA

---

## ✅ Checklist Trước Khi Bắt Đầu

```
☐ Cài đặt environment: conda activate lypen
☐ GPU có đủ memory: nvidia-smi (cần ~6GB)
☐ Sao lưu config hiện tại
☐ Hiểu vấn đề: ROA quá nhỏ (đọc DEBUG_FINDINGS.md)
☐ Chọn kịch bản: A/B/C/D/E ở trên
☐ Chuẩn bị thời gian: 2-12 giờ tùy kịch bản
☐ Có network ổn định: Kiểm chứng từng bước
```

---

## 💡 Lưu Ý Quan Trọng

### ⚠️ Sai Lầm Thường Gặp #1
```
❌ WRONG: Huấn luyện trực tiếp với limit_scale=1.0
   → Gradient rối loạn, huấn luyện thất bại

✓ RIGHT: Staged expansion [0.1, 0.2, ..., 1.0]
   → Gradient ổn định, ROA mở rộng dần
```

### ⚠️ Sai Lầm Thường Gặp #2
```
❌ WRONG: Giảm kappa quá thấp (< 0.0001)
   → Hàm Lyapunov không còn là hàm Lyapunov!
   → Verification trả về "unknown" hoặc "unsafe"

✓ RIGHT: Kappa ∈ [0.001, 0.01]
   → Cân bằng giữa ROA lớn & PSD valid
```

### ⚠️ Sai Lầm Thường Gặp #3
```
❌ WRONG: Chạy 1 lần, kiểm chứng xong là hoàn tất
   → ROA có thể không tối ưu (phụ thuộc seed)

✓ RIGHT: Chạy multiple seeds hoặc fine-tune tham số
   → Chạy lại bisection nếu ρ chưa đủ lớn
```

---

## 🎓 Cấu Trúc Thư Mục Hướng Dẫn

```
Lyapunov-stable-Neural-Control/
│
├── EXPAND_ROA_QUICK_START.md      ← Bạn đang đọc FILE NÀY
│
├── EXPAND_ROA_PRACTICAL.md        ← Lệnh & workflow sẵn sàng
│
├── EXPAND_ROA_GUIDE.md            ← Hướng dẫn chi tiết (14 phần)
│
├── EXPAND_ROA_MATH.md             ← Công thức & giải thích toán
│
├── README.md                       ← Tào lao chính của dự án
│
├── neural_lyapunov_training/
│   ├── bisect.py                  ← Kiểm chứng ROA
│   ├── train_utils.py             ← Hàm loss
│   └── lyapunov.py                ← Định nghĩa hàm Lyapunov
│
└── apps/pendulum/
    ├── state_feedback.py          ← Chạy huấn luyện
    └── config/                    ← Config files
        ├── state_feedback.yaml
        ├── state_feedback_expanded_roa.yaml  ← Mới tạo
        └── ...
```

---

## 🚀 Hành Động Ngay Bây GIỜ

### Lựa Chọn 1: Nhanh (30 phút xem lệnh)
1. Mở `EXPAND_ROA_PRACTICAL.md`
2. Copy lệnh từ "Bước 1-2"
3. Chạy trong terminal

### Lựa Chọn 2: Chuẩn (1-2 giờ đọc)
1. Mở `EXPAND_ROA_GUIDE.md`
2. Đọc Phần 1-5 (Khái niệm + Chiến lược)
3. Chạy lệnh tương ứng

### Lựa Chọn 3: Sâu (2-3 giờ học)
1. Mở `EXPAND_ROA_GUIDE.md` (toàn bộ)
2. Mở `EXPAND_ROA_MATH.md` (tham khảo công thức)
3. Chạy + Sửa đổi tham số dựa hiểu biết

---

## 📞 Cần Giúp?

**Nếu gặp lỗi hoặc không rõ:**

| Vấn đề | Giải Pháp | Trong File Nào |
|--------|-----------|-------------|
| "Không biết bắt đầu từ đâu" | Xem Practical #1️⃣ | EXPAND_ROA_PRACTICAL.md |
| "GPU out of memory" | Giảm batch_size | EXPAND_ROA_PRACTICAL.md → 🐛 |
| "Loss không giảm" | Kiểm tra kappa/network | EXPAND_ROA_GUIDE.md → Phần 12 |
| "Verification unknown" | Tăng timeout | EXPAND_ROA_PRACTICAL.md → 🐛 |
| "Muốn hiểu tại sao" | Đọc phần lý thuyết | EXPAND_ROA_GUIDE.md → Phần 2-5 |
| "Muốn công thức" | Xem MATH file | EXPAND_ROA_MATH.md |

---

## 📈 Dự Kiến Kết Quả

**Trước** (Hiện Tại):
```
ρ = 0.00139 (quá nhỏ)
Initial state [0.01, 0, 0, 0] NGOÀI ROA
Điều khiển không ổn định theo lý thuyết
```

**Sau** (Với hướng dẫn này):
```
ρ = 0.01 ~ 0.05 (được chứng minh)
Initial state [0.01, 0, 0, 0] TRONG ROA ✓
Điều khiển ổn định, có ROA to
```

**Improvement**: **10x → 50x lơn ROA** 🎉

---

## 🔗 Liên Kết Nhanh

### Các File Quan Trọng Khác
- `README.md` - Tài liệu chính của dự án
- `debug/DEBUG_FINDINGS.md` - Ghi chép debug (lý do ROA nhỏ)
- `neural_lyapunov_training/bisect.py` - Mã kiểm chứng
- `apps/pendulum/state_feedback.py` - Mã huấn luyện

### Verification Config
```
verification/
├── pendulum_state_feedback_lyapunov_in_levelset.yaml  ← Dùng cái này
└── ...
```

---

**Bắt Đầu Ngay!** ⚡

Chọn một trong 3 tài liệu ở trên và bắt tay vào việc mở rộng ROA!

```bash
# 👇 Copy lệnh này để bắt đầu
cd /home/giapinner88/Project/Lyapunov-stable-Neural-Control && \
conda activate lypen && \
echo "✓ Environment ready! Now read EXPAND_ROA_PRACTICAL.md"
```

Chúc bạn thành công! 🚀
