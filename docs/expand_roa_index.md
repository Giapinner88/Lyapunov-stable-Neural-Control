# 📚 ROA Expansion - Thư Viện Tài Liệu Toàn Diện

**Hướng dẫn mở rộng Region of Attraction (ROA) cho bài toán điều khiển con lắc Lyapunov-stable-Neural-Control**

---

## 📖 4 Tài Liệu Chính

### 1. 🚀 **EXPAND_ROA_QUICK_START.md** (30 phút)
**Bắt Đầu Ngay - Không Cần Đọc Lý Thuyết**

- ✅ Chọn kịch bản phù hợp (Aggressive / Conservative / Basic)
- ✅ Lệnh copy-paste sẵn sàng để chạy
- ✅ Checklist trước khi bắt đầu
- ✅ FAQ & Troubleshooting cơ bản
- 📍 **Dùng khi**: Bạn muốn chạy ngay

**Ví dụ**:
```bash
python apps/pendulum/state_feedback.py \
  model.lyapunov.hidden_widths=[32,32,16] \
  model.kappa=0.001 \
  train.max_iter=150 --config config.yaml
```

---

### 2. ⚡ **EXPAND_ROA_PRACTICAL.md** (1-2 giờ thực hành)
**Workflow Toàn Bộ & Các Lệnh Thực Tế**

- ✅ 5 kịch bản huấn luyện cụ thể
- ✅ Script bash tự động (`expand_roa.sh`)
- ✅ Lệnh kiểm chứng Bisection chi tiết
- ✅ Theo dõi tiến độ (W&B, logs)
- ✅ 10+ lệnh hữu ích (xóa checkpoint, so sánh kết quả, vv)
- ✅ Khắc phục sự cố: GPU OOM, Loss không giảm, Verification unknown
- 📍 **Dùng khi**: Bạn muốn từng bước cụ thể

**Nội dung**:
- Scenario A: ROA lớn, không quan tâm thời gian → Aggressive
- Scenario B: Kiểm chứng nhanh → Conservative
- Scenario C: Mở rộng cân bằng → Basic
- Cấu hình giai đoạn `limit_scale`, `rho_multiplier`

---

### 3. 📚 **EXPAND_ROA_GUIDE.md** (1-2 giờ học)
**Hướng Dẫn Chi Tiết & Lý Thuyết**

14 phần to lớn:

| Phần | Nội Dung | Độc Lập? |
|------|---------|---------|
| 1 | ROA là gì + Vấn đề hiện tại | ✓ Độc lập |
| 2 | Các yếu tố ảnh hưởng ROA (table tham số) | ✓ |
| 3 | Chiến lược 1: Tăng công suất NN | ✓ |
| 4 | Chiến lược 2: Giảm kappa | ✓ |
| 5 | Chiến lược 3: Huấn luyện giai đoạn | ✓ |
| 6 | Chiến lược 4: Candidate ROA states | ✓ |
| 7 | Chiến lược 5: Tối ưu PGD | ✓ |
| 8 | Chiến lược 6: Kiểm chứng Bisection | ✓ |
| 9 | Quy trình từng bước (danh sách công việc) | ✓ |
| 10 | Cấu hình tối ưu (file YAML mẫu) | ✓ |
| 11 | Theo dõi tiến độ & Metrics | ✓ |
| 12 | Khắc phục sự cố (3 vấn đề chính) | ✓ |
| 13 | Đọc thêm & Tài liệu | ✓ |
| 14 | Ghi chú đặc biệt | ✓ |

**Điểm mạnh**:
- Giải thích TẠI SAO mỗi tham số ảnh hưởng ROA
- Cấu hình YAML mẫu (copy-paste được)
- Timeline dự kiến (6-12 giờ)
- Diagnosis & Troubleshooting sâu

📍 **Dùng khi**: Bạn muốn hiểu vấn đề sâu & tối ưu hóa

---

### 4. 🔬 **EXPAND_ROA_MATH.md** (30 phút - 1 giờ)
**Công Thức Toán Học & Giải Thuật**

13 phần:

1. Định nghĩa cơ bản (Lyapunov, ROA)
2. Điều kiện ổn định Lyapunov
3. Loss functions (Positivity, Derivative, PSD)
4. Tác động của từng tham số (with diagrams)
5. Pseudocode Training Loop
6. Verification Bisection Algorithm
7. Kết quả kiểm chứng (safe/unsafe/unknown)
8. Công thức tối ưu hóa (Lagrangian)
9. Mối quan hệ giữa các tham số
10. Thống kê dự kiến (bảng tốc độ vs ROA)
11. Golden Rules (5 quy luật vàng)
12. Diagnostic Metrics
13. Biểu đồ trade-offs

**Ví dụ**:
$$V(x) = V_{\text{network}} + |(εI + R^T R)(x-x^*)|_1$$

$$\mathcal{L}_{\text{deriv}} = \sum_{i=1}^{N} [\max(0, \frac{dV(x_i)}{dt} + \kappa)]^2$$

📍 **Dùng khi**: Bạn cần công thức & hiểu giải thuật

---

## 🎯 Chọn Tài Liệu Phù Hợp

### Quick Decision Tree

```
Bạn muốn gì?
│
├─ "Chạy ngay mà không hiểu quá sâu"
│  └─→ EXPAND_ROA_QUICK_START + EXPAND_ROA_PRACTICAL
│
├─ "Hiểu từng bước cụ thể"
│  └─→ EXPAND_ROA_PRACTICAL (workflow) + EXPAND_ROA_GUIDE (nếu cần)
│
├─ "Hiểu lý thuyết sâu & tối ưu hóa"
│  └─→ EXPAND_ROA_GUIDE (full) + EXPAND_ROA_MATH (tham khảo)
│
├─ "Chỉ muốn công thức & giải thuật"
│  └─→ EXPAND_ROA_MATH
│
└─ "Gặp sự cố / không rõ"
   └─→ Phần 12 trong EXPAND_ROA_GUIDE hoặc 🐛 trong EXPAND_ROA_PRACTICAL
```

---

## 📊 So Sánh Tài Liệu

| Aspect | Quick Start | Practical | Guide | Math |
|--------|-----------|-----------|-------|------|
| **Độ khó** | ⭐ Dễ | ⭐⭐ TB | ⭐⭐⭐ Khó | ⭐⭐⭐⭐ VK |
| **Thời gian đọc** | 30 phút | 1-2h | 1-2h | 30m-1h |
| **Lệnh copy-paste** | ✓ Có | ✓✓ Nhiều | Ít | ✗ Không |
| **Lý thuyết** | Cơ bản | Trung bình | ✓✓ Chi tiết | ✓✓✓ Sâu |
| **Workflow** | Outline | ✓ Full | Có | ✗ Không |
| **Khắc phục sự cố** | Cơ bản | ✓ Chi tiết | ✓ Rất chi tiết | ✗ Không |
| **Công thức toán** | ✗ Không | Ít | Một chút | ✓ Full |

---

## 🚀 Quy Trình Khuyến Nghị

### Lần Đầu (Bạn Hoàn Toàn Mới)

```
1️⃣ Đọc tóm tắt: EXPAND_ROA_QUICK_START.md (15 phút)
   ↓
2️⃣ Chọn kịch bản & copy lệnh: EXPAND_ROA_PRACTICAL.md (15 phút)
   ↓
3️⃣ Chạy huấn luyện (2-4 giờ tùy kịch bản)
   ↓
4️⃣ Chạy kiểm chứng bisection (1-3 giờ)
   ↓
5️⃣ Xem kết quả, nếu ROA chưa đủ → Fine-tune
   ↓
6️⃣ Nếu muốn hiểu sâu → Đọc EXPAND_ROA_GUIDE.md phần 2-5
```

**Tổng thời gian**: 3-7 giờ (bao gồm chạy)

### Bạn Có Kinh Nghiệm Lyapunov

```
1️⃣ Skip QUICK_START, đi thẳng EXPAND_ROA_GUIDE.md
   ↓
2️⃣ Chọn chiến lược từ phần 3-8
   ↓
3️⃣ Fine-tune cấu hình (phần 10)
   ↓
4️⃣ Chạy với script bash (EXPAND_ROA_PRACTICAL.md)
```

**Tổng thời gian**: 2-3 giờ (bao gồm chạy)

---

## 📋 Nội Dung Tất Cả Tài Liệu

### EXPAND_ROA_QUICK_START.md
- 📌 3 tài liệu chính
- 🎯 Bảng chọn kịch bản (Scenario A-E)
- ⚡ Bắt đầu nhanh (30 phút)
- 📋 Checklist trước khi bắt đầu
- 💡 3 sai lầm thường gặp
- 🔗 Liên kết nhanh

**Phù hợp cho**: Người muốn bắt đầu ngay

---

### EXPAND_ROA_PRACTICAL.md
- 🎯 3 lệnh cơ bản (khởi động 30 phút)
- 📋 4 kịch bản huấn luyện cụ thể
- 🔬 Lệnh kiểm chứng chi tiết
- 📊 Xem logs & W&B dashboard
- ⚡ Workflow toàn bộ (script bash)
- 🛠️ 15+ lệnh hữu ích
- 🐛 Khắc phục 3 vấn đề chính
- 📝 Ghi chú quan trọng

**Phù hợp cho**: Người muốn workflow cụ thể & lệnh sẵn sàng

---

### EXPAND_ROA_GUIDE.md
- 📖 1. ROA là gì & Vấn đề hiện tại
- 🔑 2. Các yếu tố ảnh hưởng (table 6x5)
- 📈 3-8. 6 Chiến lược mở rộng (mỗi chiến lược: tại sao + cách + cấu hình + cảnh báo)
- 🎯 9. Quy trình từng bước + Timeline
- 📝 10. Cấu hình YAML tối ưu (mẫu)
- 📊 11. Theo dõi tiến độ (metrics)
- 🐛 12. Khắc phục sự cố (3 vấn đề)
- 📚 13. Đọc thêm & Tài liệu
- 🎓 14. Ghi chú (State vs Output Feedback, Verification)

**Phù hợp cho**: Người muốn hiểu sâu & tối ưu hóa

---

### EXPAND_ROA_MATH.md
- 🔬 1. Định nghĩa Lyapunov & ROA
- ✓ 2. Điều kiện ổn định
- 📉 3. Loss functions (3 loại)
- ⚙️ 4. Tác động tham số (biểu đồ)
- 🔁 5. Training Loop (pseudocode)
- 🎯 6. Bisection Algorithm
- 🔍 7. Verification results
- ⚖️ 8. Công thức tối ưu
- 📊 9. Mối quan hệ tham số
- 📈 10. Thống kê dự kiến
- 🏆 11. Golden Rules (5 quy luật)
- 💊 12. Diagnostic Metrics
- 📐 13. Trade-off diagrams

**Phù hợp cho**: Người cần công thức & giải thuật

---

## 🎓 Các Khái Niệm Chính Được Giải Thích

### Trong Tất Cả Tài Liệu:

✅ **ROA (Region of Attraction)**
- Định nghĩa: Sublevel set $\mathcal{L}_\rho = \{x : V(x) \leq \rho\}$
- Tại sao nhỏ: Kappa quá mạnh, mạng quá hạn chế, không có staged training
- Cách mở rộng: 6 chiến lược cụ thể

✅ **Hàm Lyapunov (NN)**
- Cấu trúc: Network + PSD term
- Tham số: `hidden_widths`, `kappa`, `V_psd_form`

✅ **Loss Functions**
- Positivity: $V(x) \geq 0$
- Derivative: $\dot{V}(x) < 0$ trong ROA
- PSD: Ma trận $R^T R$ valid

✅ **Verification**
- Phương pháp: ABCROWN (alpha-beta-CROWN)
- Bisection: Tìm $\rho_{\max}$ được chứng minh

✅ **Training Strategy**
- Staged expansion: $\text{limit\_scale} = [0.1, 0.2, ..., 1.0]$
- PGD attack: Tìm vi phạm để huấn luyện chặt chẽ
- Candidate states: Bao trùm các trạng thái khó

---

## 📁 Đặt Vị Trí

Tất cả 4 tài liệu nằm trong thư mục gốc của dự án:

```
/home/giapinner88/Project/Lyapunov-stable-Neural-Control/
├── EXPAND_ROA_QUICK_START.md      ← Bắt đầu ở đây
├── EXPAND_ROA_PRACTICAL.md
├── EXPAND_ROA_GUIDE.md
├── EXPAND_ROA_MATH.md
│
├── README.md                      (Tài liệu chính dự án)
├── config.yaml                    (Cấu hình gốc)
├── apps/pendulum/
│   ├── state_feedback.py          (Huấn luyện)
│   └── config/                    (Config files)
└── neural_lyapunov_training/
    ├── bisect.py                  (Kiểm chứng)
    └── ...
```

---

## ⚡ Các Lệnh Mở Đầu

```bash
# Bước 1: Chuẩn bị
cd /home/giapinner88/Project/Lyapunov-stable-Neural-Control
conda activate lypen

# Bước 2: Chạy huấn luyện cơ bản
python apps/pendulum/state_feedback.py \
  model.lyapunov.hidden_widths=[32,32,16] \
  model.kappa=0.001 \
  train.max_iter=150 --config config.yaml

# Bước 3: Kiểm chứng
python neural_lyapunov_training/bisect.py \
  --spec_prefix=specs/test \
  --lower_limit -12 -12 -12 -12 \
  --upper_limit 12 12 12 12 \
  --init_rho=0.001 \
  --config verification/pendulum_state_feedback_lyapunov_in_levelset.yaml
```

---

## 🎯 Mục Tiêu Của Hướng Dẫn

```
TRƯỚC:  ρ = 0.00139 (quá nhỏ)
        [0.01, 0, 0, 0] NGOÀI ROA
        ❌ Không ổn định

SAU:    ρ ≥ 0.01 (được chứng minh)
        [0.01, 0, 0, 0] TRONG ROA
        ✓ Ổn định, ROA to

CẢI THIỆN: 10x → 50x ROA 🎉
```

**Thời gian dự kiến**: 6-12 giờ (bao gồm huấn luyện & kiểm chứng)

---

## 📞 Nếu Bạn Gặp Vấn Đề

| Vấn Đề | Xem Tài Liệu | Phần |
|--------|-----------|------|
| Không biết bắt đầu | QUICK_START.md | "Hành Động Ngay" |
| Không hiểu ROA | GUIDE.md | Phần 1 |
| Muốn cấu hình YAML | GUIDE.md | Phần 10 |
| Lỗi huấn luyện | PRACTICAL.md | "🐛 Khắc Phục" |
| Muốn công thức | MATH.md | Toàn bộ |
| Fine-tuning tham số | GUIDE.md | Phần 2-8 |

---

## 🏁 Bắt Đầu Ngay Bây Giờ

```bash
# Mở file Quick Start
cat EXPAND_ROA_QUICK_START.md

# Hoặc đẳng chạy lệnh ngay
python apps/pendulum/state_feedback.py \
  model.lyapunov.hidden_widths=[32,32,16] \
  model.kappa=0.001 \
  train.pgd_steps=100 --config config.yaml
```

---

**Chúc bạn thành công mở rộng ROA!** 🚀

---

**Created**: 2026-04-02  
**Language**: Tiếng Việt  
**Coverage**: Chi tiết, toàn diện, từ beginner → advanced
