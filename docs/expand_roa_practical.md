# ROA Expansion - Thực Hành & Lệnh Sẵn Sàng

**Tài liệu này chứa các lệnh cụ thể, có thể copy-paste ngay để bắt đầu mở rộng ROA.**

---

## 🎯 Tóm Tắt Nhanh: 3 Bước Chính

```bash
# Bước 1: Chuẩn bị môi trường
cd /home/giapinner88/Project/Lyapunov-stable-Neural-Control
conda activate lypen

# Bước 2: Chạy huấn luyện với config mở rộng ROA
python apps/pendulum/state_feedback.py \
  model.lyapunov.hidden_widths=[32,32,16] \
  model.kappa=0.001 \
  train.max_iter=150 \
  train.pgd_steps=100

# Bước 3: Kiểm chứng ROA
python neural_lyapunov_training/bisect.py \
  --spec_prefix=specs/test \
  --lower_limit -12 -12 -12 -12 \
  --upper_limit 12 12 12 12 \
  --init_rho=0.001 \
  --config verification/pendulum_state_feedback_lyapunov_in_levelset.yaml
```

---

## 📋 Các Kịch Bản Huấn Luyện

### 1️⃣ Chiến Lược Cơ Bản: Tăng NN + Giảm Kappa

**Cách 1: Sửa file config (Bền vững)**

Tạo file `apps/pendulum/config/state_feedback_expanded_basic.yaml`:

```yaml
model:
  lyapunov:
    quadratic: false
    hidden_widths: [32, 32, 16]    # ↑ Tăng capacity
  kappa: 0.001                      # ↓ Giảm ràng buộc PSD
  V_decrease_within_roa: true
  limit_scale: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  rho_multiplier: [2.0, 1.8, 1.6, 1.4, 1.2]

train:
  max_iter: [50, 50, 40, 40, 30, 30, 30, 30, 20, 20]
  learning_rate: 0.001
  pgd_steps: 100
  epochs: 100
  batch_size: 1024

loss:
  candidate_roa_states_weight: [1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]
  l1_reg: [1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]
```

Chạy:
```bash
python apps/pendulum/state_feedback.py \
  --config apps/pendulum/config/state_feedback_expanded_basic.yaml
```

**Cách 2: Override CLI (Nhanh, Thử Nghiệm)**

```bash
python apps/pendulum/state_feedback.py \
  model.lyapunov.hidden_widths=[32,32,16] \
  model.kappa=0.001 \
  model.limit_scale=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] \
  model.rho_multiplier=[2.0,1.8,1.6,1.4,1.2] \
  train.max_iter=[50,50,40,40,30,30,30,30,20,20] \
  train.pgd_steps=100 \
  --config config.yaml
```

---

### 2️⃣ Chiến Lược Agressive: Mở Rộng Mạnh

**Dùng khi muốn ROA thật lớn (chấp nhận thời gian huấn luyện dài hơn)**

```bash
python apps/pendulum/state_feedback.py \
  model.lyapunov.hidden_widths=[64,32,16] \
  model.kappa=0.0001 \
  model.limit_scale=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0] \
  train.max_iter=[100,100,80,80,70,70,60,60,50,50,40,40,30,30,20,20,20,10,10,10] \
  train.pgd_steps=150 \
  train.learning_rate=0.0005 \
  --config config.yaml
```

**⚠️ Cảnh báo**: Yêu cầu GPU tốt, thời gian ~6-8 giờ

---

### 3️⃣ Chiến Lược Conservative: Kiểm Chứng Nhanh

**Dùng khi muốn kiểm chứng từng bước (an toàn, nhưng ROA nhỏ hơn)**

```bash
python apps/pendulum/state_feedback.py \
  model.lyapunov.hidden_widths=[24,24,12] \
  model.kappa=0.01 \
  model.limit_scale=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] \
  train.max_iter=[20,20,15,15,10,10,10,10,5,5] \
  train.pgd_steps=50 \
  train.learning_rate=0.001 \
  --config config.yaml
```

**Ưu điểm**: Chạy nhanh (~1-2 giờ), dễ debug

---

## 🔬 Các Lệnh Kiểm Chứng

### Chạy Bisection Để Tìm ρ Tối Đa

```bash
# ✓ Đạo cụ: Bisection để tìm ρ chính xác
cd /home/giapinner88/Project/Lyapunov-stable-Neural-Control

# Cơ bản
python neural_lyapunov_training/bisect.py \
  --spec_prefix=specs/test \
  --lower_limit -12 -12 -12 -12 \
  --upper_limit 12 12 12 12 \
  --init_rho=0.001 \
  --rho_eps=0.001 \
  --rho_multiplier=1.2 \
  --config verification/pendulum_state_feedback_lyapunov_in_levelset.yaml \
  --timeout=300

# Với mô hình cụ thể
python neural_lyapunov_training/bisect.py \
  --spec_prefix=specs/expanded_roa \
  --lower_limit -12 -12 -12 -12 \
  --upper_limit 12 12 12 12 \
  --init_rho=0.005 \
  --rho_eps=0.0001 \
  --rho_multiplier=1.15 \
  --config verification/pendulum_state_feedback_lyapunov_in_levelset.yaml \
  --timeout=600
```

**Giải thích:**
- `--init_rho=0.005`: Bắt đầu kiểm chứng từ rho = 0.005
- `--rho_eps=0.0001`: Dừng khi gap < 0.0001 (rất chính xác!)
- `--timeout=600`: 10 phút per verification (đủ cho ABCROWN)

**Kết quả sẽ được lưu:**
```
output/
├── rho_0.001_spec.txt    # Spec file cho mỗi rho
├── rho_0.001.txt         # Kết quả verification ("safe"/"unsafe")
├── ...
└── rho_l                  # ROA cuối cùng (đọc file này)
    rho_u
```

---

### Vẽ Biểu Đồ Pha

```bash
# Vẽ biểu đồ pha với ROA
python simulation/plot_phase_pendulum.py \
  --model_path output/expanded_roa_run/best_lyaloss.pth \
  --dpi 150 \
  --output_name roa_visualization.png

# Hoặc từ checkpoint nào đó
python simulation/plot_phase_pendulum.py \
  --model_path models/pendulum_state_feedback.pth

# Lưu vào output file cụ thể
python -c "
import torch
from simulation.plot_phase_pendulum import *

model = torch.load('output/expanded_roa_run/best_lyaloss.pth')
# Vẽ và lưu
"
```

---

## 📊 Theo Dõi Tiến Độ Huấn Luyện

### Xem Logs Realtime

```bash
# Terminal 1: Chạy huấn luyện
python apps/pendulum/state_feedback.py ...

# Terminal 2 (mở cái khác): Xem logs
tail -f output/*/logs.txt    # Xem logs mới nhất

# Hoặc grep logs để tìm từ khóa
grep "lyaloss\|rho\|loss" output/*/logs.txt | tail -50
```

### Bật W&B Dashboard (Khuyến Cáo)

```bash
# 1. Trong config hoặc CLI, bật wandb
python apps/pendulum/state_feedback.py \
  user.wandb_enabled=true \
  --config config.yaml

# 2. Truy cập W&B (nếu có tài khoản)
# → https://wandb.ai/your_entity/neural_lyapunov_training

# 3. Xem metrics:
#    - train/lyaloss (giảm = tốt)
#    - train/derivative_loss (minimize V decrease violations)
#    - Độ lớn của ρ theo giai đoạn
```

---

## ⚡ Workflow Tối Ưu: Toàn Bộ Quy Trình

### Scenario: Mở Rộng ROA Từ 0.00139 → 0.01+

```bash
#!/bin/bash
# Lưu dưới tên: expand_roa.sh

set -e  # Exit on error

cd /home/giapinner88/Project/Lyapunov-stable-Neural-Control
conda activate lypen

PROJECT_NAME="expanded_roa_v1"
OUTPUT_DIR="output/$PROJECT_NAME"
mkdir -p $OUTPUT_DIR

echo "═══════════════════════════════════════════"
echo "🚀 Starting ROA Expansion: $PROJECT_NAME"
echo "═══════════════════════════════════════════"

# Phase 1: Basic Expansion
echo ""
echo "📊 Phase 1: Basic Expansion (hidden_widths=[32,32,16], kappa=0.001)"
python apps/pendulum/state_feedback.py \
  model.lyapunov.hidden_widths=[32,32,16] \
  model.kappa=0.001 \
  model.limit_scale=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] \
  train.max_iter=[50,50,40,40,30,30,30,30,20,20] \
  train.pgd_steps=100 \
  --config config.yaml \
  2>&1 | tee $OUTPUT_DIR/phase1.log

# Get best model path
BEST_MODEL=$(ls -t output/*/best_lyaloss.pth 2>/dev/null | head -1)
echo "✓ Best model: $BEST_MODEL"

# Phase 2: Verification - Bisection
echo ""
echo "🔍 Phase 2: Verification Bisection"
python neural_lyapunov_training/bisect.py \
  --spec_prefix=specs/$PROJECT_NAME \
  --lower_limit -12 -12 -12 -12 \
  --upper_limit 12 12 12 12 \
  --init_rho=0.001 \
  --rho_eps=0.0001 \
  --rho_multiplier=1.15 \
  --config verification/pendulum_state_feedback_lyapunov_in_levelset.yaml \
  --timeout=600 \
  2>&1 | tee $OUTPUT_DIR/bisection.log

# Extract results
RHO_FINAL=$(cat output/$PROJECT_NAME/rho_l 2>/dev/null || echo "unknown")
echo ""
echo "═══════════════════════════════════════════"
echo "✅ FINAL RESULTS:"
echo "   ρ (verified) = $RHO_FINAL"
echo "═══════════════════════════════════════════"

# Phase 3: Visualization
echo ""
echo "📈 Phase 3: Creating ROA Visualization"
python simulation/plot_phase_pendulum.py \
  --model_path "$BEST_MODEL" \
  --dpi 150
  
echo ""
echo "✓ Done! Results in: $OUTPUT_DIR/"
```

Chạy:
```bash
chmod +x expand_roa.sh
./expand_roa.sh
```

---

## 🛠️ Các Lệnh Hữu Ích

### Xóa Checkpoint Cũ (Giải Phóng Không Gian)

```bash
# Xóa tất cả checkpoint
find output -name "*.pth" -type f -delete

# Hoặc chỉ giữ lại best
find output -name "*.pth" ! -name "best_*" -type f -delete
```

### Kiểm Tra Kích Thước GPU Memory

```bash
# Trước khi chạy
nvidia-smi

# Trong quá trình
watch -n 5 nvidia-smi  # Cập nhật mỗi 5 giây
```

### So Sánh Kết Quả Trước/Sau

```bash
# Bisection cũ
cat output/old_run/rho_l   # Kết quả cũ
cat output/old_run/rho_u

# Bisection mới
cat output/expanded_roa_v1/rho_l   # Kết quả mới
cat output/expanded_roa_v1/rho_u

# Tính cải thiện
# rho_new / rho_old = ?
```

---

## 🐛 Khắc Phục Sự Cố

### Problem 1: "CUDA Out of Memory"

```bash
# Giảm batch size
python apps/pendulum/state_feedback.py \
  train.batch_size=512 \      # Từ 1024 → 512
  train.epochs=150 \          # Bù thêm epochs
  --config config.yaml

# Hoặc giảm buffer
  train.buffer_size=65536 \
  train.Vmin_x_pgd_buffer_size=32768
```

### Problem 2: "Training Loss Not Decreasing"

```bash
# Giảm kappa (không quá)
python apps/pendulum/state_feedback.py \
  model.kappa=0.01 \         # Thay vì 0.001
  --config config.yaml

# Hoặc giảm network size
  model.lyapunov.hidden_widths=[24,24,12]
```

### Problem 3: "Bisection Returns Unknown"

```bash
# Tăng timeout
python neural_lyapunov_training/bisect.py \
  --timeout=900 \             # 15 phút (từ 300)
  --rho_multiplier=1.2 \      # Bước rho lớn hơn
  ...
```

---

## 📋 Checklist Mở Rộng ROA

```
CHUẨN BỊ:
 ☐ Cài đặt môi trường (conda activate lypen)
 ☐ Sao lưu config gốc
 ☐ Kiểm tra GPU memory đủ (nvidia-smi)

HUẤN LUYỆN:
 ☐ Chạy Phase 1 (basic expansion)
 ☐ Kiểm tra logs không có lỗi
 ☐ Xem loss giảm dần trong console
 ☐ Lưu best model path

KIỂM CHỨNG:
 ☐ Chạy bisection với --init_rho=0.001
 ☐ Chờ hoàn tất (có thể 2-3 giờ)
 ☐ Đọc kết quả: cat output/*/rho_l

ĐÁNH GIÁ:
 ☐ So sánh ρ_new vs ρ_old
 ☐ Vẽ phase portrait
 ☐ Kiểm tra trạng thái [0.01, 0, 0, 0] có trong ROA?

TIẾP THEO (nếu cần):
 ☐ Nếu ρ chưa đủ lớn → dùng Aggressive Strategy
 ☐ Nếu ổn định → lưu model + documentation
```

---

## 📚 Tài Liệu Tham Khảo Nhanh

```bash
# Xem README chính
cat README.md | grep -A 30 "Tuning Strategy"

# Xem config hiện tại
cat config.yaml

# Xem cácfile trong verification
ls -la verification/*.yaml

# Xem kết quả bisection chi tiết
ls -la output/*/rho_*.txt | head -20
```

---

## 🎓 Ghi Chú Quan Trọng

### Về Giai Đoạn Huấn Luyện

```yaml
# KHÔNG bắt đầu với limit_scale=1.0!
# Gradient sẽ rối loạn, huấn luyện thất bại

❌ WRONG:
limit_scale: [1.0]
max_iter: [100]

✓ RIGHT:
limit_scale: [0.1, 0.2, 0.3, 0.5, 1.0]
max_iter: [50, 50, 40, 30, 20]
```

### Về PGD Steps

```
pgd_steps càng cao → kiểm chứng chặt chẽ hơn
nhưng huấn luyện chậm hơn

- 50:  ✓ Nhanh, nhưng ROA nhỏ
- 100: ✓✓ Cân bằng tốt
- 150: ✓✓✓ Khá chặt, nhưng slow
- 200+: Không cần, lợi suất giảm dần
```

### Về Candidate ROA States

```julia
# Luôn bao gồm:
- Điểm cân bằng [0, 0, 0, 0]
- Các biên của không gian trạng thái

# Thêm logic các trạng thái khó:
- Vùng ngược (theta ≈ ±π)
- Năng lượng cao (cao vận tốc)
```

---

**Chúc bạn thành công mở rộng ROA!** 🚀

Nếu có vấn đề, hãy kiểm tra `EXPAND_ROA_GUIDE.md` (Hướng Dẫn Chi Tiết) hoặc `DEBUG_FINDINGS.md` (Ghi Chép Debug Trước Đó).
