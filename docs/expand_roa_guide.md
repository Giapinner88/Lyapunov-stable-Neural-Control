# Hướng Dẫn Mở Rộng Region of Attraction (ROA)

**Tác giả**: Lý thuyết Lyapunov Neural Control  
**Mục đích**: Hướng dẫn chi tiết cách mở rộng ROA để tăng miền ổn định của điều khiển  
**Ngữ cảnh**: Bài toán điều khiển con lắc với hàm Lyapunov mạng nơ-ron

---

## 1. ROA Là Gì Và Tại Sao Cần Mở Rộng?

### 1.1 Định Nghĩa ROA

**Region of Attraction (ROA)** là tập hợp lớn nhất các trạng thái ban đầu $x_0$ từ đó:
- **Hệ thống ổn định** theo lý thuyết được xác định bởi hàm Lyapunov $V(x)$
- **Điều khiển hoạt động tối ưu** trong các điều kiện được chứng minh chính thức

Về toán học:
$$\mathcal{L}_\rho = \{\xi : V(\xi) \leq \rho\}$$

Trong đó:
- $V(x)$ = hàm Lyapunov (mạng nơ-ron)
- $\rho$ = giá trị ngưỡng (sublevel set)
- **↑ $\rho$** = **↑ ROA** (miền ổn định lớn hơn)

### 1.2 Vấn Đề Hiện Tại

**Từ lịch sử debugging của bạn:**
```
❌ Problem: mean_theta = -π (điều khiển không ổn định)
❌ Root Cause: ROA quá nhỏ (rho = 0.00139)
❌ Initial State: [θ=0.01, 0, 0, 0] NẰM NGOÀI ROA
```

**Kết quả:**
- Trạng thái ban đầu bị **loại trừ** khỏi miền ổn định được chứng minh
- Điều khiển không hoạt động theo lý thuyết
- **Giải pháp**: Mở rộng ROA thông qua huấn luyện

---

## 2. Các Yếu Tố Ảnh Hưởng ROA

### 2.1 Tổng Quan Các Tham Số

| Tham Số | File Config | Tác Động | Hướng Thay | Mặc Định |
|---------|-------------|---------|-----------|---------|
| `model.kappa` | `state_feedback.yaml` | Cường độ ràng buộc PSD Lyapunov | ↓ kappa | `0.1` |
| `model.lyapunov.hidden_widths` | `state_feedback.yaml` | Công suất NN Lyapunov | ↑ width | `[16, 16, 8]` |
| `model.V_decrease_within_roa` | `state_feedback.yaml` | Ép buộc $dV/dt < 0$ trong ROA | `true` | `false` |
| `model.limit_scale` | `state_feedback.yaml` | Quy mô giới hạn trạng thái | ↑ scale | `[0.1, ..., 1.0]` |
| `model.rho_multiplier` | `state_feedback.yaml` | Hệ số mở rộng $\rho$ | ↓ multiplier | `2.0` với giai đoạn |
| `loss.candidate_roa_states_weight` | `state_feedback.yaml` | Trọng lượng để bao trùm các trạng thái roi | ↑ weight | `1e-5` |
| `train.max_iter` | `state_feedback.yaml` | Số lần lặp huấn luyện | ↑ iter | `100` |
| `train.learning_rate` | `state_feedback.yaml` | Tốc độ học | tuning | `1e-3` |

### 2.2 Thứ Tự Ảnh Hưởng (từ lớn nhất đến nhỏ nhất)

```
1st: model.lyapunov.hidden_widths  ← công suất mô hình Lyapunov
2nd: model.kappa                    ← độ mạnh của ràng buộc PSD  
3rd: loss.candidate_roa_states_weight ← v.v ổn định các trạng thái khó
4th: train.max_iter                 ← thời gian huấn luyện
5th: model.rho_multiplier           ← chiến lược mở rộng
6th: model.limit_scale              ← phạm vi không gian trạng thái
```

---

## 3. Chiến Lược 1: Tăng Công Suất Mô Hình Lyapunov

### 3.1 Tại Sao Hiệu Quả?

Hàm Lyapunov NN càng **phức tạp** → càng có thể **mô tả ROA lớn hơn** với độ chặt tương tự.

**Hiện Tại:**
```yaml
model.lyapunov.hidden_widths: [16, 16, 8]  # 3 lớp ẩn
```

**Phân tích:**
- Số tham số: ~$(4×16 + 16×16 + 16×8 + 8×1) ≈ 400$ tham số
- **Hạn chế**: Có thể không đủ để học ROA lớn

### 3.2 Bước Thực Hiện

```bash
# Bước 1: Tăng kích thước mạng Lyapunov
cd /home/giapinner88/Project/Lyapunov-stable-Neural-Control

# Tùy chỉnh qua CLI (không cần sửa YAML)
python apps/pendulum/state_feedback.py \
  model.lyapunov.hidden_widths=[32,32,16] \
  train.max_iter=150 \
  --config config.yaml

# Bước 2: Monitor độ lớn ROA trong quá trình huấn luyện
# Mở W&B dashboard hoặc xem output logs
```

### 3.3 Ví Dụ Cấu Hình (Config File)

**File**: `apps/pendulum/config/state_feedback.yaml`

```yaml
model:
  lyapunov:
    quadratic: false           # ★ Giữ NN (không dùng quadratic)
    hidden_widths: [32, 32, 16]  # ↑ Tăng từ [16,16,8]
  
  # Các tham số khác
  kappa: 0.05                 # Giảm từ 0.1
  V_decrease_within_roa: true
  limit_scale: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
```

### 3.4 Dấu Hiệu Cải Thiện

```
✓ $\rho$ (final) tăng từ 0.00139 → 0.01+
✓ Training loss xuống mượt mà
✓ Đơn vị kiểm chứng báo "safe" cho rho cao hơn
✓ Có thể khởi động từ trạng thái lớn hơn
```

---

## 4. Chiến Lược 2: Giảm Cường Độ Ràng Buộc PSD

### 4.1 Tại Sao Hiệu Quả?

Tham số `kappa` kiểm soát ép buộc $V(x) \geq |R^T R|$ (Positive Semi-Definite):

$$V(x) = V_{network}(x) + \underbrace{|(εI + R^T R)(x-x^*)|}_{\text{PSD term}}$$

**↑ kappa** → **ép buộc trở nên cứng nhắc** → **ROA bị ép buộc thu hẹp**

### 4.2 Bước Thực Hiện

```bash
# Giảm: từ 0.1 → 0.001
python apps/pendulum/state_feedback.py \
  model.kappa=0.001 \
  train.max_iter=200 \
  --config config.yaml
```

### 4.3 Ví Dụ Cấu Hình

```yaml
model:
  kappa: 0.001             # Giảm từ 0.1
  lyapunov:
    quadratic: false
    hidden_widths: [32, 32, 16]
```

### 4.4 Cảnh Báo ⚠️

```
⚠️ Nếu kappa quá nhỏ:
   - Mạng có thể không duy trì tính PSD!
   - Hàm Lyapunov không còn chứng minh ổn định
   - Kết quả kiểm chứng trở thành "unknown" hoặc "unsafe"

✓ Cân bằng: kappa = 0.001 ~ 0.01
```

---

## 5. Chiến Lược 3: Huấn Luyện Giai Đoạn Với Mở Rộng Tuyến Tính

### 5.1 Ý Tưởng

**Thay vì** huấn luyện một lần với `limit_scale=1.0` (riskier):

**Hãy** huấn luyện dần dần:
- **Giai đoạn 1**: `limit_scale=0.1` → ROA nhỏ nhưng ổn định
- **Giai đoạn 2**: `limit_scale=0.2` → Mở rộng khi có nền tảng
- ...
- **Giai đoạn N**: `limit_scale=1.0` → ROA lớn

### 5.2 Tại Sao Hiệu Quả?

- **Điều kiện khởi đầu tốt hơn** (gradient không bắt đầu từ rối loạn)
- **Hội tụ nhanh hơn** 
- **Kiểm chứng từng bước** cho độ tin cậy

### 5.3 Cấu Hình Giai Đoạn

**File**: `apps/pendulum/config/state_feedback.yaml`

```yaml
model:
  # Giai đoạn mở rộng
  limit_scale: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  
  # Hệ số rho: giảm dần (từ 2.0 → 1.2) trong mỗi giai đoạn
  rho_multiplier: [2.0, 1.8, 1.6, 1.4, 1.2]
  
  lyapunov:
    quadratic: false
    hidden_widths: [32, 32, 16]
  
  kappa: 0.003

train:
  # Số lần lặp cho mỗi giai đoạn
  max_iter: [50, 50, 40, 40, 30, 30, 30, 30, 20, 20]
  
  learning_rate: 0.001
  batch_size: 1024
  epochs: 100
  pgd_steps: 50
  
loss:
  # Bao trùm các trạng thái ở biên ROA
  candidate_roa_states_weight: [1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]
  l1_reg: [1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]
```

### 5.4 Chạy Huấn Luyện

```bash
python apps/pendulum/state_feedback.py \
  model.limit_scale=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] \
  model.rho_multiplier=[2.0,1.8,1.6,1.4,1.2] \
  train.max_iter=[50,50,40,40,30,30,30,30,20,20] \
  --config config.yaml
```

---

## 6. Chiến Lược 4: Sử Dụng Candidate ROA States

### 6.1 Ý Tưởng

**Vấn đề**: Huấn luyện ngẫu nhiên không khám phá được những trạng thái khó nhất ở **biên ROA**.

**Giải pháp**: Cung cấp **danh sách các trạng thái ứng viên** mà bạn muốn **chắc chắn nằm trong ROA**.

### 6.2 Ví Dụ Từ Codebase

**File**: `apps/pendulum/config/state_feedback_reproduce.yaml`

```yaml
loss:
  candidate_roa_states:
    - [-5.0, 9]      # theta ≈ -5 rad, dtheta = 9 rad/s
    - [-5.0, 8]
    - [-5.0, 7]
    - [-6.0, 7]
    - [-3, -2.0]    # Khác phía
    - [-3, -3.0]
    - [-3, -4.0]
    - [-3, -5.0]
    - [-3, -6.0]
    - [-4, -5.0]
    - [5.0, -9]     # Phía dương (đối xứng)
    - [5.0, -8]
    - [5.0, -7]
    - [6.0, -7]
    - [3.0, 6.0]
  
  candidate_roa_states_weight: 1.0e-05  # Trọng lượng mất mát
  candidate_scale: 1.0
  always_candidate_roa_regulizer: true
```

### 6.3 Cách Chọn Trạng Thái Ứng Viên

1. **Từ dữ liệu mô phỏng** quan sát được:
   ```bash
   # Chạy mô phỏng và ghi lại các trạng thái thú vị
   python simulation/simulate_pendulum.py
   ```

2. **Từ phân tích pha vật lý**:
   - Điểm cân bằng: $x^* = [0, 0, 0, 0]$ (luôn bao gồm)
   - Vùng biên: $\theta ≈ ±\pi/2$, $d\theta ≈ ±v_{max}$
   - Vùng khó: từ vị trí ngược (theta ≈ π)

3. **Từ kết quả kiểm chứng**:
   ```bash
   # Sau mỗi huấn luyện, chạy bisection để tìm ranh giới ROA
   python neural_lyapunov_training/bisect.py \
     --init_rho=0.01 \
     --config verification/path_to_config.yaml
   ```

---

## 7. Chiến Lược 5: Tối Ưu Hóa Các Tham Số PGD

### 7.1 Ý Tưởng

**PGD** (Projected Gradient Descent) là phương pháp tấn công:
- Tìm các trạng thái **vi phạm Lyapunov** (chứng minh ROA không đúng)
- Dùng để **cập nhật buffer** trong huấn luyện adversarial

**↑ pgd_steps** → **tìm vi phạm tốt hơn** → **huấn luyện chặt chẽ hơn**

### 7.2 Cấu Hình

```yaml
train:
  pgd_steps: 100         # Từ 50 → 100 cho ROA lớn
  buffer_size: 131072    # Lưu trữ lâu dài các vi phạm
  Vmin_x_pgd_buffer_size: 65536
  
loss:
  V_decrease_within_roa: true  # Ép buộc dV/dt < 0
  Vmin_x_boundary_weight: 0.1  # Ép buộc trên biên
```

### 7.3 Chi Phí Tính Toán

```
↑ pgd_steps = ↑ thời gian huấn luyện
  - 50 steps  ≈ +100% thời gian
  - 100 steps ≈ +200% thời gian
  - 200 steps ≈ +400% thời gian

⟹ Cân bằng: 50-100 là hợp lý cho ROA trung bình
```

---

## 8. Chiến Lược 6: Kiểm Chứng Và Mở Rộng Nhị Phân

### 8.1 Quy Trình Bisection

```
1. Đặt rho_l = 0, rho_u = "rất cao"
2. Kiểm chứng rho_m = (rho_l + rho_u) / 2
3. Nếu "safe": rho_l = rho_m (tìm cao hơn)
4. Nếu "unsafe": rho_u = rho_m (tìm thấp hơn)
5. Lặp lại cho đến khi rho_u - rho_l < epsilon
```

### 8.2 Chạy Bisection

**File**: `neural_lyapunov_training/bisect.py`

```bash
python neural_lyapunov_training/bisect.py \
  --spec_prefix=specs/pendulum_state_feedback \
  --lower_limit -12 -12 -12 -12 \
  --upper_limit 12 12 12 12 \
  --init_rho=0.001 \
  --rho_eps=0.001 \
  --rho_multiplier=1.2 \
  --config=verification/pendulum_state_feedback_lyapunov_in_levelset.yaml \
  --timeout=300
```

### 8.3 Kết Quả

```
Ví dụ dự kiến:
┌─────────────────────────────────┐
│ rho_l = 0.005                   │  ← ROA nhỏ nhất
│ rho_u = 0.008                   │  ← ROA lớn nhất (chứng minh)
│ rho_gap = 0.003 < epsilon ✓     │
└─────────────────────────────────┘
⟹ ROA được chứng minh: Lᵨ = {x : V(x) ≤ 0.0065}
```

---

## 9. Quy Trình Tối Ưu Hóa ROA: Từng Bước

### 9.1 Danh Sách Công Việc

```
1. ☐ Chuẩn bị: Sao lưu hiện tại, tạo checkpoint
2. ☐ Giai đoạn 1: Tăng công suất NN → hidden_widths=[32,32,16]
3. ☐ Giai đoạn 2: Giảm kappa → 0.001-0.003
4. ☐ Giai đoạn 3: Huấn luyện mở rộng với limit_scale giai đoạn
5. ☐ Giai đoạn 4: Thêm candidate_roa_states
6. ☐ Giai đoạn 5: Kiểm chứng bisection → xác định ρ lớn nhất
7. ☐ Fine-tune tham số dựa auf kết quả kiểm chứng
8. ☐ Làm lại từ bước 6 cho đến ROA hài lòng
```

### 9.2 Timeline

```
Giai đoạn 1-2 (Chuẩn bị):     ~0.5 giờ (sửa config)
Giai đoạn 3 (Huấn luyện):     ~2-4 giờ (tùy GPU)
Giai đoạn 4-5 (Kiểm chứng):   ~1-3 giờ (verifier)
Giai đoạn 6 (Fine-tune):      ~2-4 giờ (tùy số vòng)

TỔNG: ~6-12 giờ cho ROA lớn
```

---

## 10. Ví Dụ Cấu Hình Cho ROA Mở Rộng

### 10.1 Cấu Hình Tối Ưu (Khuyên Dùng)

**Tạo file**: `apps/pendulum/config/state_feedback_expanded_roa.yaml`

```yaml
seed: 42

user:
  run_dir: ./output
  wandb_enabled: true
  wandb_entity: your_wandb_entity

model:
  load_lyaloss: null  # Bắt đầu từ scratch
  save_lyaloss: true
  
  # ★ Chiến lược 1: Tăng công suất NN
  lyapunov:
    quadratic: false
    hidden_widths: [32, 32, 16]  # Tăng từ [16, 16, 8]
    R_rows: 2
    eps: 1.0e-04
    absolute_output: true
  
  # ★ Chiến lược 2: Giảm kappa
  kappa: 0.001              # Giảm từ 0.1
  V_decrease_within_roa: true
  V_psd_form: "L1"
  
  # ★ Chiến lược 3: Mở rộng giai đoạn
  limit_scale: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  rho_multiplier: [2.0, 1.8, 1.6, 1.4, 1.2]
  
  # Các thông số khác
  limit: [12.0, 12.0]
  velocity_integration: "ExplicitEuler"
  position_integration: "ExplicitEuler"
  dt: 0.05
  controller_nlayer: 4
  controller_hidden_dim: 8
  controller_path: null
  u_max: 6.0
  load_controller: null

train:
  train_lyaloss: true
  Vmin_x_pgd_path: null
  hard_max: true
  num_samples_per_boundary: 1024
  learning_rate: 1.0e-03
  lr_controller: 1.0e-03
  
  # ★ Chiến lược 5: Tối ưu PGD
  batch_size: 1024
  epochs: 100
  pgd_steps: 100          # Từ 50 → 100
  Vmin_x_pgd_buffer_size: 65536
  buffer_size: 131072
  samples_per_iter: 4096
  
  # ★ Chiến lược 3: Lặp lại cho từng giai đoạn
  max_iter: [50, 50, 40, 40, 30, 30, 30, 30, 20, 20]
  derivative_x_buffer_path: null
  lr_scheduler: false
  wandb:
    enabled: ${user.wandb_enabled}
    project: neural_lyapunov_training
    name: ${now:%Y.%m.%d-%H.%M.%S}_expanded_roa
    dir: ${user.run_dir}
    entity: ${user.wandb_entity}
  update_Vmin_boundary_per_epoch: false

loss:
  # ★ Chiến lược 4: Candidate ROA states
  candidate_scale: 0.8
  candidate_roa_states_weight: 
    [1.0e-03, 1.0e-03, 1.0e-04, 1.0e-04, 1.0e-04, 1.0e-04, 1.0e-05, 1.0e-05, 1.0e-05, 1.0e-05]
  candidate_roa_states:
    # Vùng năng lượng cao (điều khiển khó)
    - [-2.0, 10.0]
    - [-1.0, 10.0]
    - [ 0.0, 10.0]
    - [ 1.0, 10.0]
    - [ 2.0, 10.0]
    # Vùng ngược (theta ≈ ±π)
    - [-3.11, 2.0]
    - [-3.11, 1.0]
    - [-3.11, 0.0]
    - [-3.11, -1.0]
    - [-3.11, -2.0]
    - [3.11, 2.0]
    - [3.11, 1.0]
    - [3.11, 0.0]
    - [3.11, -1.0]
    - [3.11, -2.0]
  
  l1_reg: [1.0e-04, 1.0e-04, 1.0e-05, 1.0e-05, 1.0e-05, 1.0e-05, 1.0e-05, 1.0e-05, 1.0e-05, 1.0e-05]
  observer_ratio: [1.0e-04, 1.0e-04, 1.0e-05, 1.0e-05, 1.0e-05, 1.0e-05, 1.0e-05, 1.0e-05, 1.0e-05, 1.0e-05]
  always_candidate_roa_regulizer: true
  
  ibp_ratio_derivative: 0.1
  sample_ratio_derivative: 0.9
  ibp_ratio_positivity: 0.1
  sample_ratio_positivity: 0.9
  
  Vmin_x_boundary_weight: 0.01
  Vmax_x_boundary_weight: 0.01
```

### 10.2 Chạy Với Cấu Hình Mới

```bash
cd /home/giapinner88/Project/Lyapunov-stable-Neural-Control

# Chạy với cấu hình mở rộng ROA
python apps/pendulum/state_feedback.py \
  --config apps/pendulum/config/state_feedback_expanded_roa.yaml \
  user.wandb_enabled=true \
  user.run_dir=./output/expanded_roa_run

# Hoặc override trực tiếp (không cần file mới)
python apps/pendulum/state_feedback.py \
  model.lyapunov.hidden_widths=[32,32,16] \
  model.kappa=0.001 \
  train.pgd_steps=100 \
  model.limit_scale=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] \
  --config config.yaml
```

---

## 11. Theo Dõi Tiến Độ

### 11.1 Trong Quá Trình Huấn Luyện

```bash
# 1. Xem logs realtime
tail -f output/expanded_roa_run/logs.txt

# 2. W&B Dashboard (nếu bật wandb_enabled=true)
# → https://wandb.ai/your_entity/neural_lyapunov_training

# 3. Metrics chính cần theo dõi:
   - "train/lyaloss": Mất mát Lyapunov (↓ tốt)
   - "train/derivative_loss": Vi phạm dV/dt (↓ tốt)
   - "val/rho_verified": ROA xác minh (↑ tốt)
```

### 11.2 Sau Huấn Luyện: Kiểm Chứng

```bash
# 1. Vẽ biểu đồ pha ROA
python simulation/plot_phase_pendulum.py \
  --model_path output/expanded_roa_run/best_lyaloss.pth

# 2. Chạy bisection để xác định ρ chính xác
python neural_lyapunov_training/bisect.py \
  --spec_prefix=specs/expanded_roa \
  --lower_limit -12 -12 -12 -12 \
  --upper_limit 12 12 12 12 \
  --init_rho=0.001 \
  --rho_eps=0.001 \
  --config verification/pendulum_state_feedback_lyapunov_in_levelset.yaml

# 3. Xem kết quả
cat output/expanded_roa_run/rho_l
cat output/expanded_roa_run/rho_u
```

---

## 12. Khắc Phục Sự Cố

### 12.1 Vấn Đề: Mất Mát Huấn Luyện Không Giảm

**Nguyên Nhân**:
- `kappa` quá nhỏ → tính PSD không hợp lệ
- Mạng too large, overfitting
- Learning rate không phù hợp

**Giải Pháp**:
```yaml
model:
  kappa: 0.01         # Tăng từ 0.001
  lyapunov:
    hidden_widths: [24, 24, 12]  # Giảm từ [32,32,16]

train:
  learning_rate: 0.0001   # Giảm học
  lr_scheduler: true      # Bật scheduler
```

### 12.2 Vấn Đề: Kiểm Chứng Trả Về "Unknown"

**Nguyên Nhân**:
- Ràng buộc Lyapunov không vận động tốt
- ROA Candidates không chính xác

**Giải Pháp**:
```bash
# Tăng timeout + PGD aggressiveness
python neural_lyapunov_training/bisect.py \
  --timeout=600 \
  --config verification/config.yaml
```

### 12.3 Vấn Đề: ROA Quá Nhỏ So Với Kỳ Vọng

**Nguyên Nhân**:
- Dynamics quá khó (ex: dt quá lớn)
- Controller từ từ (small u_max)
- Mạng Lyapunov quá rigid

**Giải Pháp**:
```yaml
model:
  u_max: 8.0                           # Tăng lực điều khiển
  dt: 0.02                            # Giảm timestep
  lyapunov.hidden_widths: [64, 32, 16]  # Lớn hơn
  kappa: 0.0001                       # Rất nhỏ (cẩn thận!)
```

---

## 13. Đọc Thêm & Tài Liệu

### 13.1 Các Tài Liệu Chính

| Tệp | Nội Dung |
|-----|---------|
| [neural_lyapunov_training/lyapunov.py](neural_lyapunov_training/lyapunov.py) | Định nghĩa hàm Lyapunov NN |
| [neural_lyapunov_training/train_utils.py](neural_lyapunov_training/train_utils.py) | Hàm mất mát Lyapunov |
| [neural_lyapunov_training/bisect.py](neural_lyapunov_training/bisect.py) | Tìm kiếm nhị phân ρ tối đa |
| [README.md](README.md#tuning-strategy-for-larger-roa) | Tuning guide chính thức |
| [DEBUG_FINDINGS.md](debug/DEBUG_FINDINGS.md) | Ghi chép debug lịch sử |

### 13.2 Các Bài Viết Lý Thuyết

```
Lyapunov Stability: https://en.wikipedia.org/wiki/Lyapunov_stability
Neural Network Verification: https://arxiv.org/abs/2009.06669
Convex Relaxation Methods: https://arxiv.org/abs/1711.07356
```

---

## 14. Kết Luận

**ROA Expansion Workflow**:

```
┌─────────────────┐
│ Start: rho=0.001│
└────────┬────────┘
         │
    ★ Và Strategy 1: ↑ NN Capacity
    ★ Và Strategy 2: ↓ Kappa
    ★ Và Strategy 3: Staged Expansion
    ★ Và Strategy 4: Candidate States
    │
         ▼
┌──────────────────────┐
│ Train: max_iter=200+ │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ Verify: Bisection    │
│ rho_final = ?        │
└────────┬─────────────┘
         │
    ✓ Enough? DONE
    ✗ Too Small?  Loop back
         │
         ▼
    [Checkpoint + Done]
```

**Thời gian dự kiến**: **6-12 giờ** cho ROA lớn với cấu hình tối ưu.

**Thành công được định nghĩa là**: 
- ✓ Trạng thái ban đầu [0.01, 0, 0, 0] **nằm trong ROA xác minh**
- ✓ $\rho \geq 0.01$ (hoặc tùy theo yêu cầu)
- ✓ Mô phỏng từ các trạng thái trong ROA đạt được ổn định

---

## Phụ Lục: Ghi Chú Đặc Biệt

### A. Tại Sao ROA Hiện Tại Quá Nhỏ?

Từ `DEBUG_FINDINGS.md` của bạn:
```
Root Cause: rho = 0.00139 (quá nhỏ)
⟹ [theta=0.01, 0, 0, 0] nằm NGOÀI ROA
⟹ Điều khiển không hoạt động theo lý thuyết
```

**Nguyên nhân căn bản**:
1. Mạng Lyapunov quá nhỏ → không học được hình dạng phức tạp
2. Kappa = 0.1 quá mạnh → ép buộc PSD quá rigid
3. Không sử dụng staged expansion → mạng bị trapped trong local minimum

### B. Khác Biệt: State Feedback vs Output Feedback

```yaml
# State Feedback (đơn giản hơn)
model.controller: nhận đầy đủ state [theta, dtheta, ...]
model.observer: null
⟹ ROA thường lớn hơn

# Output Feedback (khó hơn)
model.observer: phải ước lượng state
model.controller: nhận output observer
⟹ ROA thường nhỏ hơn (nhưng gần thực tế hơn)
```

**Bắt đầu với State Feedback để mở rộng ROA, sau đó áp dụng Output Feedback**.

### C. Lưu Ý Về Verification

```
Verification tool: ABCROWN (complete_verifier)
- Kiểm chứng formal (tìm counter-example hoặc chứng minh safe)
- Timeout: thường 200-600 giây
- Kết quả: "safe" | "unsafe" | "unknown"

Training ≠ Verification
- Training: Tìm hàm Lyapunov tốt (dùng PGD sampling)
- Verification: Chứng minh formal rằng nó đúng (dùng SMT solver)
```

---

**Hết Hướng Dẫn**  
*Last Updated: 2026-04-02*
