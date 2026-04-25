# Mismatch giữa Controller + Observer với MuJoCo

## Bối Cảnh (Context)

Hệ thống output-feedback có 3 thành phần song song:
1. **Training loop** ([apps/pendulum/output_feedback.py](../apps/pendulum/output_feedback.py))
   - Định nghĩa controller neural và observer Luenberger
   - Giả định frame trạng thái: **upright = 0** (góc chuẩn hóa)
   - Chu kỳ lấy mẫu: **dt = 0.01s** (rời rạc)
   - Mô hình động lực: tính toán từ Python (continuous approximation)

2. **Simulation MuJoCo** ([simulation/simulate_pendulum.py](../simulation/simulate_pendulum.py))
   - Đọc sensor từ MuJoCo và transform sang frame train
   - Chu kỳ điều khiển: **control_dt = 0.01s**
   - Chu kỳ physics MuJoCo: **timestep = 0.005s** ([assets/pendulum.xml](../assets/pendulum.xml))
   - Mô hình động lực: tích phân RK4 trong MuJoCo

3. **Định nghĩa dynamics** ([neural_lyapunov_training/pendulum.py](../neural_lyapunov_training/pendulum.py))
   - `PendulumDynamics`: mô hình học thuần Python
   - Observation function `h(x)`: trả về chỉ góc (nth=1)
   - Equilibrium: upright = **[0, 0]** (góc, vận tốc = 0)

---

## Các Nguồn Mismatch Được Xác Định

### 1. **Frame Trạng Thái & Chuẩn Hóa Quan Sát**

| Layer | Frame | Reference | Ghi chú |
|-------|-------|-----------|---------|
| Train | `theta_norm ∈ [-π, π]` | Upright = 0 | Sinh ra từ `PendulumDynamics.forward()` |
| MuJoCo XML | `qpos[0] ∈ ℝ` | Tùy bộ khởi tạo | ZQuốc Hinge axis = "0 1 0", không wrap |
| Simulation | `theta_norm` từ wrap | `theta_upright_raw = π` | Cộng chuẩn hóa tại line 112 |
| Controller input | `[theta_norm, theta_dot, y_obs]` | Batch size (batch, 3) | `NeuralNetworkController.in_dim = 3` |
| Observer state | `x_hat ∈ ℝ²` | Upright = 0 | Theo `NeuralNetworkLuenbergerObserver.z_dim = 2` |

**Rủi ro**:
- Nếu MuJoCo không có wrap-to-pi tự động, `qpos` có thể là 5.5 khi vật lắc sang trái, nhưng train kỳ vọng max -3.14 → -3.14.
- Observer không biết `qpos` có bao vòng → dự báo sai vị trí hình chiếu.

---

### 2. **Khớp Chu Kỳ Lấy Mẫu (Time-Step Mismatch)**

| Layer | dt | Ghi chú |
|-------|----|----|
| Training (discrete) | 0.01s | `SecondOrderDiscreteTimeSystem(pendulum, dt=0.01)` |
| Controller call | 0.01s | `if data.time - last_ctrl_time >= control_dt - 1e-6` |
| MuJoCo physics | 0.005s | `<option timestep="0.005" integrator="RK4"/>` |

**Rủi ro**: 
- Controller + observer được học dưới bản đồ rời rạc **x[n+1] = f(x[n], u[n])** với step 0.01.
- MuJoCo chạy 2 bước physics (2 × 0.005) mỗi khi control được gọi.
- Nếu MuJoCo và mô hình train khác (khác damping, khác inertia precision), sai lệch tích lũy.
- Observer sẽ thấy **state flow = (2 × 0.005)-step MuJoCo** ≠ **discrete map @ 0.01 từ training**.

**Fix ngắn hạn**:
- MuJoCo chạy từng bước 0.005 → cập nhật controller mỗi 2 physics steps.
- Hold torque giữa các cập nhật.

**Fix dài hạn**:
- Train lại controller/observer trên discrete transition của MuJoCo, không phải continuous approximation.

---

### 3. **Khớp Giới Hạn Actuator & Scaling Torque**

| Layer | Giới hạn | Ghi chú |
|-------|---------|---------|
| Train output | `[-u_max, u_max]` | `u_max = 0.25` ở `output_feedback.py`, line 93 |
| Controller clip | `clamp` | `NeuralNetworkController(..., clip_output="clamp", u_lo=[-0.25], u_up=[0.25])` |
| MuJoCo XML | `[-1.0, 1.0]` | `<motor ... ctrlrange="-1.0 1.0"/>` |

**Rủi ro**:
- Controller neural sinh ra torque trong [-0.25, 0.25].
- MuJoCo actuator nhận lệnh từ [-1.0, 1.0].
- Nếu không scaling, **MuJoCo sẽ chỉ apply ~25% công suất**.
- Hoặc nếu bạn scale lên [-1, 1], observer không được train dưới dynamics đó.

**Cách xử lý**:
- Buộc phải nhất quán: hoặc scale controller output, hoặc align ctrlrange trong XML.
- Nên tạo adapter layer để map controller output → MuJoCo input rõ ràng.

---

### 4. **Observer Luenberger: Mô hình mismatch**

File: [neural_lyapunov_training/controllers.py](../neural_lyapunov_training/controllers.py), line 169

```python
def forward(self, z, u, y):
    z_nominal = self.dynamics(z, u)  # z[n+1] ≈ f(z[n], u[n])
    obs_error = y - self.h(z)
    Le = self.fc_net(torch.cat((z, obs_error), 1))
    L0 = self.fc_net(torch.cat((z, (K * self.zero_obs_error).to(z.device)), 1))
    unclipped_z = z_nominal + Le - L0  # Correction = Le - L0
    return unclipped_z
```

**Rủi ro**:
- `self.dynamics(z, u)` gọi `SecondOrderDiscreteTimeSystem` của train → giả định **continuous integrator Euler**.
- MuJoCo chạy **RK4** → transition khác.
- Neural net `fc_net` học để bù lỗi, nhưng chỉ cho data train. Khi MuJoCo khác:
  - Nếu `z_nominal` sai khá, `Le` phải bù lớn → observer saturate hoặc oscillate.
  - Nếu `z_nominal` sai chút chút, `Le` học từ train không cover được → diverge dần.

**Không nên**:
- Cố gắng tăng `fc_hidden_dim` (e.g., [32, 32]) để trainer observer "fix" mismatch dynamics.
- Nó sẽ che lỗi, nhưng không giải quyết bản chất → controller sẽ vẫn fail khi chuyển sang domain khác.

---

## Phương Án Xử Lý (Plan)

### **Phương Án 1: Ngắn Hạn (Quick Fix) - Ít Rủi Ro**

**Mục tiêu**: Làm cho simulation chạy ổn định với controller/observer hiện tại.

**Các bước**:

1. **Chuẩn hóa frame trạng thái**
   - Thêm hàm helper `mujoco_state_to_train_frame(qpos, qvel)` → `(theta_norm, theta_dot)`
   - Dùng wrap-to-pi cho góc
   - Log state trước/sau transform để kiểm tra

2. **Khớp chu kỳ lấy mẫu**
   - Đổi MuJoCo XML: `<option timestep="0.01" integrator="RK4"/>`
   - Hoặc: giữ 0.005, nhưng cập nhật control mỗi 2 steps (0.01)
   - Đảm bảo controller/observer được gọi đúng tại thời điểm, không chậm

3. **Scaling torque**
   - Thêm scaling layer: `tau_to_ctrl = tau_nn / 0.25` (nếu bạn muốn dùng full [-1, 1])
   - Hoặc đổi XML `ctrlrange="-0.25 0.25"` để khớp controller output
   - Log và kiểm tra torque có bị clip không

4. **Logging & Diagnostic**
   - Theo dõi: sai số observer `|| x_hat - x_true ||`
   - Theo dõi: tỷ lệ torque bị clip
   - Theo dõi: one-step prediction residual `|| z[n+1] - dynamics(z[n], u[n]) ||`

**File cần sửa**:
- [simulation/simulate_pendulum.py](../simulation/simulate_pendulum.py)
- [assets/pendulum.xml](../assets/pendulum.xml)

---

### **Phương Án 2: Dài Hạn (Robust) - Nên Làm Để Ổn Định**

**Mục tiêu**: Controller + observer học đúng động lực MuJoCo chứ không phải continuous approximation.

**Các bước**:

1. **Sinh dataset từ MuJoCo**
   - Chạy MuJoCo với random action/state khởi tạo
   - Lưu transition: `{s[n], u[n]} → s[n+1]`
   - Cùng lưu linearization: `A[n], B[n]` quanh từng state

2. **Fit/Learn discrete model**
   - Nếu muốn LQR/EKF baseline: dùng finite difference hoặc explicit linearization trên MuJoCo.
   - Nếu muốn neural: train surrogate model trên MuJoCo data.

3. **Retrain controller + observer**
   - Thay `PendulumDynamics` bằng discrete model từ MuJoCo.
   - Chạy lại [apps/pendulum/output_feedback.py](../apps/pendulum/output_feedback.py) hoặc state_feedback.py.
   - Xác minh Lyapunov conditions trên MuJoCo dynamics.

4. **Validation**
   - Test simulation trên model mới.
   - So sánh ROA (Region of Attraction) giữa model xấp xỉ vs MuJoCo thực.

**Thời gian**: Tuần 1-2 nếu có sẵn MuJoCo environment.

---

### **Phương Án 3: Chẩn Đoán Lỗi (Debugging First)**

**Trước khi thực hiện Fix 1 hay Fix 2**, cần biết:**
- Mismatch gây ra bởi **frame**? → Kiểm tra log state / observer error.
- Mismatch gây ra bởi **time-step**? → Kiểm tra prediction residual.
- Mismatch gây ra bởi **torque saturation**? → Kiểm tra `data.ctrl[0]` có lúc = ±1 không.
- Mismatch gây bởi **observer diverge**? → Kiểm tra `|| x_hat - x_true ||` tăng theo thời gian.

**Script chẩn đoán**:
- Log 3 thứ sau mỗi update: `[t, x_true, x_hat, u_nn, residual]`
- Vẽ biểu đồ: error vs time, torque vs time, energy vs time.

---

## Các File Liên Quan

| File | Vai Trò | Line quan trọng |
|------|---------|-----------------|
| [simulation/simulate_pendulum.py](../simulation/simulate_pendulum.py) | Chạy mô phỏng với MuJoCo | Line 109 (control_dt), 112 (frame transform) |
| [assets/pendulum.xml](../assets/pendulum.xml) | Config MuJoCo | Line 2 (timestep, integrator), 16 (ctrlrange) |
| [apps/pendulum/output_feedback.py](../apps/pendulum/output_feedback.py) | Training controller + observer | Line 93 (u_max), 103 (in_dim=3), 125 (observer init) |
| [neural_lyapunov_training/pendulum.py](../neural_lyapunov_training/pendulum.py) | Định nghĩa dynamics | Line 4 (class), 48 (h function) |
| [neural_lyapunov_training/controllers.py](../neural_lyapunov_training/controllers.py) | Observer Luenberger | Line 169 (forward) |
| [neural_lyapunov_training/dynamical_system.py](../neural_lyapunov_training/dynamical_system.py) | Discrete time system | Xác định cách tích phân |

---

## Khuyến Nghị Ngay Lập Tức

1. **Kiểm tra frame** ✓ Đã làm ở [simulation/simulate_pendulum.py](../simulation/simulate_pendulum.py) line 112.
   - Nhưng cần log để xác nhận.

2. **Kiểm tra timestep** ✗ Chưa sửa.
   - Sửa ngay: đổi XML `timestep="0.005"` → `"0.01"` hoặc cập nhật control mỗi 2 steps.

3. **Kiểm tra torque scale** ✗ Chưa sửa.
   - Xác định rõ: controller output [-0.25, 0.25] → MuJoCo input [-1, 1] hay không?
   - Nếu có, thêm scaling.

4. **Thêm logging** ✗ Chưa có.
   - Lưu state/observer error/torque vào CSV để phân tích.

---

## Tóm Tắt Mismatch

```
┌─────────────────────────────────────────────────────────────┐
│ Training (Python Model)                                     │
│ ├─ Frame: θ_norm ∈ [-π, π], θ_dot                         │
│ ├─ dt: 0.01s (discrete)                                    │
│ ├─ Model: Continuous ODE → Euler approximation             │
│ └─ Controller: in_dim=3, clamp(u) ∈ [-0.25, 0.25]         │
└──────────────┬──────────────────────────────────────────────┘
               │
               │ Load weights at simulation
               ↓
┌─────────────────────────────────────────────────────────────┐
│ Simulation (MuJoCo)                                         │
│ ├─ Frame: qpos[hinge], qvel[hinge] → wrap to θ_norm        │
│ ├─ dt: 0.005s (physics), 0.01s (control) ⚠ MISMATCH      │
│ ├─ Model: RK4 integration                                   │
│ ├─ Actuator: ctrl ∈ [-1, 1] ⚠ SCALE MISMATCH            │
│ └─ Observer: expects x̂[n+1] from 0.01-step map            │
│    but receives 2× 0.005-step flow ⚠ DYNAMICS MISMATCH    │
└─────────────────────────────────────────────────────────────┘

Result: Observer error → Controller error → Unstable
```

---

## Tài Liệu Tham Khảo

- MuJoCo documentation: XML schema, integrators, actuators
- Luenberger observer theory: discrete-time systems
- Neural Lyapunov verification: mismatch robustness
- User memory: `debugging.md` (MuJoCo discrete ARE, LQR tuning)
