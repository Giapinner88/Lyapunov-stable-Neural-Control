# ROA Expansion - Công Thức Toán Học & Tóm Tắt

## 1. Định Nghĩa Cơ Bản

### Hàm Lyapunov

$$V(x) = V_{\text{nominal}}(x) + V_{\text{network}}(x) + V_{\text{PSD}}(x)$$

Trong đó:
- **Nominal term**: $V_{\text{nominal}}(x) = 0$ (thường bỏ qua)
- **Network term**: $V_{\text{network}}(x) = \phi(x) - \phi(x^*)$ 
  - Với `absolute_output=True`: $|V_{\text{network}}(x)|$
- **PSD term**: $|(εI + R^T R)(x-x^*)|_1$ hoặc $(x-x^*)^T(εI + R^T R)(x-x^*)$

### Region of Attraction (ROA)

$$\mathcal{L}_\rho = \{x : V(x) \leq \rho\}$$

Ý nghĩa:
- $\rho$ càng lớn → ROA lớn
- **Mục tiêu**: Maximize $\rho$ sao cho hệ thống ổn định trong $\mathcal{L}_\rho$

---

## 2. Điều Kiện Ổn Định Lyapunov

### Điều Kiện 1: Positive Definiteness (V > 0)

$$V(x) > 0 \quad \forall x \neq x^*$$
$$V(x^*) = 0$$

**Thực hiện**: Thêm PSD term cứng nhắc:
$$V_{\text{PSD}}(x) = |\vec{r}|_1 = \sum_i \left|\sum_j R_{ij}(x_j - x_j^*)\right|$$

**Tham số điều chỉnh**: `model.V_psd_form` (L1 hoặc quadratic)

### Điều Kiện 2: Lyapunov Decrease ($-F(x) > 0$)

$$\dot{V}(x) = \frac{dV}{dt} = \nabla V^T \cdot \dot{x} < 0 \quad \forall x \in \mathcal{L}_\rho \setminus \{x^*\}$$

Hệ thống:
$$\dot{x} = f(x, u^*(x))$$

**Thực hiện**: Loss function để enforce:
$$\mathcal{L}_{\text{deriv}} = \sum_{x \in \mathcal{D}} \max(0, -\dot{V}(x) + \kappa)$$

**Tham số điều chỉnh**: 
- `V_decrease_within_roa`: bật/tắt điều kiện
- `kappa`: mạnh độ ràng buộc

---

## 3. Loss Functions Chính

### 3.1 Lyapunov Derivative Loss

```python
# Mục tiêu: -dV/dt > 0 (V giảm)
L_deriv = max(0, dV/dt + kappa)

# Khi có 2 ràng buộc:
# Ngoài: k_E * (E - E_target) * dtheta * cos(theta) - ...
# Trong ROA: -dV/dt > 0 (tự động ổn định)
```

$$\mathcal{L}_{\text{deriv}} = \sum_{i=1}^{N} \left[\max(0, \frac{dV(x_i)}{dt} + \kappa)\right]^2$$

**Tham số**:
- `kappa`: Margin ổn định (mặc định 0.1)
  - ↑ `kappa` → ép buộc lớn, ROA nhỏ
  - ↓ `kappa` → ép buộc yếu, nhưng cần chặt được

### 3.2 Positivity Loss

$$\mathcal{L}_{\text{pos}} = \sum_{i=1}^{N} [\min(V(x_i), 0)]^2$$

Đảm bảo $V(x) \geq 0$ luôn luôn.

### 3.3 PSD Regularization Loss

$$\mathcal{L}_{\text{PSD}} = -\text{det}(R^T R) + \lambda |R|^2$$

Đảm bảo ma trận $(εI + R^T R)$ là xác định dương.

---

## 4. Tác Động Của Các Tham Số

### 4.1 Kappa ($\kappa$)

```
V_decrease_within_roa = True  ⟹  -dV/dt ≥ κ (enforce trong ROA)

κ ↑ :  Ép buộc MẠNH → ROA thu hẹp
κ ↓ :  Ép buộc YẾU   → ROA lớn nhưng khó kiểm chứng
```

**Biểu đồ**:
```
ROA size vs Kappa
    ^
    │     ╱╲
ROA │    ╱  ╲
    │  ╱      ╲____  ← Tối ưu ở đây
    │╱
    └────────────────► kappa
      0.0001  0.01  1.0
```

### 4.2 Hidden Widths (Công Suất Mạng)

```yaml
hidden_widths: [16, 16, 8]      # Số tham số: ~500
hidden_widths: [32, 32, 16]     # Số tham số: ~2000
hidden_widths: [64, 32, 16]     # Số tham số: ~4000
```

**Tác động**:
$$\text{Param Count} \propto (\sum w_i) \cdot w_i$$

- ↑ Param → ↑ Expressive power → ↑ ROA (potential)
- Nhưng ↑ Risk of overfitting, ↑ Training time

### 4.3 Rho Multiplier ($\alpha$)

Trong verification (bisection):
$$\rho_{\text{next}} = \alpha \cdot \rho_{\text{current}}$$

- $\alpha = 2.0$ : Bước lớn (nhanh nhưng risky)
- $\alpha = 1.5$ : Bình thường
- $\alpha = 1.2$ : Bước nhỏ (chặt chẽ)

```
Bisection: ρ_l ... ρ_m ... ρ_u
                  ↑ ↓
                multiply by α
```

### 4.4 Limit Scale

```yaml
limit_scale: [0.1, 0.2, 0.3, ..., 1.0]

# Không gian trạng thái huấn luyện
state_limit = 12.0
training_limit[phase_i] = state_limit * limit_scale[phase_i]
```

**Mục đích**: Staged expansion (từng bước, an toàn)

---

## 5. Giải Thuật Huấn Luyện

### 5.1 Pseudocode Training Loop (1 Giai Đoạn)

```
Input: Initial controller/observer, Lyapunov_NN, System dynamics
Parameters: max_iter, learning_rate, lower_limit, upper_limit

for iteration = 1 to max_iter:
    # 1. Sample trạng thái
    x_samples ~ Uniform(lower_limit, upper_limit)
    
    # 2. Tính PGD attack (tìm vi phạm)
    x_adv = PGD_attack(x_samples, loss_function, steps=pgd_steps)
    
    # 3. Cập nhật buffer
    buffer ← cat(buffer, x_adv)
    buffer ← keep_worst(buffer, size=buffer_size)
    
    # 4. Sample từ buffer
    x_batch ~ Sample(buffer, batch_size)
    
    # 5. Forward pass
    V = Lyapunov_NN(x_batch)
    dV_dt = compute_derivative(V, dynamics, controller)
    
    # 6. Tính loss
    L_pos = max(0, -V)²
    L_deriv = max(0, dV_dt + κ)²
    L_total = L_pos + weights * L_deriv
    
    # 7. Backprop
    θ.grad ← ∇ L_total
    θ ← θ - learning_rate * θ.grad
    
    # 8. Log & Monitor
    if iteration % 10 == 0:
        print(f"Iter {iteration}: L_total={L_total:.4f}")

Output: Trained Lyapunov_NN
```

### 5.2 Training Across Phases

```
Loop over phases (i = 1 to num_phases):
    set limit_scale = limit_scales[i]
    set rho_multiplier = rho_multipliers[i]
    set max_iter = max_iters[i]
    
    Train(Lyapunov_NN, controller, observer)
    
    # Optional: Verify each phase
    if verify_each_phase:
        ρ_i ← Bisection(Lyapunov_NN, limits)
        Print(f"Phase {i}: ρ = {ρ_i}")
```

---

## 6. Verification: Bisection Algorithm

```
Input: Lyapunov function V(x), Initial ρ₀

# Phase 1: Tìm khoảng [ρ_l, ρ_u]
ρ ← ρ₀
if verify(ρ) == "safe":
    while verify(ρ) == "safe":
        ρ ← ρ * multiplier    # Tìm rho cao hơn
    ρ_l ← ρ / multiplier
    ρ_u ← ρ
else:
    while verify(ρ) ≠ "safe":
        ρ ← ρ / multiplier    # Tìm rho thấp hơn
    ρ_l ← ρ
    ρ_u ← ρ * multiplier

# Phase 2: Nhị phân trong [ρ_l, ρ_u]
while (ρ_u - ρ_l) > ε:
    ρ_m ← (ρ_l + ρ_u) / 2
    
    if verify(ρ_m) == "safe":
        ρ_l ← ρ_m              # Tìm rho cao hơn
    else:
        ρ_u ← ρ_m              # Tìm rho thấp hơn

Output: ρ_l (ROA được chứng minh)
```

**Độ phức tạp**: $O(\log_{\alpha}(\text{range}/\epsilon))$ iterations

---

## 7. Các Loại Verification & Kết Quả

### 7.1 Kết Quả Kiểm Chứng

```
Output              Ý Nghĩa
─────────────────────────────────────────
"safe"              ✓ Lyapunov conditions đúng
                    ✓ ROA verified

"unsafe"            ✗ Tìm thấy counter-example
                    ✗ Hàm Lyapunov sai

"unknown"           ? Verifier không quyết định được
                    ? Timeout hoặc quá phức tạp
```

### 7.2 ABCROWN Verifier

```
Thư viện: alpha-beta-CROWN (complete_verifier/)

Nhập:
  - Mạng NN (Lyapunov)
  - Input bounds (region state space)
  - Specification (Lyapunov conditions)

Xuất ra:
  - "certified safe" ↔ verified
  - "violated" ↔ unsafe (với counter-example)
  - "unknown" ↔ timeout
```

---

## 8. Công Thức Tối Ưu Hóa ROA

### 8.1 Mục Tiêu Tổng Quát

$$\max_\rho \{ \rho : \text{verify}(\text{Lyapunov}_v, \rho) = \text{"safe"} \}$$

Tương đương với:
$$\max_\rho \rho \quad \text{s.t.} \quad \begin{cases}
V(x^*) = 0 \\
V(x) > 0 & \forall x \neq x^* \\
\dot{V}(x) < 0 & \forall x \in \mathcal{L}_\rho \\
\text{verify}(\text{Lyapunov}_v, \rho) = \text{"safe"}
\end{cases}$$

### 8.2 Trong Huấn Luyện (Proxy Objective)

$$\min_{v, k, o} \; \mathcal{L} = w_1 \mathcal{L}_{\text{pos}} + w_2 \mathcal{L}_{\text{deriv}} + w_3 \mathcal{L}_{\text{PSD}} + w_4 \mathcal{L}_{\text{observer}}$$

**Tham số**:
- $v$: tham số Lyapunov NN
- $k$: tham số controller
- $o$: tham số observer
- $w_i$: trọng lượng loss

---

## 9. Mối Quan Hệ Giữa Các Tham Số

```
┌─ training stronger (more data, longer) ─┐
│                                          │
│ ↑ hidden_widths ──► more expressive     │
│ ↓ kappa ───────────► less constrained   │
│ ↑ pgd_steps ───────► tighter check      │
│ ↑ max_iter ────────► longer training    │
│                                          │
│ ═══════════════════════════════════════  │
│                    ↓↓↓                   │
│                 ↑ ROA                    │
└──────────────────────────────────────────┘

nhưng

↑ Computational Cost ➜ Trade-off cần cân bằng
```

---

## 10. Thống Kê Dự Kiến

### Dựa Trên Kinh Nghiệm Dự Án Hiện Tại

| Kịch bản | Speed | ROA Expected | Comments |
|---------|-------|-------------|----------|
| Conservative | 1-2h | 0.003-0.005 | Nhanh, verify từng bước |
| Basic | 2-4h | 0.005-0.015 | Cân bằng tốt |
| Aggressive | 6-8h | 0.01-0.05 | Lâu, ROA to |
| Very Aggressive | 8-12h | 0.02-0.1 | Very slow, risky |

---

## 11. Quy Luật Vàng (Golden Rules)

1. **Staged Expansion**: 
   $$\text{limit\_scale} \in [0.1, 0.2, \ldots, 1.0]$$
   KHÔNG phải là $[1.0]$

2. **Kappa Tuning**:
   $$\kappa \in [10^{-4}, 10^{-2}]$$
   KHÔNG bao giờ quá lớn (>0.1)

3. **Network Size**:
   $$\text{hidden\_widths} \in [16, 64] \times [16, 32] \times [8, 16]$$

4. **PGD Steps**:
   $$\text{pgd\_steps} \in [50, 150]$$
   Tăng từng từ, quan sát hiệu quả

5. **Buffer Management**:
   $$\text{buffer\_size} \geq 4 \times \text{batch\_size}$$

---

## 12. Diagnostic Metrics

### Trong Training, Monitor:

1. **Training Loss**
   ```
   ✓ Giảm mịn → OK
   ✗ Tăng/giao động mạnh → Learning rate quá cao
   ```

2. **Positivity Loss Component**
   ```
   Expected: Xấp xỉ 0
   ✗ Còn cao → V(x) có âm, tăng V_psd weight
   ```

3. **Derivative Loss Component**
   ```
   Expected: Giảm theo iteration
   ✗ Tăng → Controller/Lyapunov không hội tụ
   ```

### Sau Training, Verify:

1. **ROA Size**
   ```
   ρ_final / ρ_initial = ?
   Target: ↑ 10x (từ 0.001 → 0.01)
   ```

2. **Verification Status**
   ```
   ✓ "safe" → ROA verified
   ✗ "unknown" → Tăng timeout hoặc simplify model
   ```

---

## 13. Biểu Đồ Tóm Tắt

### Trade-offs

```
Model Capacity vs Verification Difficulty

Complexity
    ▲
    │      /████████╲
    │    /██████████  ╲
    │  /████████████    ╲___
    │ /██████████████████     ╲___
    │/████████████████████████     ╲____
    └─────────────────────────────────────► Time
             Easy          Hard        Too Hard
```

### ROA Expansion Curve

```
ROA Size vs Training Time

ρ
  │
  │       training continues
  │      /╲
  │    /╲  ╲  ___
  │  /    ╲  ╲╱   ╲___
  │/        ╲╱         ╲_____ → plateau
  └────────────────────────────► time
 0.001s        hours       days
```

---

**Tài liệu Toán Học Tóm Tắt** - Sử dụng kết hợp với `EXPAND_ROA_GUIDE.md` để hiểu sâu hơn.
