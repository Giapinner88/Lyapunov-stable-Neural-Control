# Hướng Dẫn Chi Tiết: Lyapunov-stable Neural Control cho Cartpole
**Dựa trên bài báo:** *Lyapunov-stable Neural Control for State and Output Feedback: A Novel Formulation* (Yang et al., ICML 2024)

---

## Mục Lục

1. [Tổng quan lý thuyết](#1-tổng-quan-lý-thuyết)
2. [Bài toán Cartpole](#2-bài-toán-cartpole)
3. [Kiến trúc mô hình](#3-kiến-trúc-mô-hình)
4. [Formulation mới: ROA lớn hơn](#4-formulation-mới-roa-lớn-hơn)
5. [Thuật toán Training (CEGIS + PGD)](#5-thuật-toán-training-cegis--pgd)
6. [Pipeline đầy đủ từ training đến verification](#6-pipeline-đầy-đủ-từ-training-đến-verification)
7. [Cấu trúc file code hiện có](#7-cấu-trúc-file-code-hiện-có)
8. [Hướng dẫn training thực tế](#8-hướng-dẫn-training-thực-tế)
9. [Verification với α,β-CROWN](#9-verification-với-αβ-crown)
10. [Debugging và tips thực tiễn](#10-debugging-và-tips-thực-tiễn)

---

## 1. Tổng quan lý thuyết

### 1.1 Lyapunov Stability là gì?

Cho hệ động lực discrete-time:

```
x_{t+1} = f(x_t, u_t)
```

Một hàm **Lyapunov** `V(x)` là hàm scalar chứng minh tính ổn định của hệ nếu thỏa:

| Điều kiện | Ý nghĩa |
|-----------|---------|
| `V(x*) = 0` | Bằng 0 tại điểm cân bằng |
| `V(x) > 0` với mọi `x ≠ x*` | Dương definite |
| `V(x_{t+1}) - V(x_t) ≤ -κ·V(x_t)` | Giảm theo thời gian (exponential) |

Nếu tồn tại `V` thỏa 3 điều kiện trên trong một tập `S`, thì `S` là **Region-of-Attraction (ROA)** — mọi trajectory xuất phát trong `S` sẽ hội tụ về `x*`.

### 1.2 Vấn đề với NN controllers

Neural network controller có thể hoạt động tốt empirically, nhưng **không có formal guarantee** về stability. Bài báo này giải quyết: làm sao jointly train một NN controller + Lyapunov function và sau đó **verify chính xác** với công cụ α,β-CROWN.

### 1.3 Đóng góp chính của bài báo

**Formulation cũ (DITL, Chang et al.):**
```
Enforce: V(x_{t+1}) - V(x_t) ≤ -κV(x_t)  ∀x_t ∈ B
```
→ Quá restrictive: phải thỏa trên toàn bộ box B, kể cả vùng ngoài ROA thực sự.

**Formulation mới (bài báo này):**
```
Enforce: (-F(x_t) ≥ 0 ∧ x_{t+1} ∈ B) ∨ (V(x_t) ≥ ρ)  ∀x_t ∈ B
```
→ Chỉ cần thỏa **trong ROA** (sublevel set `V < ρ`). ROA lớn hơn, dễ verify hơn.

---

## 2. Bài toán Cartpole

### 2.1 State space

State vector: `x = [cart_pos, cart_vel, pole_angle, pole_vel]` — 4 chiều.

### 2.2 Dynamics (Euler integration, τ = 0.05s)

```python
# Các thông số vật lý
gravity     = 9.8    # m/s²
masscart    = 1.0    # kg
masspole    = 0.1    # kg
total_mass  = 1.1    # kg
length      = 1.0    # m (half-pole length)
tau         = 0.05   # timestep
max_force   = 30.0   # N

# Equations of motion
temp       = masscart + masspole * sin(θ)²
thetaacc   = (-F·cos(θ) - masspole·L·θ̇²·cos(θ)·sin(θ) + total_mass·g·sin(θ)) / (L·temp)
xacc       = (F + masspole·sin(θ)·(L·θ̇² - g·cos(θ))) / temp

# Next state (Euler)
x_next      = x + τ·ẋ
ẋ_next      = ẋ + τ·xacc
θ_next      = θ + τ·θ̇
θ̇_next      = θ̇ + τ·thetaacc
```

### 2.3 Mục tiêu

Stabilize cartpole về trạng thái cân bằng: `x* = [0, 0, 0, 0]` (cột đứng thẳng).
- Region-of-interest (box): `B = [-1, 1]⁴`
- Force constraint: `-30 ≤ F ≤ 30` N

---

## 3. Kiến trúc mô hình

### 3.1 PolicyNet (Controller)

```python
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = nn.Linear(4, 1, bias=False)  # Linear controller

    def forward(self, x):
        return self.policy(x)
```

- **Đơn giản**: Linear policy `u = W·x` — 4 parameters.
- Output được clamp vào `[-max_force, max_force]` bằng ReLU:
  ```python
  force = (-max_force) + F.relu(force - (-max_force))
  force = max_force - F.relu(max_force - force)
  ```
- Bởi vì policy là linear, có thể dùng LQR để khởi tạo.

### 3.2 LyapunovNet

```python
class LyapunovNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lyapunov = nn.Sequential(
            nn.Linear(4, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 1, bias=False),
        )
```

- **Architecture**: `4 → 32 → 32 → 1`, no bias, ReLU activation.
- **Tổng số params**: `4×32 + 32×32 + 32×1 = 128 + 1024 + 32 = 1184`.
- Output raw `φ_V(x)` — cần xử lý thêm để đảm bảo positive definite.

> **Lưu ý về positive definiteness:** Bài báo dùng công thức:
> ```
> V(x) = |φ_V(x) - φ_V(x*)| + ‖(εI + RᵀR)(x - x*)‖₁
> ```
> Trong implementation `cartpole.py` đơn giản hơn — raw output của LyapunovNet được dùng trực tiếp và tính `lyap_next - lyap` để kiểm tra điều kiện giảm.

### 3.3 Cartpole wrapper (combined model)

```python
class Cartpole(nn.Module):
    def forward(self, state):
        # 1. Compute action from policy
        action = self.policy_model(state)
        # 2. Clamp to force limits
        force = clamp(action, -max_force, max_force)
        # 3. Integrate dynamics
        state_next = euler_step(state, force)
        # 4. Compute Lyapunov values
        lyap      = self.lyap_model(state)        # V(x_t)
        lyap_next = self.lyap_model(state_next)   # V(x_{t+1})
        # 5. Outputs
        y_0 = lyap                    # V(x_t) — must be positive
        y_1 = lyap_next - lyap        # ΔV — must be negative (inside ROA)
        out_of_hole = relu(|state| - 0.1).sum()  # = 0 near equilibrium

        return [y_0, y_1, out_of_hole]  # shape: (batch, 3)
```

**3 outputs quan trọng:**
- `y_0 = V(x)`: phải > 0 với mọi `x ≠ 0`
- `y_1 = V(x_{t+1}) - V(x_t)`: phải ≤ `-κV(x)` trong ROA
- `out_of_hole`: = 0 trong ball nhỏ quanh gốc (dùng để skip kiểm tra Lyapunov tại x*)

---

## 4. Formulation mới: ROA lớn hơn

### 4.1 Định nghĩa các thành phần

```
B     = [-1,1]⁴            # Region-of-interest (box)
S     = {x ∈ B | V(x) < ρ} # Invariant set (ROA ước lượng)
ρ     = γ · min_{x ∈ ∂B} V(x)  # Sublevel set threshold (γ < 1)
κ     = hằng số dương (convergence rate)
F(x)  = V(f_cl(x)) - (1-κ)·V(x)  # Lyapunov decrease condition
H(x') = ‖ReLU(x' - x_up)‖₁ + ‖ReLU(x_lo - x')‖₁  # Out-of-box penalty
```

### 4.2 Verification condition (Theorem 3.3)

Cần verify:
```
∀x ∈ B:  (-F(x) ≥ 0  ∧  x_{t+1} ∈ B)  ∨  (V(x) ≥ ρ)
```

**Diễn giải:** Với mọi điểm x trong box B, hoặc:
- (a) Lyapunov giảm (`-F ≥ 0`) **VÀ** state tiếp theo vẫn trong B → tức x nằm trong ROA và ổn định, **HOẶC**
- (b) `V(x) ≥ ρ` → tức x nằm ngoài ROA, không cần kiểm tra

### 4.3 Training condition (Theorem 3.4)

Loss tương ứng để enforce condition trên:

```python
def lyap_violation_loss(x, rho, c0=1.0):
    x_next = dynamics(x, policy(x))
    F_val  = lyap(x_next) - (1 - kappa) * lyap(x)      # Lyapunov decrease check
    H_val  = out_of_box_penalty(x_next)                  # Box invariance check
    
    combined = F.relu(F_val) + c0 * H_val                # Both violations
    
    # Key: only enforce when x is inside ROA (V < rho)
    loss = F.relu(torch.min(combined, rho - lyap(x)))
    return loss
```

**Tại sao `min(..., ρ - V(x))`?**  
Nếu `V(x) ≥ ρ`, tức `ρ - V(x) ≤ 0`, thì `min(combined, ρ - V(x)) ≤ 0`, nên `relu(...)=0` — không penalize.
Chỉ penalize khi x **trong** ROA và vi phạm điều kiện Lyapunov.

---

## 5. Thuật toán Training (CEGIS + PGD)

### 5.1 CEGIS Framework

**Counter-Example Guided Inductive Synthesis** — vòng lặp xen kẽ:
1. **Inner (Falsifier)**: Tìm counterexample vi phạm Lyapunov condition bằng PGD attack.
2. **Outer (Learner)**: Update network weights để fix các counterexample đã tìm được.

### 5.2 Loss function đầy đủ

```python
L(θ; D, ρ) = Σ_{x_adv ∈ D}  L_Vdot(x_adv; ρ)   # Lyapunov condition violation
           + c1 · L_roa(ρ)                          # Expand ROA
           + c2 · ‖θ‖₁                              # L1 regularization (reduce Lipschitz)
```

**L_roa** — đẩy ROA bao phủ các candidate states:
```python
def L_roa(rho, candidates):
    # candidates: states ta muốn ROA bao phủ
    return sum(relu(lyap(x_cand)/rho - 1) for x_cand in candidates)
```

**‖θ‖₁** — regularize để Lyapunov network có Lipschitz constant nhỏ → dễ verify hơn.

### 5.3 Chi tiết PGD attack (tìm counterexample)

```python
def find_counterexamples(model, x_init, rho, n_steps=50, step_size=0.01):
    x_adv = x_init.clone().requires_grad_(True)
    for _ in range(n_steps):
        loss = lyap_violation_loss(x_adv, rho)
        loss.backward()
        x_adv = x_adv + step_size * x_adv.grad.sign()
        x_adv = project_to_box(x_adv, lo=-1, hi=1)  # stay in B
        x_adv = x_adv.detach().requires_grad_(True)
    return x_adv
```

### 5.4 Tìm ρ (boundary của ROA)

`ρ` được xác định bằng cách:
1. Sample nhiều điểm `x_j` trên biên `∂B`
2. Dùng PGD để **minimize** `V(x_j)` → tìm điểm có V nhỏ nhất trên boundary
3. `ρ = γ · min_j V(x_j)` với `γ < 1` (e.g., `γ = 0.9`)

```python
def compute_rho(model, n_boundary=1000, gamma=0.9, n_pgd_steps=50):
    # Sample points on boundary of B
    x_boundary = sample_boundary(n_boundary)  # ||x||_inf = 1
    
    # Minimize V on boundary via PGD
    for _ in range(n_pgd_steps):
        v = lyap(x_boundary)
        v.sum().backward()
        x_boundary -= alpha * x_boundary.grad
        x_boundary = project_to_boundary(x_boundary)  # keep on ∂B
    
    rho = gamma * lyap(x_boundary).min().item()
    return rho
```

### 5.5 Toàn bộ training loop

```python
def train(model, n_iters=10000, batch_size=512):
    optimizer = Adam(model.parameters(), lr=1e-3)
    counterexamples = []
    
    for iter in range(n_iters):
        # === Step 1: Compute rho ===
        rho = compute_rho(model, gamma=0.9)
        
        # === Step 2: Find counterexamples (PGD) ===
        x_rand = torch.rand(batch_size, 4) * 2 - 1  # uniform in B
        x_adv  = find_counterexamples(model, x_rand, rho)
        counterexamples.append(x_adv.detach())
        
        # === Step 3: Update network ===
        all_cx = torch.cat(counterexamples[-50:])  # sliding window
        
        loss = (
            lyap_violation_loss(all_cx, rho).mean()
            + c1 * L_roa(rho, candidates)
            + c2 * l1_regularization(model)
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iter % 100 == 0:
            print(f"Iter {iter}: loss={loss.item():.4f}, rho={rho:.4f}")
```

---

## 6. Pipeline đầy đủ từ training đến verification

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                           │
│                                                             │
│  1. Khởi tạo PolicyNet (LQR warmstart) + LyapunovNet        │
│  2. CEGIS loop:                                             │
│     a. Tính ρ = γ · min_{∂B} V(x)                          │
│     b. PGD attack → tìm counterexamples vi phạm ΔV ≤ -κV   │
│     c. Update θ để minimize loss (vi phạm + ROA + L1)       │
│  3. Lưu model khi không còn counterexample (hoặc hết iter)  │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│                  VERIFICATION PHASE                         │
│                                                             │
│  1. Xây dựng computation graph cho α,β-CROWN               │
│     - Input: x ∈ B = [-1,1]⁴                               │
│     - Output: y = [V(x), ΔV, out_of_hole]                   │
│  2. Tìm ρ_max bằng bisection:                               │
│     - Verify condition (14) với ρ_test                      │
│     - Nếu verified → ρ_lo = ρ_test                          │
│     - Nếu failed   → ρ_up = ρ_test                          │
│  3. ROA chính thức = {x ∈ B | V(x) < ρ_max}               │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Cấu trúc file code hiện có

### 7.1 `cartpole.py` — Model definition

```
cartpole.py
├── env_params        # Physical constants
├── PolicyNet         # Linear controller: 4 → 1 (no bias)
├── LyapunovNet       # MLP: 4 → 32 → 32 → 1 (no bias, ReLU)
└── Cartpole          # Combined wrapper
    ├── forward(state) → [y0, y1, out_of_hole]  (with_x_next=False)
    └── forward(state) → [y0, y1, out_of_hole, state_next]  (with_x_next=True)
```

### 7.2 `cartpole.pth` — Pretrained weights

```
cartpole.pth:
├── policy_model.policy.weight          shape: [1, 4]   (PolicyNet)
├── lyap_model.lyapunov.0.weight        shape: [32, 4]  (LyapunovNet layer 1)
├── lyap_model.lyapunov.2.weight        shape: [32, 32] (LyapunovNet layer 2)
└── lyap_model.lyapunov.4.weight        shape: [1, 32]  (LyapunovNet layer 3)
```

### 7.3 `cartpole_lyapunov.pth` — Lyapunov weights only

```
cartpole_lyapunov.pth:
├── lyapunov.0.weight    shape: [32, 4]
├── lyapunov.2.weight    shape: [32, 32]
└── lyapunov.4.weight    shape: [1, 32]
```

### 7.4 `cartpole.yaml` — α,β-CROWN config

```yaml
model:
  name: Customized("cartpole.py", "Cartpole")  # dùng with_x_next=False
  path: ${CONFIG_PATH}/models/cartpole.pth

data:
  dataset: box_data(lower=[-1,-1,-1,-1], upper=[1,1,1,1])  # B = [-1,1]⁴

solver:
  batch_size: 100000         # lớn cho GPU throughput
  bound_prop_method: crown   # CROWN bound propagation

bab:
  branching:
    method: sb               # Strong branching
    input_split.enable: True # Split input space
```

### 7.5 `cartpole_with_rho.yaml` — Config cho bisection ρ_max

```yaml
model:
  name: Customized("cartpole.py", "Cartpole", with_x_next=True)  # thêm state_next output
# Dùng spec file cartpole_with_rho.csv (encode condition với ρ cụ thể)
```

---

## 8. Hướng dẫn training thực tế

### 8.1 Cài đặt môi trường

```bash
# Clone repo gốc
git clone https://github.com/Verified-Intelligence/Lyapunov_Stable_NN_Controllers
cd Lyapunov_Stable_NN_Controllers

# Cài dependencies
pip install torch numpy scipy matplotlib
pip install auto_LiRPA  # α,β-CROWN

# Kiểm tra GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### 8.2 Khởi tạo PolicyNet từ LQR

```python
import numpy as np
from scipy import linalg

# Linearize cartpole around x* = 0
# A, B: linearized dynamics matrices (tính từ Jacobian)
A = np.array([...])  # 4x4
B_ctrl = np.array([...])  # 4x1

# LQR: solve Riccati equation
Q = np.diag([1, 0.1, 10, 0.1])  # penalize pole angle most
R = np.array([[0.01]])

P = linalg.solve_discrete_are(A, B_ctrl, Q, R)
K = np.linalg.inv(R + B_ctrl.T @ P @ B_ctrl) @ B_ctrl.T @ P @ A

# Load vào PolicyNet
policy_net.policy.weight.data = torch.tensor(-K, dtype=torch.float32)
```

### 8.3 Training script skeleton

```python
import torch
import torch.nn.functional as F
from cartpole import Cartpole

# Hyperparameters
kappa   = 0.1    # Lyapunov convergence rate
gamma   = 0.9    # ρ = γ · min V(∂B)
c0      = 10.0   # balance Lyapunov vs box invariance
c1      = 10.0   # ROA expansion weight
c2      = 0.01   # L1 regularization
lr      = 1e-3
n_iters = 5000
batch   = 1024

model = Cartpole()
# Load LQR warmstart for policy
# ...

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

counterexample_buffer = []

for iteration in range(n_iters):
    model.eval()
    
    # --- Tính ρ ---
    x_bnd = sample_boundary(2000)  # random points on ||x||_inf = 1
    for _ in range(30):
        x_bnd.requires_grad_(True)
        v = model.lyap_model(x_bnd)
        v.sum().backward()
        x_bnd = (x_bnd - 0.01 * x_bnd.grad.sign()).detach()
        x_bnd = project_boundary(x_bnd)  # giữ trên ∂B
    rho = gamma * model.lyap_model(x_bnd).min().item()
    rho = max(rho, 1e-4)  # prevent collapse
    
    # --- PGD: tìm counterexample ---
    x_rand = torch.rand(batch, 4) * 2 - 1
    x_adv  = x_rand.clone()
    for _ in range(50):
        x_adv.requires_grad_(True)
        out    = model(x_adv)
        v, dv  = out[:, 0:1], out[:, 1:2]
        F_val  = dv + kappa * v                     # ΔV + κV (phải ≤ 0)
        x_next = x_adv + ...                        # hoặc dùng with_x_next=True
        H_val  = out_of_box(x_next)
        violation = F.relu(F_val) + c0 * H_val
        loss_adv = F.relu(torch.min(violation, rho - v)).mean()
        loss_adv.backward()
        x_adv = (x_adv + 0.005 * x_adv.grad.sign()).detach()
        x_adv = x_adv.clamp(-1, 1)
    counterexample_buffer.append(x_adv.detach())
    
    # --- Update network ---
    model.train()
    cx = torch.cat(counterexample_buffer[-20:])
    
    out    = model(cx)
    v, dv  = out[:, 0:1], out[:, 1:2]
    F_val  = dv + kappa * v
    H_val  = ...  # out-of-box check
    violation = F.relu(F_val) + c0 * H_val
    
    L_vdot = F.relu(torch.min(violation, rho - v)).mean()
    
    # Candidate states: sample on LQR sublevel set
    x_cand  = sample_candidates(500)
    L_roa_  = F.relu(model.lyap_model(x_cand) / rho - 1).mean()
    
    L_l1 = sum(p.abs().sum() for p in model.parameters())
    
    loss = L_vdot + c1 * L_roa_ + c2 * L_l1
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if iteration % 100 == 0:
        print(f"[{iteration}] loss={loss.item():.4f} rho={rho:.4f}")

# Lưu model
torch.save(model.state_dict(), 'cartpole_trained.pth')
```

### 8.4 Lưu đúng format để verify

```python
# cartpole.pth cần chứa cả policy và lyapunov
torch.save(model.state_dict(), 'models/cartpole.pth')

# cartpole_lyapunov.pth chỉ chứa lyapunov (nếu cần riêng)
torch.save(model.lyap_model.state_dict(), 'models/cartpole_lyapunov.pth')
```

---

## 9. Verification với α,β-CROWN

### 9.1 Cài đặt α,β-CROWN

```bash
git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN
cd alpha-beta-CROWN
pip install -e .
```

### 9.2 Tạo spec file (.vnnlib)

VNNLIB encode điều kiện verification. Cho mỗi giá trị `ρ_test`, cần tạo spec:

```
; cartpole.vnnlib
; Input: x ∈ [-1,1]⁴ (trừ hole quanh gốc)
; Check: (-F(x) ≥ 0 ∧ x_next ∈ B) ∨ (V(x) ≥ ρ)

(declare-const X_0 Real)  ; x[0]: cart pos
(declare-const X_1 Real)  ; x[1]: cart vel
(declare-const X_2 Real)  ; x[2]: pole angle
(declare-const X_3 Real)  ; x[3]: pole vel

; Input constraints: x ∈ B = [-1,1]⁴
(assert (>= X_0 -1.0)) (assert (<= X_0 1.0))
(assert (>= X_1 -1.0)) (assert (<= X_1 1.0))
(assert (>= X_2 -1.0)) (assert (<= X_2 1.0))
(assert (>= X_3 -1.0)) (assert (<= X_3 1.0))

; Output: y = [V(x), ΔV, out_of_hole]
(declare-const Y_0 Real)  ; V(x)
(declare-const Y_1 Real)  ; ΔV = V(x_next) - V(x)
(declare-const Y_2 Real)  ; out_of_hole

; Property to DISPROVE (find violation):
; NOT ((-ΔV - κV ≥ 0 ∧ x_next ∈ B) ∨ V(x) ≥ ρ)
; ≡ (ΔV + κV > 0 ∨ x_next ∉ B) ∧ V(x) < ρ
(assert (< Y_0 RHO))     ; V(x) < ρ  (inside ROA)
(assert (> Y_1 -KAPPA))  ; ΔV > -κV  (Lyapunov NOT decreasing)
```

### 9.3 Chạy verification

```bash
# Cấu trúc thư mục cần thiết
mkdir -p verification/models verification/specs

cp cartpole.py cartpole.pth verification/
cp cartpole.yaml cartpole.vnnlib verification/

# Tạo CSV file (list các spec cần verify)
echo "cartpole.vnnlib,cartpole_out.csv" > verification/specs/cartpole.csv

# Chạy α,β-CROWN
cd alpha-beta-CROWN
CONFIG_PATH=../verification python abcrown.py \
    --config ../verification/cartpole.yaml
```

### 9.4 Bisection để tìm ρ_max

```python
def find_rho_max(config, rho_init, lambda_factor=1.2, tol=1e-3):
    rho = rho_init
    
    # Tìm initial bounds
    if verify(config, rho):
        rho_lo = rho
        while verify(config, rho * lambda_factor):
            rho *= lambda_factor
        rho_up = rho * lambda_factor
    else:
        rho_up = rho
        while not verify(config, rho / lambda_factor):
            rho /= lambda_factor
        rho_lo = rho / lambda_factor
    
    # Bisection
    while rho_up - rho_lo > tol:
        rho_mid = (rho_lo + rho_up) / 2
        if verify(config, rho_mid):
            rho_lo = rho_mid
        else:
            rho_up = rho_mid
    
    return rho_lo  # ρ_max verified
```

### 9.5 Kết quả mong đợi

Theo bài báo, với Cartpole:
- Verification time: **~129 giây** (vs 448s của DITL với MIP)
- ROA nontrivially intersects boundary of `B`
- Larger certified ROA so with previous approaches

---

## 10. Debugging và tips thực tiễn

### 10.1 Vấn đề thường gặp

**1. ρ sụp đổ về 0:**
```python
# Nguyên nhân: V nhỏ quá trên boundary
# Fix: add lower bound constraint cho ρ, hoặc tăng L_roa weight
rho = max(rho, 0.01)
```

**2. PGD không tìm được counterexample nhưng verification fail:**
```python
# Nguyên nhân: PGD bị local optima
# Fix: nhiều random restarts
x_adv = best_of_k_pgd_restarts(x_rand, rho, k=10)
```

**3. Loss giảm nhưng verification fail:**
Thường do L1 regularization chưa đủ mạnh → tăng `c2`.
α,β-CROWN khó verify network có Lipschitz constant lớn.

**4. NaN trong training:**
```python
# Fix: gradient clipping + check for degenerate states
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 10.2 Checklist trước khi chạy verification

```
□ Model output đúng format: [V(x), ΔV, out_of_hole]
□ V(x*) = 0: forward(zeros(4)) → y_0 ≈ 0
□ V(x) > 0: forward(random_x) → y_0 > 0 (không phải luôn đúng với raw NN)
□ ΔV < 0 cho hầu hết states gần gốc
□ File .pth, .yaml, .vnnlib đúng path
□ cartpole.py accessible từ working directory của α,β-CROWN
□ batch_size trong yaml phù hợp với GPU memory
```

### 10.3 Monitoring training

```python
metrics_to_track = {
    'rho':            rho,                    # phải tăng dần
    'loss_vdot':      L_vdot.item(),          # phải giảm
    'loss_roa':       L_roa_.item(),          # phải giảm
    'max_violation':  violation.max().item(), # phải giảm về 0
    'n_counterex':    len(counterexample_buffer),
    'policy_norm':    model.policy_model.policy.weight.norm().item(),
}
```

### 10.4 Khi nào dừng training?

Training đủ tốt khi **PGD attack không còn tìm được counterexample** trong nhiều iterations liên tiếp — đây là signal để chuyển sang verification phase.

```python
if max_violation < threshold:
    no_violation_count += 1
else:
    no_violation_count = 0

if no_violation_count >= 20:  # 20 iters liên tiếp không có vi phạm
    print("Ready for verification!")
    break
```

---

## Tóm tắt công thức quan trọng

| Ký hiệu | Định nghĩa | Ý nghĩa |
|---------|-----------|---------|
| `B` | `[-1,1]⁴` | Region-of-interest box |
| `S` | `{x ∈ B \| V(x) < ρ}` | Certified ROA (invariant set) |
| `ρ` | `γ · min_{∂B} V(x)` | ROA threshold |
| `κ` | constant > 0 | Exponential convergence rate |
| `F(x)` | `V(f_cl(x)) - (1-κ)V(x)` | Lyapunov decrease indicator |
| `H(x')` | `‖ReLU(x'-x_up)‖₁ + ‖ReLU(x_lo-x')‖₁` | Out-of-box penalty |
| `L_Vdot` | `ReLU(min(ReLU(F)+c₀H, ρ-V))` | Training loss per sample |
| `L_roa` | `Σ ReLU(V(x_cand)/ρ - 1)` | ROA expansion loss |

**Verification condition (Theorem 3.3):**
```
∀x ∈ B:  (-F(x) ≥ 0  ∧  x_{t+1} ∈ B)  ∨  (V(x) ≥ ρ)
```

---

*Tài liệu này tổng hợp từ: Yang et al. "Lyapunov-stable Neural Control for State and Output Feedback: A Novel Formulation", ICML 2024, và source code tại github.com/Verified-Intelligence/Lyapunov_Stable_NN_Controllers*