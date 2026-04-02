# MuJoCo Implementation Strategy While Preserving Lyapunov Certification

## 1) Problem Statement

Mục tiêu là chạy controller output-feedback ổn định trên MuJoCo mà vẫn giữ được pipeline Lyapunov hiện có.

Vấn đề gốc không nằm ở "train thế nào" mà nằm ở "train trên dynamics nào":
- Nếu train trên dynamics xấp xỉ, controller sẽ tối ưu cho dynamics đó.
- Khi deploy trên MuJoCo, plant thực khác (integration, timestep, actuator, nonlinearities) nên xuất hiện mismatch.

Kết luận: muốn giảm mismatch, cần học theo dynamics MuJoCo.

## 2) Design Principle

Giữ nguyên phần lõi có giá trị nhất:
- Lyapunov loss
- Controller + Observer structure
- Verification flow (PGD checks, rho computation)

Chỉ thay thế plant dùng cho training:
- Từ dynamics analytic hiện tại
- Sang dynamics surrogate khả vi được học từ dữ liệu MuJoCo

Cách này gọi là Hybrid: MuJoCo-supervised dynamics + Lyapunov-certified policy training.

## 3) Why Not Train Directly End-to-End on MuJoCo?

Training hiện tại cần backprop qua dynamics trong loss output-feedback.
MuJoCo không cung cấp autograd trực tiếp theo cùng luồng này.

Do đó có 2 đường:
1. Model-free trực tiếp MuJoCo: sát plant nhưng khó giữ chứng nhận Lyapunov hiện tại.
2. Hybrid surrogate: vẫn backprop được và giữ framework chứng nhận.

Tài liệu này chọn đường 2.

## 4) Current Pipeline Anchors (To Keep)

Các điểm then chốt trong code hiện tại cần giữ:
- Entry training output-feedback: apps/pendulum/output_feedback.py
- Dynamics interface: neural_lyapunov_training/dynamical_system.py
- Lyapunov DOF derivative loss: neural_lyapunov_training/lyapunov.py
- Runtime simulation on MuJoCo: simulation/simulate_pendulum.py

## 5) Target Hybrid Architecture

### 5.1 Data Flow

1. MuJoCo Data Collector tạo dataset transition:
   - input: x_t, u_t
   - target: x_{t+1}

2. Surrogate Dynamics Trainer học map rời rạc:
   - x_{t+1} = f_hat(x_t, u_t)

3. Output-feedback trainer thay dynamics backend sang f_hat:
   - giữ nguyên controller/observer/Lyapunov losses

4. Verification giữ nguyên flow:
   - rho estimation
   - PGD violation checks

5. Closed-loop validation chạy lại trên MuJoCo.

### 5.2 Scope of Change

Không thay:
- Objective và logic Lyapunov/observer/controller
- Format checkpoint canonical cho simulation

Có thay:
- Nguồn dynamics dùng trong training
- Dataset + training script cho surrogate
- Config để chọn dynamics backend

## 6) Implementation Plan (Phased)

## Phase A: MuJoCo Transition Dataset

Mục tiêu: tạo dữ liệu đủ phủ vùng hoạt động.

A1. State/action ranges
- theta_norm trong [-pi, pi]
- theta_dot theo giới hạn vật lý thực tế
- torque theo range deploy thực (khớp với actuator mapping)

A2. Sampling strategy
- Uniform + biased near equilibrium
- Include hard regions: high velocity, near angle wrap boundaries

A3. Saved fields
- x_t, u_t, x_{t+1}
- optional: y_t, integration meta, domain randomization tags

A4. Domain randomization (khuyến nghị)
- damping, mass, friction perturbation nhẹ
- giúp policy robust hơn khi sai số mô hình còn tồn tại

## Phase B: Train Surrogate Dynamics

Mục tiêu: surrogate vừa đúng one-step vừa ít drift multi-step.

B1. One-step loss
- MSE(x_{t+1}, f_hat(x_t, u_t))

B2. Multi-step rollout loss
- rollout H bước để giảm lỗi tích lũy

B3. Acceptance criteria
- one-step error thấp trong toàn miền
- multi-step rollout ổn định trong horizon quan tâm

B4. Outputs
- surrogate checkpoint
- scaler/normalization stats
- evaluation report

## Phase C: Integrate into Output-Feedback Training

Mục tiêu: thay dynamics backend với thay đổi tối thiểu.

C1. Thêm config switch
- dynamics.backend: analytic | surrogate
- dynamics.surrogate_ckpt: path

C2. Khởi tạo dynamics theo backend
- analytic: logic cũ
- surrogate: load f_hat và bọc theo interface DiscreteTimeSystem

C3. Giữ nguyên phần còn lại
- LyapunovDerivativeDOFLoss
- ObserverLoss
- staged training by limit_scale/rho_multiplier
- PGD verifier

## Phase D: DAgger-Style Refinement (Optional but Recommended)

Mục tiêu: thu hẹp distribution gap.

D1. Dùng policy hiện tại rollout trên MuJoCo
D2. Thu thêm transition đúng vùng state policy thường đi qua
D3. Re-train surrogate
D4. Re-train policy + verify
D5. Lặp 2-3 vòng

## 7) Certification and Validation Strategy

## 7.1 What Certification Still Means

Chứng nhận Lyapunov thu được là theo dynamics dùng để train/verify (surrogate).
Vì vậy cần bổ sung bằng chứng "surrogate gần MuJoCo" để tăng độ tin cậy khi deploy.

## 7.2 Required Validation on MuJoCo

1. Monte Carlo closed-loop stability rate
2. Observer error convergence statistics
3. Torque clipping ratio
4. Violation proxy metrics (no formal proof on raw MuJoCo, but empirical safety checks)

## 8) Practical Risks and Mitigations

R1. Frame mismatch (angle wrapping, upright convention)
- Mitigation: chuẩn hóa state adapter thống nhất train/sim/deploy

R2. Time-step mismatch
- Mitigation: đồng nhất control update period giữa data collection và training

R3. Actuator scale mismatch
- Mitigation: explicit torque adapter và logging saturation

R4. Surrogate overfit
- Mitigation: multi-step validation + domain randomization + DAgger loop

## 9) Minimal Deliverables

1. Collector script
- ví dụ: tools/collect_mujoco_transitions.py

2. Surrogate training script
- ví dụ: tools/train_surrogate_dynamics.py

3. Surrogate dynamics wrapper
- ví dụ: neural_lyapunov_training/mujoco_surrogate_dynamics.py

4. Config updates
- apps/pendulum/config/output_feedback.yaml

5. Integration in trainer
- apps/pendulum/output_feedback.py

6. Validation script on MuJoCo
- ví dụ: simulation/validate_surrogate_on_mujoco.py

## 10) Suggested 1-Week Execution

Day 1-2
- Collector + dataset v1
- Dataset quality checks

Day 3
- Surrogate v1 training + one-step/multi-step eval

Day 4
- Plug surrogate backend vào output_feedback trainer

Day 5
- Re-train controller/observer/lyapunov + PGD verify

Day 6
- MuJoCo validation runs + diagnostics plots

Day 7
- DAgger refinement vòng 1 + final report

## 11) Decision Guideline

Nếu ưu tiên cao nhất là giữ chứng nhận Lyapunov hiện tại:
- Chọn Hybrid surrogate dynamics.

Nếu ưu tiên cao nhất là performance thuần trên MuJoCo và chấp nhận mất chứng nhận hiện hành:
- Chuyển sang model-free trực tiếp.

Với hiện trạng dự án này, Hybrid là phương án cân bằng tốt nhất giữa tính thực dụng và tính bảo chứng.
