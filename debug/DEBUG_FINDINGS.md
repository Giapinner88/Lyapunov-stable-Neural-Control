# Debug Log: Pendulum Output-Feedback vs MuJoCo (upright = 0)

## 1) Muc tieu va trieu chung

- Muc tieu mong muon: con lac giu can bang quanh upright, voi bien loi goc theta_norm = 0.
- Trieu chung quan sat: khi dung checkpoint NN output-feedback hien co, quy dao khong hoi tu ve 0 trong MuJoCo; nhieu case hoi tu ve vung gan pi (downward).

## 2) Loi ky thuat gap trong qua trinh debug

- ModuleNotFoundError: matplotlib (khi chay script phan tich CSV trong env thieu package).
- ModuleNotFoundError: mujoco (khi chay script bang env khong co mujoco).
- SyntaxError trong cac script debug tam thoi do sai indentation/f-string.
- Sai khop architecture observer khi thu checkpoint khac:
  - `pendulum_output_feedback_nn.pth` khong khop observer [8, 8]
  - can observer hidden la [16, 16, 8, 8] moi load duoc.

## 3) Cac thu nghiem da lam va ket qua

### 3.1 Kiem tra he quy chieu goc trong MuJoCo

- `qpos=0` -> sensor theta raw = 0
- `qpos=pi` -> sensor theta raw = pi
- Cong thuc `theta_norm = wrap_to_pi(theta_raw - pi)` hoat dong dung.

Ket luan: khong phai bug wrap goc.

### 3.2 Thu NN-only voi checkpoint chinh

- Dung `models/pendulum_output_feedback.pth`, khoi tao gan upright (`pi + 0.01`).
- Ket qua dai han: `theta_norm` hoi tu gan `3.06` (khong ve 0).

Ket luan: policy hien tai khong giu upright trong MuJoCo.

### 3.3 Thu dao dau torque va dao dau do goc

- A/B test cac to hop dau cua measurement va torque.
- Khong co to hop nao dua he ve upright on dinh tai 0 nhu ky vong.

Ket luan: khong phai loi don gian do nguoc dau actuator/sensor.

### 3.4 Thu checkpoint output-feedback khac

- `pendulum_output_feedback_small_torque.pth`: van khong hoi tu upright.
- `pendulum_output_feedback_nn.pth`: load duoc khi dung observer [16,16,8,8], nhung van khong hoi tu upright trong setup hien tai.

Ket luan: van de khong nam rieng o 1 checkpoint cu the.

### 3.5 Kiem tra local behavior cua dynamics train + controller

Kiem tra truc tiep trong `PendulumDynamics.forward` quanh theta = 0, u = 0:

- `theta = +0.1` -> `d(theta_dot) = -1.958732`
- `theta = -0.1` -> `d(theta_dot) = +1.958732`

Nghia la open-loop dang keo ve 0 (0 la diem on dinh tu nhien trong mo hinh train).

Fit tuyen tinh local cua policy NN output-feedback quanh 0:

- `u ~= a*theta + b*theta_dot + c*y + d`
- `a,b,c,d = [-0.097775, -0.815126, -0.097775, -0.013408]`
- Vi du: `u(theta=+0.05) = -0.020364`, `u(theta=-0.05) = +0.020364`

Ket luan: checkpoint khong "fail ngau nhien"; no hoc hanh vi nhat quan voi convention dynamics dang train.

## 4) Kham pha quan trong (root cause)

- Mismatch ve convention vat ly giua y nghia upright mong muon trong mo phong va convention trong pipeline train output-feedback hien tai.
- Do do, chi "set quy tac" trong simulation la khong du de bien checkpoint hien co thanh upright-stabilizer.

Noi cach khac:

- Van de chinh khong phai dong code mo phong bi sai wrap/scale.
- Van de chinh khong phai checkpoint hu hong ngau nhien.
- Van de chinh la su khong dong nhat convention giua train model va muc tieu upright trong MuJoCo.

## 5) Trang thai script mo phong sau khi fix de chay on dinh

- `simulation/simulate_pendulum.py` da duoc chinh ve balance-only upright an toan bang PD:
  - khoi tao gan upright (`pi + 0.05`)
  - dieu khien PD quanh `theta_norm = 0`
  - clamp theo `actuator_ctrlrange` cua MuJoCo
  - co cong tac `USE_NEURAL_CONTROLLER` de so sanh.

Dieu nay cho phep mo phong can bang upright on dinh ngay lap tuc trong khi cho retrain dung convention.

## 6) Ket luan cuoi cung

- Nghi ngo "training lan truoc co the fail" la hop ly, nhung bang chung cho thay checkpoint dang hoc nhat quan voi mo hinh train hien tai.
- De dat dung yeu cau "upright = 0" trong MuJoCo bang NN output-feedback, can retrain voi convention dynamics phu hop muc tieu upright (khong chi sua simulation).
