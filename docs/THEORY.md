# THEORY: Lyapunov-stable Neural Control (Detailed, RK4 Discrete-Time)

Muc tieu cua tai lieu nay:
- Chot ly thuyet dang dung trong code hien tai.
- Lam ro cac bat bien can giu giua train va verify.
- Tong hop cac loi da gap, nguyen nhan goc, va cach sua da ap dung.

---

## 1) Bai toan va mo hinh roi rac

Ta chung minh on dinh cho he da lay mau:

$$
x_{k+1} = \Phi(x_k, u_k), \quad u_k = \pi_\theta(x_k)
$$

Trong repo nay, $\Phi$ la map roi rac sinh boi bo tich phan RK4 tu he lien tuc:

$$
\dot x = f_c(x,u), \qquad
\Phi(x,u) = \text{RK4}(f_c, x, u, dt)
$$

Ly do dung he roi rac:
- Controller duoc trien khai theo chu ky lay mau.
- Verifier va huan luyen deu thao tac one-step tren $x_k \to x_{k+1}$.

---

## 2) Dieu kien Lyapunov roi rac dang su dung

Dieu kien muc tieu:

$$
V(x_{k+1}) - (1-\alpha)V(x_k) \le 0, \quad \alpha \in (0,1)
$$

Tuong duong:

$$
V(x_{k+1}) - V(x_k) + \rho V(x_k) \le 0, \quad \rho = \alpha
$$

Trong verifier, dai luong duoc kiem la:

$$
	ext{violation}(x) = V(x_{k+1}) - V(x_k) + \rho V(x_k)
$$

Y nghia:
- Neu upper bound cua violation < 0 tren mot hop, hop do duoc chung minh an toan.
- Neu upper bound > 0, co the la vi pham that, hoac bound con bao thu.

---

## 3) Kien truc V(x) va bao dam nghiem ngat

Dang V trong code:

$$
V(x)=\|\phi_V(x)-\phi_V(0)\|_1 + x^T P x, \quad P=\epsilon I + R^T R
$$

Voi $\epsilon>0$, ma tran $P$ la SPD. He qua:
- $V(0)=0$.
- $V(x)>0$ voi moi $x\neq 0$.

Luu y quan trong:
- Da bo hoan toan offset cong hang cuoi mang (kieu +c).
- Khong dung ReLU o dau ra V de tranh tao nen duong gia.

---

## 4) Dieu kien can bang tai goc cho controller

De tranh day he ra khoi diem can bang, code controller da cuong buc:

$$
u(x)=\big(\hat u(x)-\hat u(0)\big)\,u_{bound}
$$

Nen $u(0)=0$ dung chinh xac, khong phu thuoc vao sai so hoc cua trong so.

Tac dung:
- Giam nguy co $x=0$ bi day sang trang thai khac sau 1 buoc RK4.
- Giu chat che dieu kien violation tai tam hop verify.

---

## 5) Train pipeline hien tai (2 phase + CEGIS)

### Phase 1: LQR pretraining
- Hoc controller theo LQR quanh goc.
- Hoc V theo dang bac hai $x^T S x$ quanh goc.

Loss:

$$
\mathcal{L}_{pre} = \text{MSE}(u_{nn},u_{lqr}) + \text{MSE}(V_{nn},x^TSx)
$$

### Phase 2: CEGIS
- Attacker (PGD, multi-restart) tim diem vi pham.
- Counterexample Bank luu loi moi + loi cu.
- Learner toi uu tren batch tron de tranh quen loi.

Loss chinh trong learner:

$$
\mathcal{L}_{global}=\mathbb{E}[\text{ReLU}(\text{violation}+m)]
$$

Da them thanh phan moi de ep verify tot hon:
- Local-box decrease loss quanh goc.
- Equilibrium loss tai goc ($u(0)$ va $V(0)$).
- Violation margin $m>0$ de tang do chat dieu kien.

---

## 6) Verify pipeline hien tai

Script test_verifier da ho tro 2 che do:
- Chay don 1 eps.
- Quet eps de tim nguong upper bound am.

Nguyen tac doc ket qua:
- UB < 0: chung minh thanh cong cho hop dang xet.
- UB > 0: chua chung minh duoc. Can phan tich tiep:
	- Sampling co vi pham that hay khong.
	- Hoac bound CROWN bao thu do hop qua lon, phi tuyen manh.

---

## 7) Cac loi da gap va cach sua (postmortem)

### Loi 1: khoi tao Lyapunov thieu tham so
Trieu chung:
- TypeError khi tao NeuralLyapunov.

Nguyen nhan:
- Constructor can nx nhung test tao khong truyen.

Sua:
- Dong bo khoi tao voi nx=2 o test/evaluate/export.

### Loi 2: API RK4 khong dong nhat
Trieu chung:
- Goi dynamics.rk4_step(...) fail o mot so entry point.

Nguyen nhan:
- BaseDynamics co step(...) nhung verifier/export goi rk4_step(...).

Sua:
- Them alias rk4_step(...) trong BaseDynamics, goi ve step(...).

### Loi 3: auto_LiRPA khong ho tro onnx::Einsum
Trieu chung:
- Tracing fail voi unsupported operation Einsum.

Nguyen nhan:
- torch.einsum trong tinh toan $x^TPx$ bi export thanh onnx::Einsum.

Sua:
- Doi sang dang matmul + sum tuong duong.

### Loi 4: warning BoundSub "constant operand has batch dimension"
Trieu chung:
- Warning khi lan truyen bound.

Nguyen nhan:
- phi(0) duoc giu dang tensor co batch [1,n].

Sua:
- Chuyen origin ve [n], khi can moi unsqueeze/squeeze dung diem.

### Loi 5: checkpoint cu khong nap duoc sau khi doi shape origin
Trieu chung:
- size mismatch origin [1,n] vs [n].

Nguyen nhan:
- Thay doi architecture/buffer shape nhung van dung checkpoint cu.

Sua:
- Them backward compatibility trong load_state_dict: neu origin [1,n] thi squeeze.

### Loi 6: UB luon duong du da giam eps
Trieu chung:
- Sweep eps khong tim duoc UB < 0.

Chan doan da thuc hien:
- Kiem tra u(0), violation(0).
- Quet nhay rho.
- Sampling nhieu diem trong hop de tach "vi pham that" va "bound bao thu".

Ket luan:
- Co vung vi pham that trong hop (sample max violation > 0), khong chi do bound.

Xu ly:
- Tang luc CEGIS: attacker manh hon, margin, local-box loss, equilibrium loss, replay ratio.
- Can retrain theo objective moi, checkpoint cu khong du dieu kien de chung minh.

---

## 8) Bat bien can giu de tranh vo pipeline

1. Dong nhat he roi rac:
- Train, eval, verify phai dung cung map RK4 va cung dt.

2. Dong nhat hyperparam canh tranh:
- alpha_lyap (train) va rho (verify) can nhat quan theo muc tieu so sanh.

3. Dong nhat entry points:
- train.py, evaluate.py, test_verifier.py, core/export.py phai trung constructor nx/nu/hidden/u_bound.

4. Tuong thich checkpoint:
- Moi thay doi shape buffer/param can co lop backward compatibility hoac retrain.

5. Neu doi dang V(x):
- Bat buoc kiem lai V(0), u(0), violation(0) truoc khi tin ket qua sweep.

---

## 9) Quy trinh debug khuyen nghi (thuc chien)

1. Kiem tra sanity tai goc:
- V(0) phai = 0.
- u(0) phai = 0.
- violation(0) nen <= 0.

2. Chay verify 1 eps nho (vi du 1e-3 den 1e-2).

3. Quet eps de xac dinh xu huong UB.

4. Neu UB duong:
- Sampling trong hop de kiem tra co vi pham that hay khong.
- Neu co: quay lai train (tang luc attacker/loss).
- Neu khong: chia nho hop hoac tang do chat verifier.

---

## 10) Trang thai hien tai va huong tiep theo

Trang thai hien tai:
- Pipeline RK4 train/verify da thong.
- Da sua cac loi runtime/chinh sach quan trong trong graph va checkpoint.
- Da co bo cong cu quet eps de danh gia kha nang chung minh.

Huong tiep theo de dat UB am:
- Retrain dai hon voi objective moi.
- Verify theo lo trinh eps tu nho den lon.
- Neu can, chia mien thanh nhieu hop de giam bao thu CROWN.
