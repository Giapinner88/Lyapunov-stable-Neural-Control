# THEORY: Lyapunov-stable Neural Control

Muc tieu cua file nay:
- Thuat lai chi tiet thuat toan dang xay dung trong repo.
- Lien ket giua ly thuyet va implementation hien tai.
- Tong hop tai lieu tham khao trong thu muc docs de lam co so cho cac buoc tiep theo.

---

## 1) Bai toan

Ta xet he phi tuyen roi rac:

$$
x_{t+1} = f(x_t, u_t), \quad u_t = \pi_\theta(x_t)
$$

Muc tieu la hoc dong thoi:
- Controller neural network $\pi_\theta(x)$.
- Lyapunov neural network $V_\phi(x)$.

Sao cho dieu kien giam Lyapunov dung trong mot mien lon nhat co the:

$$
V(x_{t+1}) - (1-\alpha)V(x_t) \le 0, \quad \alpha \in (0,1)
$$

Neu dieu kien tren dung trong mot tap con cua khong gian trang thai, tap do la ung vien ROA (region of attraction) duoc chung minh.

---

## 2) Nen tang tham khao tu docs

Tai lieu trong docs:
- docs/2404.07956v2.pdf

Y chinh rut ra va ap dung cho repo nay:
- Hoc dong thoi policy va Lyapunov certificate.
- Dung falsification nhanh (PGD attacker) de tim phan vi du.
- Dung regularization va thiet ke bai toan huan luyen de mo rong mien co the chung minh.
- Hau kiem (post-hoc verification) la buoc quan trong de chuyen tu ket qua thuc nghiem sang bao dam hinh thuc.

Noi dung hien tai cua repo tap trung vao phan synthesis/training:
- LQR pre-training de tao khoi tao on dinh gan diem can bang.
- CEGIS voi attacker va Counterexample Bank de lien tuc mo rong mien on dinh empirically.

---

## 3) Cau truc mo hinh hien tai

1. Dynamics:
- PendulumDynamics voi bo tich phan RK4.
- Dynamics ke thua nn.Module de dong bo device bang to(device).

2. Controller:
- NeuralController sinh luc dieu khien, co gioi han bien do dau ra bang tanh va u_bound.

3. Lyapunov network:
- NeuralLyapunov xay dung dang:

$$
V(x)=\|\phi_V(x)-\phi_V(0)\|_1 + x^T(\epsilon I + R^T R)x
$$

- Thanh phan thu hai dam bao tinh positive-definite co cau truc.

4. Attacker:
- PGD theo chuan L-infinity voi cap nhat theo dau gradient.
- Co multi-restart va seed strategy de giam ket o cuc dai gia.

5. Learner + Bank:
- Counterexample Bank luu x_bad cua nhieu epoch.
- Moi buoc hoc dung batch tron loi moi va loi cu.

---

## 4) Thuat toan huan luyen 2 phase

### Phase 1: LQR pre-training (local warm start)

Muc tieu:
- Day controller giong hanh vi LQR quanh goc.
- Day Lyapunov network xap xi dang bac hai tu ma tran S cua Riccati.

Buoc hoc:
1. Mau x_small trong ban kinh nho quanh goc.
2. Tinh u_lqr = -Kx, kep theo gioi hanh dieu khien vat ly.
3. Toi uu tong loss:

$$
\mathcal{L}_{pre} = \text{MSE}(u_{nn},u_{lqr}) + \text{MSE}(V_{nn},x^TSx)
$$

Y nghia:
- Tao khoi tao tot, giup Phase 2 on dinh hon va nhanh hon.

### Phase 2: CEGIS loop voi Counterexample Bank

Muc tieu:
- Tim va va cac diem vi pham manh nhat theo vung dang hoc.

Vong lap moi epoch:
1. Attacker nhan x_seed, toi uu de cuc dai hoa:

$$
	ext{violation}(x)=V(x_{next})-(1-\alpha)V(x)
$$

2. Nop x_bad vao Counterexample Bank.
3. Learner lay batch tron (moi + cu) tu bank, toi uu:

$$
\mathcal{L}_{lyap}=\mathbb{E}[\text{ReLU}(V(x_{next})-(1-\alpha)V(x))]
$$

4. Lap lai cho den khi loss nho va bank bao hoa theo suc chua.

---

## 5) Vi sao can Counterexample Bank

Neu chi hoc tren loi moi cua epoch hien tai, mo hinh de quen loi cu (catastrophic forgetting).

Counterexample Bank giai quyet viec do bang cach:
- Giu lich su phan vi du kho.
- Tron phan vi du moi + cu trong moi buoc hoc.
- Tao curriculum tu dong: loi de duoc xu ly som, loi kho con lai trong bank se tiep tuc duoc phat hien va sua.

---

## 6) Cac bat bien implementation can giu

1. Muc tieu cua attacker va learner phai dong nhat theo cung alpha.
2. Dynamics, models, seeds, va bounds phai cung device.
3. Khi doi architecture Lyapunov, phai kiem tra lai:
- Shape cua tat ca Linear layers.
- Tinh toan phi_V(0) voi shape dung va khong can graph gradient.
4. Log training can theo doi dong thoi:
- LQR pretrain loss.
- Violation loss.
- Kich thuoc Counterexample Bank.

---

## 7) Lo trinh tiep theo

1. Verification hinh thuc:
- Ket noi quy trinh hau kiem voi bo verifier trong verification.
- Dinh nghia ro mien can chung minh va cach trich ROA.

2. Nang cap attacker:
- Priority replay theo muc violation.
- Adaptive step size va so restart theo giai doan training.

3. Mo rong output feedback:
- Them observer neural va bai toan chung minh ket hop controller-observer theo huong cua paper trong docs.

---

## 8) Nhat ky cap nhat

- Ban hien tai: da co 2-phase training (LQR pre-training + CEGIS) va Counterexample Bank tich hop.
- Muc tieu tiep theo: chot quy trinh verification end-to-end de co bao dam hinh thuc cho ROA.
