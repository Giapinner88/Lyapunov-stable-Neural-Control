# Huong Dan Chay Repo

Tai lieu nay huong dan chay nhanh toan bo repo cho ca CartPole va Pendulum.

## 1. Chuan bi moi truong

```bash
cd /home/giapinner88/Project/Lyapunov-stable-Neural-Control

# Kich hoat env (neu ban dang dung env lypen)
conda activate lypen

# Cai thu vien co ban
pip install -r requirements.txt

# Neu chua co auto_LiRPA (cho verify CROWN)
pip install auto-LiRPA
```

Neu ban muon dung them alpha-beta-CROWN complete_verifier:

```bash
# Da clone san alpha-beta-CROWN trong repo
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/alpha-beta-CROWN:$(pwd)/alpha-beta-CROWN/complete_verifier"
```

## 2. Train CartPole

```bash
python train.py --system cartpole --pretrain-epochs 120 --cegis-epochs 320 --alpha-lyap 0.05
```

Checkpoint se duoc luu mac dinh tai:
- checkpoints/cartpole/cartpole_controller.pth
- checkpoints/cartpole/cartpole_lyapunov.pth

## 3. Verify CartPole

### 3.1 Verify nhanh (sample-based)

```bash
python verify.py --skip-crown --output-dir reports/verification_results
```

### 3.2 Verify co CROWN local radius

```bash
python verify.py \
  --controller checkpoints/cartpole/cartpole_controller.pth \
  --lyapunov checkpoints/cartpole/cartpole_lyapunov.pth \
  --crown-method CROWN \
  --crown-eps-max 0.2 \
  --output-dir reports/verification_results
```

## 4. Evaluate CartPole

```bash
python evaluate_cartpole.py \
  --controller checkpoints/cartpole/cartpole_controller.pth \
  --lyapunov checkpoints/cartpole/cartpole_lyapunov.pth \
  --n-tests 100 \
  --output-dir reports/evaluation_results
```

## 5. Chay cac script Pendulum (giu lai de lam tiep)

### 5.1 Ve phase portrait

```bash
python evaluate.py
```

Anh se duoc luu tai:
- reports/pendulum_phase_portrait.png

### 5.2 So sanh baseline

```bash
python compare_methods.py --output reports/comparison_report.md
```

Checkpoint Pendulum mac dinh:
- checkpoints/pendulum/pendulum_controller.pth
- checkpoints/pendulum/pendulum_lyapunov.pth

## 6. Lenh test nhanh khi gap loi import

Neu ban muon kiem tra file co chay truc tiep duoc khong:

```bash
/home/giapinner88/miniconda3/envs/lypen/bin/python /home/giapinner88/Project/Lyapunov-stable-Neural-Control/core/trainer.py
/home/giapinner88/miniconda3/envs/lypen/bin/python /home/giapinner88/Project/Lyapunov-stable-Neural-Control/verify.py --help
```

## 7. Thu tu chay khuyen nghi cho CartPole

1. Train: `python train.py --system cartpole`
2. Verify: `python verify.py --output-dir reports/verification_results`
3. Evaluate: `python evaluate_cartpole.py --output-dir reports/evaluation_results`

## 8. File ket qua quan trong

- reports/verification_results/verification_summary.txt
- reports/evaluation_results/eval_summary.txt
- checkpoints/cartpole/cartpole_controller.pth
- checkpoints/cartpole/cartpole_lyapunov.pth
