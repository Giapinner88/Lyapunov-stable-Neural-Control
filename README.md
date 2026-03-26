# Lyapunov-Stable Neural Control

Repository for Lyapunov-stable neural control with two systems:
- CartPole (priority track)
- Pendulum (kept for continuation)

## Current Status

### CartPole progress: ~80%

Implemented:
- OOP training pipeline with CEGIS/PGD in [core/trainer.py](core/trainer.py)
- CartPole dynamics and LQR baseline in [core/dynamics.py](core/dynamics.py)
- Config-based training setup in [core/training_config.py](core/training_config.py)
- ROA utilities and rho estimation in [core/roa_utils.py](core/roa_utils.py)
- Verification CLI with sample-based bisection in [verify.py](verify.py)
- Optional CROWN local-radius check in [core/verification.py](core/verification.py)
- CartPole evaluation script in [evaluate_cartpole.py](evaluate_cartpole.py)
- Direct-execution import fixes for scripts (no more `No module named core`)

Pending / not fully completed yet:
- Full alpha-beta-CROWN complete_verifier workflow integration for CartPole specs
- Better certified ROA quality (current sample-based ROA ratio can be very small)
- Unified benchmark report pipeline for reproducible final metrics

## Repository Layout

```text
core/                    # Models, dynamics, training, verification modules
checkpoints/
  cartpole/              # CartPole checkpoints
  pendulum/              # Pendulum checkpoints
reports/                 # Generated verification/evaluation outputs
notes/                   # Project notes/logs
docs/
  RUN_REPO.md            # Practical run commands
  CARTPOLE_README.md     # CartPole details
  pipeline.md            # Theory/pipeline notes
```

## Quick Start

### 1) Environment

```bash
conda activate lypen
pip install -r requirements.txt
```

Optional for CROWN local verification:

```bash
pip install auto-LiRPA
```

If using local alpha-beta-CROWN clone in this repo:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/alpha-beta-CROWN:$(pwd)/alpha-beta-CROWN/complete_verifier"
```

### 2) Train CartPole

```bash
python train.py --system cartpole --pretrain-epochs 120 --cegis-epochs 320 --alpha-lyap 0.05
```

Outputs:
- checkpoints/cartpole/cartpole_controller.pth
- checkpoints/cartpole/cartpole_lyapunov.pth

### 3) Verify CartPole

Sample-based only:

```bash
python verify.py --skip-crown --output-dir reports/verification_results
```

With CROWN local radius:

```bash
python verify.py --crown-method CROWN --crown-eps-max 0.2 --output-dir reports/verification_results
```

### 4) Evaluate CartPole

```bash
python evaluate_cartpole.py --n-tests 100 --output-dir reports/evaluation_results
```

## Pendulum Track (kept)

- Phase portrait: [evaluate_pendulum.py](evaluate_pendulum.py)
- Diagnostic scripts: [diagnostic.py](diagnostic.py), [test_verifier.py](test_verifier.py)
- Comparison script: [compare_methods.py](compare_methods.py)

## Documentation

- Practical run guide: [docs/RUN_REPO.md](docs/RUN_REPO.md)
- CartPole details: [docs/CARTPOLE_README.md](docs/CARTPOLE_README.md)
- Theory notes: [docs/pipeline.md](docs/pipeline.md), [docs/THEORY.md](docs/THEORY.md)