# Lyapunov-Stable Neural Control (CartPole)

Repository is streamlined to CartPole-first workflow and reorganized to match the reference style:
- `examples/` for runnable workflows
- `neural_lyapunov_training/` for core algorithms
- minimal root entrypoints for train/evaluate/verify/export

## Structure

```text
neural_lyapunov_training/   # Models, dynamics, CEGIS trainer, verification logic
examples/                   # cartpole_training.py, cartpole_evaluation.py, cartpole_verification.py
alpha-beta-CROWN/           # verifier stack
checkpoints/                # trained model weights
train.py                    # training CLI entrypoint
evaluate_cartpole.py        # evaluation CLI entrypoint
verify.py                   # verification CLI entrypoint
export_vnnlib.py            # ONNX/VNNLIB export
run_full_pipeline.py        # orchestration
cartpole_verification.yaml  # alpha-beta-CROWN config
```

## Quick Start

```bash
conda activate lnc
pip install -r requirements.txt
```

### Train

```bash
python train.py --system cartpole
```

### Evaluate

```bash
python evaluate_cartpole.py --n-tests 100 --eval-scale 0.4 --output-dir evaluation_results
```

### Verify

```bash
python verify.py --alpha-lyap 0.01 --skip-crown --output-dir verification_results
```

### Export verifier artifacts

```bash
python export_vnnlib.py --alpha-lyap 0.01 --tolerance 1e-6
```

## Paper-aligned training additions

- Bias-enabled MLP blocks (controller + Lyapunov network)
- Candidate-ROA regularization integrated into CEGIS loss
- Local/equilibrium/LQR-anchor regularizers enabled in CartPole profile

Candidate-ROA CLI knobs:

```bash
python train.py --system cartpole \
  --candidate-roa-weight 0.2 \
  --candidate-roa-num-samples 256 \
  --candidate-roa-scale 0.4 \
  --candidate-roa-rho-quantile 0.9
```
