# Lyapunov-Stable Neural Control

Repository for Lyapunov-stable neural control with two systems:
- CartPole (priority track)
- Pendulum (kept for continuation)

## Current Status

### CartPole progress: Implementation Complete, Quality Gap Identified

**Completed Components:**
- OOP training pipeline with CEGIS/PGD in [core/trainer.py](core/trainer.py)
- CartPole dynamics and LQR baseline in [core/dynamics.py](core/dynamics.py)
- Config-based training setup in [core/training_config.py](core/training_config.py)
- ROA utilities and rho estimation in [core/roa_utils.py](core/roa_utils.py)
- Verification CLI with sample-based bisection in [verify.py](verify.py)
- Optional CROWN local-radius check in [core/verification.py](core/verification.py)
- CartPole evaluation script in [evaluate_cartpole.py](evaluate_cartpole.py)
- Method comparison analysis in [compare_roa_regions.py](compare_roa_regions.py)
- Training results analysis in [analyze_training_results.py](analyze_training_results.py)

**Key Findings:**
- Neural Lyapunov approach achieves point-wise stability but poor formal ROA
- Quadratic LQR baseline significantly outperforms neural methods in verified ROA (~68x)
- CROWN upper bounds are too loose for negative certification
- Gap indicates verification challenge rather than training failure

## Results

### Evaluation Metrics (100 test trajectories)
| Metric | Value |
|--------|-------|
| Convergence Rate | 10.0% |
| Lyapunov Decrease Rate | 10.0% |
| Stabilization Rate | 52.0% |

### Verification Results (Region of Attraction)
| Method | Verified ROA (%) | Verified ρ | Sublevel Area (%) |
|--------|------------------|-----------|------------------|
| Neural (NN Controller + NN Lyapunov) | 0.77% | 0.000586 | 0.0116% |
| Quadratic (LQR + V(x)=x^T P x) | 52.85% | 0.047120 | 0.5285% |
| **Performance Gap** | **68.5x** | **80.4x** | **45.7x** |

### Formal Verification Bounds (CROWN)
| Method | Max Violation (Point-wise) | Mean Violation | CROWN Upper Bound | Alpha-CROWN Upper Bound |
|--------|---------------------------|----------------|-------------------|------------------------|
| Quadratic | -0.000000 | -0.001898 | 0.165 | 0.165 |
| Neural | 0.000006 | -0.001220 | 0.026 | 0.026 |



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
python evaluate_cartpole.py --n-tests 100 --output-dir evaluation_results
```

### 5) Compare Methods

View ROA comparison between neural and quadratic Lyapunov:

```bash
python compare_roa_regions.py --output-dir reports
```

Analyze training and verification results:

```bash
python analyze_training_results.py --output-dir reports
```

## Pendulum Track (kept)

- Phase portrait: [evaluate_pendulum.py](evaluate_pendulum.py)
- Legacy diagnostic/comparison scripts were removed during repository cleanup.

## Documentation

- Practical run guide: [docs/RUN_REPO.md](docs/RUN_REPO.md)
- CartPole details: [docs/CARTPOLE_README.md](docs/CARTPOLE_README.md)

## Known Issues & Analysis

### Verification Challenge (CROWN Looseness)
- **Issue**: CROWN provides only loose upper bounds, preventing negative certification
- **Observation**: Point-wise testing shows stability, but formal bounds remain positive (≥0)
- **Root Cause**: ReLU dependency tracking loss in neural networks; CROWN linear approximation insufficiency
- **Impact**: A system stable in practice may not be formally verifiable with CROWN
- **Reference**: See [notes/RETHINK_APPROACH.txt](notes/RETHINK_APPROACH.txt) for detailed analysis

### Neural vs. Quadratic Performance Gap
- Neural methods achieve **68.5x smaller verified ROA** than quadratic LQR baseline
- Suggests fundamental limitation in neural Lyapunov certification with current methods
- May require alternative verification approaches (beyond CROWN bounds)

## Future Work

1. **Verification Strategy Alternatives**
   - Investigate tight-bound verifiers beyond CROWN
   - Consider mixed-integer formulations or semidefinite programming
   - Evaluate alpha-beta-CROWN complete_verifier for tighter bounds

2. **Neural Lyapunov Improvements**
   - Design loss functions encouraging tighter verification bounds
   - Explore alternative network architectures (e.g., polynomial networks)
   - Test higher-order Taylor-based approximations

3. **Hybrid Approaches**
   - Combine neural controllers with quadratic Lyapunov functions
   - Use neural networks for local refinement within LQR baseline ROA
- Theory notes: [docs/pipeline.md](docs/pipeline.md), [docs/THEORY.md](docs/THEORY.md)