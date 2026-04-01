# Lyapunov-stable Neural Control for State and Output Feedback

**Research Application for Nonlinear System Stabilization**

This repository implements the methodology from the paper:

*Lujie Yang\*, Hongkai Dai\*, Zhouxing Shi, Cho-Jui Hsieh, Russ Tedrake, and Huan Zhang*
"[Lyapunov-stable Neural Control for State and Output Feedback: A Novel Formulation](https://arxiv.org/pdf/2404.07956.pdf)" (\*Equal contribution)

## Purpose

This project synthesizes and formally verifies neural network controllers and observers with provable Lyapunov stability guarantees. The main focus is on **nonlinear control systems**, particularly:
- **Pendulum** (main case studies: state-feedback and output-feedback control)
- **Cartpole** (in baseline comparisons)

### Key Contributions

* Novel formulation for stability verification with **theoretically larger verifiable region-of-attraction (ROA)** than prior work
* New training framework combining neural network controllers/observers with Lyapunov certificates using empirical falsification and strategic regularization
* **No expensive solvers** (MIP, SMT, SDP) required during training or verification
* **Output-feedback synthesis**: neural-network controllers and observers co-designed with Lyapunov functions for general nonlinear dynamics

## Repository Structure

```
project-root/
├── apps/pendulum/                    # Pendulum training runtime layer
│   ├── state_feedback.py             # Entry point for state-feedback training
│   ├── output_feedback.py            # Entry point for output-feedback training
│   └── config/                       # Hydra configuration files
│       ├── state_feedback.yaml       # Main config for state-feedback
│       ├── output_feedback.yaml      # Main config for output-feedback
│       └── user/                     # User-specific overrides
│
├── neural_lyapunov_training/         # Core learning/verification package
│   ├── models.py                     # Neural network architectures (Lyapunov, controller, observer)
│   ├── controllers.py                # Controller synthesis utilities
│   ├── dynamical_system.py           # Pendulum/system dynamics definitions
│   ├── lyapunov.py                   # Lyapunov function and stability conditions
│   ├── train_utils.py                # Training loss, callbacks, logging
│   ├── bisect.py                     # Binary search for max ROA radius (ρ)
│   ├── generate_vnnlib.py            # Generate verification specifications
│   └── try_verifier.py               # Verification interface
│
├── data/pendulum/                    # Training data and checkpoints
│   ├── state_feedback/               # State-feedback datasets and initialized weights
│   └── output_feedback/              # Output-feedback datasets and initialized weights
│
├── models/                           # Pre-trained neural network controllers
│   ├── pendulum_state_feedback.pth
│   ├── pendulum_output_feedback.pth
│   └── (other pre-trained models)
│
├── baselines/nlc_discrete/           # Baseline: NLC discrete implementation
│   ├── pendulum.py                   # NLC controller for pendulum
│   ├── models/                       # Pre-trained baseline models
│   └── specs/                        # Verification specifications
│
├── verification/                     # Verification setup and configs
│   ├── *.yaml                        # Verification configuration files (state/output feedback)
│   ├── complete_verifier/            # Alpha-beta-CROWN verifier (external)
│   └── specs/                        # Generated VNNLIB verification specs
│
├── tests/                            # Unit tests for debugging
│   ├── test_controllers.py
│   ├── test_lyapunov.py
│   ├── test_pendulum.py
│   └── test_quadrotor2d.py
│
├── assets/                           # Documentation and resources
├── alpha-beta-CROWN/                 # External verifier package (git submodule)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

### Key Directories Explained

| Directory | Purpose |
|-----------|---------|
| **apps/pendulum/** | Runtime entrypoints for pendulum training. Direct script execution with Hydra config management. |
| **neural_lyapunov_training/** | Core reusable package for Lyapunov synthesis, training, and stability verification logic. Designed modular for extension to other systems. |
| **data/pendulum/** | Training datasets, initial weights for Lyapunov/controller, and saved checkpoints during training. |
| **models/** | Pre-trained neural network models ready for immediate use in control or verification. |
| **baselines/nlc_discrete/** | Reference NLC (Neural Lyapunov Control) discrete-time implementation for pendulum comparison. |
| **verification/** | Verification infrastructure: config files and specification generation for formal stability proof. |
| **tests/** | Unit tests and debugging utilities to validate dynamics, controllers, and Lyapunov conditions. |

# Quick Start

## Installation

Create a conda environment and install the dependencies except those for verification:
```bash
conda create --name lnc python=3.11
conda activate lnc
pip install -r requirements.txt
```

We use [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA.git) and [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN.git) for verification. To install both of them, run:
```bash
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
(cd alpha-beta-CROWN/auto_LiRPA && pip install -e .)
(cd alpha-beta-CROWN/complete_verifier && pip install -r requirements.txt)
```

To set up the path:
```
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/alpha-beta-CROWN:$(pwd)/alpha-beta-CROWN/complete_verifier"
```

## Verification

Pre-trained pendulum models and specifications are included. To verify model stability:

```bash
cd verification
export CONFIG_PATH=$(pwd)
cd complete_verifier

# Verify state-feedback controller
python abcrown.py --config $CONFIG_PATH/pendulum_state_feedback_lyapunov_in_levelset.yaml

# Verify output-feedback (controller + observer)
python abcrown.py --config $CONFIG_PATH/pendulum_output_feedback_lyapunov_in_levelset.yaml
```

**Verification Output:** Each run produces a certificate in the form:
- **ROA (Region of Attraction)**: The largest sublevel set $\mathcal{L}_\rho = \{\xi : V(\xi) \leq \rho\}$ where stability is formally proven
- **Lyapunov function** $V(\xi)$ (neural network)
- **Controller/Observer** (neural networks)

## Training

Run pendulum training from the new modular app layer:

```bash
# State-feedback: train neural Lyapunov + controller
python apps/pendulum/state_feedback.py

# Output-feedback: train neural Lyapunov + controller + observer together
python apps/pendulum/output_feedback.py
```

Alternatively, use module invocation:
```bash
python -m apps.pendulum.state_feedback
python -m apps.pendulum.output_feedback
```

### Hydra Configuration

Hydra configs are in **`apps/pendulum/config/`**. To override defaults:

```bash
# Use a specific user config
python apps/pendulum/state_feedback.py user=my_user

# Override specific parameters
python apps/pendulum/state_feedback.py seed=42 model.kappa=0.001

# Reproduce paper results
python apps/pendulum/state_feedback.py --config-name state_feedback_reproduce
```

**User Config Template:** `apps/pendulum/config/user/state_feedback_default.yaml`  
Create your own user profile by adding a new YAML in that directory.

**Training Outputs:** Each run creates a directory in `user.run_dir` (default: `./output/pendulum_state/`):
- `checkpoints/` – Model weights at each epoch
- `config.yaml` – Exact config used (for reproducibility)
- `wandb/` – WandB logs (if enabled)

### Configuration Tuning for ROA & Stability

Configuration files in **`apps/pendulum/config/`** control training behavior, ROA expansion, and stability verification. Key parameters to tune:

#### Model Architecture & Lyapunov

| Parameter | File | Effect | Default |
|-----------|------|--------|---------|
| `model.kappa` | `state_feedback.yaml` | Regularization weight for Lyapunov PSD constraint. **↑ kappa** → enforces stronger positive definiteness but may shrink ROA. | `0.01` |
| `model.V_decrease_within_roa` | `state_feedback.yaml` | If `true`, enforces $-F(x) > 0$ (Lyapunov decrease) within sublevel set. | `true` |
| `model.V_psd_form` | `state_feedback.yaml` | Lyapunov parameterization: `L1` (via Cholesky) or `quadratic` (direct). `L1` is more stable. | `L1` |
| `model.lyapunov.quadratic` | `state_feedback.yaml` | If `true`, Lyapunov is quadratic (small ROA, fast). If `false`, neural network (larger ROA, slower). | `false` |
| `model.lyapunov.hidden_widths` | `state_feedback.yaml` | Layer widths of Lyapunov NN. **↑ width** → more expressive, larger potential ROA but slower training. | `[16]` |

**Tuning Strategy for Larger ROA:**
```yaml
model:
  kappa: 0.001               # Reduce regularization strength
  V_psd_form: L1            # Keep Cholesky-based parameterization
  lyapunov:
    quadratic: false        # Use neural network (not quadratic)
    hidden_widths: [32, 16] # Increase network capacity
```

#### Training Dynamics & Loss

| Parameter | File | Effect | Default |
|-----------|------|--------|---------|
| `train.loss_weightage.v` | `state_feedback.yaml` | Weight for Lyapunov positivity loss. **↑ v** → enforces $V > 0$ harder. | `1.0` |
| `train.loss_weightage.roa` | `state_feedback.yaml` | Weight for ROA expansion loss. **↑ roa** → maximizes $\rho$ more aggressively. | `1.0` |
| `train.loss_weightage.decrease` | `state_feedback.yaml` | Weight for Lyapunov decrease ($-F > 0$) loss. **↑ decrease** → ensures stability. | `1.0` |
| `train.sample_num` | `state_feedback.yaml` | Number of training samples per epoch. **↑ sample_num** → better coverage, more time. | `2000` |
| `train.batch_size` | `state_feedback.yaml` | Batch size for SGD. **↑ batch_size** → faster but less gradient noise. | `64` |
| `train.num_epochs` | `state_feedback.yaml` | Total training iterations. **↑ epochs** → longer training, potentially larger ROA. | `50` |

**Tuning Strategy for Faster Training + Reasonable ROA:**
```yaml
train:
  loss_weightage:
    v: 0.5          # Reduce Lyapunov positivity weight
    roa: 1.5        # Increase ROA expansion focus
    decrease: 1.0
  sample_num: 1000  # Fewer samples per epoch
  batch_size: 128   # Larger batches
  num_epochs: 30    # Fewer epochs
```

#### Controller & System

| Parameter | File | Effect | Default |
|-----------|------|--------|---------|
| `model.controller.activation` | `state_feedback.yaml` | Activation function for controller NN. `tanh`, `relu`, `elu`. `tanh` bounded, `relu` unbounded. | `tanh` |
| `model.controller.limit_scale` | `state_feedback.yaml` | Output saturation: $u \in [\!-\text{scale}, \text{scale}\!]$. **↑ scale** → weaker control limits, easier stabilization. | `[0.1, ..., 1.0]` |
| `dynamics.dt` | `state_feedback.yaml` | Discrete timestep for simulation. **↓ dt** → more accurate integration, stricter stability. | `0.01` |

**Tuning Strategy for Tighter Verification (Smaller ROA but Formally Verified):**
```yaml
model:
  controller:
    activation: tanh
    limit_scale: [0.1, 0.2, 0.3, 0.4, 0.5]  # Conservative torque limits
dynamics:
  dt: 0.001  # Smaller timestep for tighter integration
```

#### Override via Command Line

Instead of editing YAML, override at runtime:
```bash
# Expand Lyapunov network & reduce regularization
python apps/pendulum/state_feedback.py \
  model.kappa=0.0001 \
  model.lyapunov.hidden_widths=[64,32] \
  train.loss_weightage.roa=2.0 \
  train.num_epochs=100

# Conservative controller with tight verification
python apps/pendulum/state_feedback.py \
  model.controller.limit_scale=[0.1,0.15,0.2,0.25,0.3] \
  dynamics.dt=0.001 \
  user=my_conservative_user
```

#### Output Feedback Tuning

**`apps/pendulum/config/output_feedback.yaml`** additionally includes observer parameters:

| Parameter | Effect | Default |
|-----------|--------|---------|
| `model.observer.hidden_widths` | Observer NN capacity. **↑ width** → better state estimation, smoother controller input. | `[16]` |
| `train.loss_weightage.observer` | Weight for observer training loss. **↑ weight** → enforce faster state error convergence. | `1.0` |

**Tuning for Better Observer:**
```yaml
model:
  observer:
    hidden_widths: [32, 16]  # Larger observer network
  controller:
    hidden_widths: [32, 16]
train:
  loss_weightage:
    observer: 1.5  # Prioritize observer quality
```

**Example: Full Custom Configuration**

Create `apps/pendulum/config/user/my_large_roa.yaml`:
```yaml
# Focus on maximizing ROA with moderate verification time
seed: 1234
run_dir: ./output/pendulum_large_roa/${now:%Y-%m-%d}/${now:%H-%M-%S}
wandb_enabled: false

# Expand Lyapunov & reduce regularization
model:
  kappa: 0.0001
  lyapunov:
    hidden_widths: [64, 32]
  controller:
    hidden_widths: [32, 16]

# Aggressive ROA optimization
train:
  num_epochs: 100
  sample_num: 5000
  loss_weightage:
    roa: 2.0
    v: 0.8
    decrease: 1.0

# Data paths
load_lyaloss: data/pendulum/state_feedback/lyaloss_init.pth
```

Then run:
```bash
python apps/pendulum/state_feedback.py user=my_large_roa
```

### Pendulum Dynamics

Pendulum model is defined in **`neural_lyapunov_training/pendulum.py`**. It includes:
- Continuous-time dynamics: $\dot\theta = \omega$, $\dot\omega = -g/\ell \sin\theta - d \cdot m \cdot \omega / I + u / I$
- Discrete-time simulation with configurable timestep
- Linearization and LQR reference

### Training Framework

The training pipeline integrates:
1. **Controller** (from `apps/pendulum/state_feedback.py` or `output_feedback.py`)
2. **Dynamics** (from `neural_lyapunov_training/dynamical_system.py`)
3. **Lyapunov Loss** (from `neural_lyapunov_training/lyapunov.py`)
4. **Training Loop** (from `neural_lyapunov_training/train_utils.py`)

Key files:
- **`neural_lyapunov_training/models.py`** – Defines `LyapunovNet` (Lyapunov function), `Controller`, and `Observer` architectures
- **`neural_lyapunov_training/controllers.py`** – Utilities for controller synthesis and rollout
- **`neural_lyapunov_training/levelset.py`** – Sublevel set verification and ROA estimation

## Custom Verification (Advanced)

**For newly trained models only.** Pre-trained model specifications are already provided in `verification/specs/`.

Theorem 3.3 in the paper defines the stability condition. To verify a custom trained model:

### Step 1: Find Maximum ROA via Bisection

Use **`neural_lyapunov_training/bisect.py`** to binary-search the largest $\rho$ satisfying the stability condition:

```bash
cd verification
python -m neural_lyapunov_training.bisect \
  --lower_limit -12 -12 --upper_limit 12 12 --hole_size 0.001 \
  --init_rho 603.5 --rho_eps 0.1 \
  --config pendulum_state_feedback_lyapunov_in_levelset.yaml \
  --timeout 100
```

**Parameters:**
- `--lower_limit / --upper_limit` – Verification region (problem-specific; see Table 3 in paper appendix)
- `--hole_size` – Exclude small region around origin (default 0.1%; numerically safe)
- `--init_rho` – Initial guess for sublevel set value ρ
- `--config` – Verification config YAML file

**Output:** Bisection results with final bounds `rho_l` (lower, verified) and `rho_u` (upper, unverified).

### Step 2: Generate Specifications

After obtaining $\rho_l$, generate VNNLIB specification files:

```bash
cd verification
python -m neural_lyapunov_training.generate_vnnlib \
  --lower_limit -12 -12 --upper_limit 12 12 --hole_size 0.001 \
  --value_levelset 672 \
  specs/pendulum_state_feedback_custom
```

---

## Core Modules Reference

### `neural_lyapunov_training/` Package

| File | Purpose |
|------|---------|
| `models.py` | Neural network architectures for Lyapunov ($V$), controller ($u$), and observer ($\hat{x}$) |
| `dynamical_system.py` | System dynamics interface (Pendulum, Cartpole, etc.) |
| `controllers.py` | Controller training utilities and LQR reference |
| `lyapunov.py` | Lyapunov stability conditions and verification |
| `train_utils.py` | Loss functions, training loops, callbacks, WandB integration |
| `levelset.py` | Sublevel set estimation and ROA computation |
| `bisect.py` | Binary search script for maximum ROA radius ρ |
| `generate_vnnlib.py` | Generate VNNLIB verification specifications |
| `try_verifier.py` | Verification interface with Alpha-Beta-CROWN |

### `apps/pendulum/` Runtime

| File | Purpose |
|------|---------|
| `state_feedback.py` | Entrypoint for state-feedback training (train $V$ + controller $u$ from full state) |
| `output_feedback.py` | Entrypoint for output-feedback training (train $V$ + controller $u$ + observer $\hat{x}$ jointly) |
| `config/` | Hydra configuration hierarchy |

### Data Organization

- **`data/pendulum/state_feedback/`** – Initial weights, training datasets for state-feedback experiments
- **`data/pendulum/output_feedback/`** – Initial weights, training datasets for output-feedback experiments
- **`models/`** – Pre-trained `.pth` checkpoints (ready to use)
- **`baselines/nlc_discrete/models/`** – Pre-trained baseline models for comparison

### Verification Stack

- **`verification/`** – Top-level verification directory
  - **`*.yaml`** – Verification configs (one per system+feedback type)
  - **`specs/`** – Generated VNNLIB specification files
  - **`complete_verifier/`** – Alpha-Beta-CROWN verifier (git submodule from external repo)

---

## Paper Theorems & Implementation

### Theorem 3.3: Stability Condition

A system with controller $u = \mu_\theta(x)$ and Lyapunov function $V$ is stable on ROA if:

$$
\begin{cases}
V(x) > 0 & \forall x \in \mathcal{L}_\rho \setminus \{0\} \\
-F(x) > 0 & \forall x \in \mathcal{L}_\rho \setminus \{0\}
\end{cases}
$$

where $\mathcal{L}_\rho = \{x : V(x) \leq \rho\}$ and $F(x) := \nabla V(x)^T (Ax + Bu + d)$ (or discrete variant).

**Implementation:** See:
- `neural_lyapunov_training/lyapunov.py` – Loss computation
- `neural_lyapunov_training/bisect.py` – Verification loop
- `verification/*.yaml` – Verifier input specifications

### Training Objective

Jointly minimize:
- **Lyapunov decrease:** $-F(x_t) > 0$ within ROA
- **Positive definiteness:** $V(x) > 0$ for $x \neq 0$
- **ROA size:** Maximize $\rho$

See `neural_lyapunov_training/train_utils.py:compute_lyapunov_loss()`.

---

## Experiments

### Reproduce Paper Results (Pendulum)

**State-Feedback:**
```bash
python apps/pendulum/state_feedback.py --config-name state_feedback_reproduce
```

**Output-Feedback:**
```bash
python apps/pendulum/output_feedback.py --config-name output_feedback_reproduce
```

### Baselines

Compare with NLC Discrete-time baseline (pre-computed):
```bash
# Baseline is in baselines/nlc_discrete/pendulum.py
# Models and specs are in baselines/nlc_discrete/models/ and specs/
```

---

## Troubleshooting

### ModuleNotFoundError: neural_lyapunov_training
- **Cause:** Python path not properly set
- **Solution:** Run scripts from project root, or ensure `sys.path.insert(0, <project-root>)` in entry script

### Hydra Warning: "_self_ missing"
- **Expected:** Hydra will warn about missing `_self_` in defaults
- **Fix:** Add `- _self_` at end of defaults list in config YAML

### WandB Warning about pkg_resources
- **Expected:** External deprecation warning from setuptools
- **Impact:** Non-fatal; logging still works

### Verification timeout
- **Cause:** Complex model or tight ROA
- **Solution:** Increase `--timeout`, or reduce `--rho_eps` for faster bisection convergence
