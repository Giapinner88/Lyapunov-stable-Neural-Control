# Lyapunov-stable Neural Control for State and Output Feedback: A Novel Formulation

Our work studies neural network controllers and observers with provable Lyapunov stability guarantees. Due to the complexity of neural networks and nonlinear system dynamics, it is often challenging to obtain formal stability guarantees for learning-enabled systems. Our work tackles this challenge with a few key contributions:

* We propose a novel formulation for stability verification that **theoretically defines a larger verifiable region-of-attraction** (ROA) than shown in the literature.
* We propose a **new training framework** for learning neural network (NN) controllers/observers together with Lyapunov certificates using fast empirical falsification and strategic regularization.
* Our method **does not rely on expensive solvers** such as MIP, SMT, or SDP in both the training and verification stages.
* For the first time in literature, we synthesize and formally verify **neural-network controllers and observers** (output-feedback) together with Lyapunov functions for **general nonlinear** dynamical systems and observation functions.

More details can be found in **our paper:**

*Lujie Yang\*, Hongkai Dai\*, Zhouxing Shi, Cho-Jui Hsieh, Russ Tedrake, and Huan Zhang*
"[Lyapunov-stable Neural Control for State and Output Feedback: A Novel Formulation](https://arxiv.org/pdf/2404.07956.pdf)" (\*Equal contribution)

## Pendulum Focus

This repository is currently organized around the pendulum model, with separate runtime entry points for:

* state-feedback training
* output-feedback training

The structure remains modular (`neural_lyapunov_training/`) so additional models can be added later without changing the core architecture.

# Code

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

We have provided the trained pendulum models and specifications for verification.
To run verification with the provided specifications:

```bash
cd verification
export CONFIG_PATH=$(pwd)
cd complete_verifier

python abcrown.py --config $CONFIG_PATH/pendulum_state_feedback_lyapunov_in_levelset.yaml
python abcrown.py --config $CONFIG_PATH/pendulum_output_feedback_lyapunov_in_levelset.yaml
```

## Training

Run pendulum training directly from the new app layer:

```bash
python -m apps.pendulum.state_feedback
python -m apps.pendulum.output_feedback
```

or as scripts:

```bash
python apps/pendulum/state_feedback.py
python apps/pendulum/output_feedback.py
```

Hydra configuration files are in `apps/pendulum/config/`.

To override user-specific settings:

```bash
python -m apps.pendulum.state_feedback user=USERNAME
```

Use `apps/pendulum/config/user/state_feedback_default.yaml` as a template.

To reproduce the pendulum state-feedback result in the paper:

```bash
python -m apps.pendulum.state_feedback --config-name state_feedback_reproduce
```

Each run creates an output directory specified by `user.run_dir`, containing learned models, wandb logs, and the exact `config.yaml` used for reproducibility.

## Preparing Specifications for Verification

This section is for the stability verification of new models (e.g., controllers, observers, and/or Lyapunov functions trained from scratch). If you use the pre-trained models described [above](#verification), these instructions are not needed since we already provided specifications in this repo.

Theorem 3.3 in our paper defines the condition for stability verification. The steps below first find the largest $\rho$ in Theorem 3.3 using a bisection script.

### Bisection for $\rho$

We can use bisection to find the largest $\rho$ that satisfies the verification objective.
We use the script `neural_lyapunov_training/bisect.py` for automatic bisection.

We need to specify the region for verification using `--lower_limit`, `--upper_limit`, and `--hole_size`.
The `--lower_limit` and `--upper_limit` define the region of interests $\mathcal{B}$ (which is problem specific; see Table 3 in the Appendix of our paper). The `--hole_size` excludes a very small region (default 0.1%) around the origin, which the verifier may have numerical issues with since the Lyapunov function values are very close to 0.

We provide an initial $\rho$ value by `--init_rho`
and specify the precision for the bisection by `--rho_eps`.
`--init_rho` is the initial guess of sublevel set value, and you can use the `rho` reported after training finishes (see training steps above).
For verification during the bisection, a configuration file needs to be specified
by `--config` and a timeout value is needed by `--timeout`.
Optionally, a `--output_folder` argument may be used to specify an output folder the bisection.
Additional arguments for the verifier may also be provided.

For models which may take a long time to verify, you may add `--check_x_adv_only`
to only check $\xi_{t+1}\in\mathcal{B}$ but not $-F(\xi_t)>0$
(Theorem 3.3 in our paper) during the bisection.
In case that the $\rho$ you obtain from the bisection does not lead to a
successful verification when you generate full specifications using the $\rho$
(see the next sections), you may further reduce $\rho$ manually.

```bash
cd verification
python -m neural_lyapunov_training.bisect \
--lower_limit -12 -12 --upper_limit 12 12 --hole_size 0.001 \
--init_rho 603.5202 --rho_eps 0.1 \
--config pendulum_state_feedback_lyapunov_in_levelset.yaml \
--timeout 100
```

The bisection will output the result of each bisection iteration. For example:
```txt
Generating specs with rho=603.5202
Start verification
Output path: ./output/rho_603.52020.txt
Result: defaultdict(<class 'list'>, {'safe': [0, 1, 2, 3]})
safe
```
In the end, it will output the lower bound and upper bound for $\rho$ as:
```txt
rho_l=708.0606252685546
rho_u=708.1342971679687
```
We can take the lower bound of $\rho$ denoted as `rho_l` above. Note that if you are _not_ using `--check_x_adv_only`, then the model is verified with the sublevel set value `rho_l`; in this case, the next step is not necessary, although it can still be helpful to save the specifications with this specific `rho_l` for reproducing the verification results without bisection.

### Specification generation with a specific $\rho$

After obtaining a suitable $\rho$ value,
we generate the specifications for verification with a fixed $\rho$.
We use VNNLIB format files to describe the specification.
The command below will generate several VNNLIB files and a CSV file with a list of
all VNNLIB filenames. Each VNNLIB file describes a subproblem to verify, and
the number of subproblems is determined by the state dimension.
Similar to the aforementioned bisection, `--lower_limit`, `--upper_limit`, and `--hole_size` need to be specified. And the sublevel set value $\rho$ is provided by `--value_levelset`.
All specification files will be saved in the `specs` folder.

```
cd verification
python -m neural_lyapunov_training.generate_vnnlib \
--lower_limit -12 -12 --upper_limit 12 12 --hole_size 0.001 \
--value_levelset 672 \
specs/pendulum_state_feedback
```

You can then run verification, as we mentioned in the "Verification" section.
