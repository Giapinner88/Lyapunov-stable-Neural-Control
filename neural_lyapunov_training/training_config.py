from dataclasses import dataclass, field


@dataclass
class SystemConfig:
    name: str = "cartpole"


@dataclass
class BoxConfig:
    x_min: tuple[float, ...] = (-1.0, -1.0, -1.0, -1.0)
    x_max: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)


@dataclass
class ModelConfig:
    nx: int = 4
    nu: int = 1
    u_bound: float = 30.0
    state_limits: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    lyapunov_phi_dim: int = 1
    lyapunov_absolute_output: bool = True


@dataclass
class AttackerConfig:
    num_steps: int = 100
    step_size: float = 0.02
    num_restarts: int = 6


@dataclass
class CEGISConfig:
    bank_capacity: int = 200000
    bank_mode: str = "fifo"
    replay_new_ratio: float = 0.35
    violation_margin: float = 1e-6
    local_box_radius: float = 0.20
    local_box_samples: int = 512
    local_box_weight: float = 0.0
    equilibrium_weight: float = 0.0
    lqr_anchor_weight: float = 0.0
    local_sampling_mode: str = "levelset"
    local_levelset_c: float | None = None
    local_levelset_quantile: float = 0.60
    local_levelset_oversample_factor: int = 6
    ibp_ratio: float = 0.0
    ibp_eps: float = 0.01
    candidate_roa_weight: float = 0.0
    candidate_roa_num_samples: int = 0
    candidate_roa_scale: float = 0.4
    candidate_roa_rho: float | None = None
    candidate_roa_rho_quantile: float = 0.9
    candidate_roa_always: bool = False


@dataclass
class TrainingLoopConfig:
    pretrain_epochs: int = 150
    cegis_epochs: int = 350
    alpha_lyap: float = 0.08
    learning_rate: float = 1e-4
    batch_size: int = 512
    attack_seed_size: int = 256
    train_batch_size: int = 512
    learner_updates: int = 3
    sweep_every: int = 50
    checkpoint_every: int = 30
    lqr_anchor_radius: tuple[float, ...] = (0.1, 0.1)


@dataclass
class CurriculumConfig:
    start_scale: float = 0.7
    end_scale: float = 1.0


@dataclass
class OutputConfig:
    controller_path: str = "checkpoints/cartpole/cartpole_controller.pth"
    lyapunov_path: str = "checkpoints/cartpole/cartpole_lyapunov.pth"


@dataclass
class TrainerConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    box: BoxConfig = field(default_factory=BoxConfig)
    attacker: AttackerConfig = field(default_factory=AttackerConfig)
    cegis: CEGISConfig = field(default_factory=CEGISConfig)
    loop: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def pendulum_default_config() -> TrainerConfig:
    return TrainerConfig(
        system=SystemConfig(name="pendulum"),
        model=ModelConfig(
            nx=2,
            nu=1,
            u_bound=6.0,
            state_limits=(3.1415, 8.0),
            lyapunov_phi_dim=1,
            lyapunov_absolute_output=True,
        ),
        box=BoxConfig(
            x_min=(-3.1415, -8.0),
            x_max=(3.1415, 8.0),
        ),
        loop=TrainingLoopConfig(
            pretrain_epochs=150,
            cegis_epochs=350,
            alpha_lyap=0.08,
            learning_rate=1e-3,
            batch_size=512,
            attack_seed_size=256,
            train_batch_size=512,
            learner_updates=3,
            sweep_every=50,
            checkpoint_every=30,
            lqr_anchor_radius=(0.1, 0.1),
        ),
        output=OutputConfig(
            controller_path="checkpoints/pendulum/pendulum_controller.pth",
            lyapunov_path="checkpoints/pendulum/pendulum_lyapunov.pth",
        ),
    )


def cartpole_default_config() -> TrainerConfig:
    return TrainerConfig(
        system=SystemConfig(name="cartpole"),
        model=ModelConfig(
            nx=4,
            nu=1,
            u_bound=30.0,
            state_limits=(1.0, 1.0, 1.0, 1.0),
            lyapunov_phi_dim=1,
            lyapunov_absolute_output=True,
        ),
        box=BoxConfig(
            x_min=(-1.0, -1.0, -1.0, -1.0),
            x_max=(1.0, 1.0, 1.0, 1.0),
        ),
        loop=TrainingLoopConfig(
            pretrain_epochs=150,
            cegis_epochs=400,
            alpha_lyap=0.01,
            learning_rate=1e-4,
            batch_size=1024,
            attack_seed_size=640,
            train_batch_size=1024,
            learner_updates=2,
            sweep_every=20,
            checkpoint_every=20,
            lqr_anchor_radius=(0.10, 0.10, 0.10, 0.10),
        ),
        attacker=AttackerConfig(
            num_steps=120,
            step_size=0.015,
            num_restarts=8,
        ),
        cegis=CEGISConfig(
            bank_capacity=500000,
            bank_mode="reservoir",
            replay_new_ratio=0.20,
            violation_margin=1e-4,
            local_box_radius=0.15,
            local_box_samples=1536,
            local_box_weight=0.30,
            equilibrium_weight=0.05,
            lqr_anchor_weight=0.05,
            local_sampling_mode="levelset",
            local_levelset_c=None,
            local_levelset_quantile=0.60,
            local_levelset_oversample_factor=6,
            ibp_ratio=0.0,
            ibp_eps=0.01,
            candidate_roa_weight=0.2,
            candidate_roa_num_samples=256,
            candidate_roa_scale=0.4,
            candidate_roa_rho=None,
            candidate_roa_rho_quantile=0.9,
            candidate_roa_always=False,
        ),
        curriculum=CurriculumConfig(
            start_scale=0.4,
            end_scale=1.0,
        ),
        output=OutputConfig(
            controller_path="checkpoints/cartpole/cartpole_controller.pth",
            lyapunov_path="checkpoints/cartpole/cartpole_lyapunov.pth",
        ),
    )


def get_default_config(system: str) -> TrainerConfig:
    if system == "pendulum":
        return pendulum_default_config()
    if system == "cartpole":
        return cartpole_default_config()
    raise ValueError(f"Unsupported system: {system}")
