from dataclasses import dataclass, field


@dataclass
class SystemConfig:
    name: str = "pendulum"


@dataclass
class BoxConfig:
    x_min: tuple[float, ...] = (-3.1415, -8.0)
    x_max: tuple[float, ...] = (3.1415, 8.0)


@dataclass
class ModelConfig:
    nx: int = 2
    nu: int = 1
    u_bound: float = 6.0
    state_limits: tuple[float, ...] = (3.1415, 8.0)


@dataclass
class AttackerConfig:
    num_steps: int = 80
    step_size: float = 0.02
    num_restarts: int = 6


@dataclass
class CEGISConfig:
    bank_capacity: int = 200000
    replay_new_ratio: float = 0.35
    violation_margin: float = 5e-4
    local_box_radius: float = 0.20
    local_box_samples: int = 512
    local_box_weight: float = 0.35
    equilibrium_weight: float = 0.1


@dataclass
class TrainingLoopConfig:
    pretrain_epochs: int = 150
    cegis_epochs: int = 350
    alpha_lyap: float = 0.08
    learning_rate: float = 1e-3
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
    controller_path: str = "checkpoints/pendulum/pendulum_controller.pth"
    lyapunov_path: str = "checkpoints/pendulum/pendulum_lyapunov.pth"


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
        ),
        box=BoxConfig(
            x_min=(-1.0, -1.0, -1.0, -1.0),
            x_max=(1.0, 1.0, 1.0, 1.0),
        ),
        loop=TrainingLoopConfig(
            pretrain_epochs=120,
            cegis_epochs=320,
            alpha_lyap=0.05,
            learning_rate=1e-3,
            batch_size=768,
            attack_seed_size=384,
            train_batch_size=768,
            learner_updates=3,
            sweep_every=0,
            checkpoint_every=20,
            lqr_anchor_radius=(0.12, 0.12, 0.12, 0.12),
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
