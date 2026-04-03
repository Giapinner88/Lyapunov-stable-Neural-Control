import argparse
from pathlib import Path

import mujoco
import numpy as np


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def get_train_state(data: mujoco.MjData, theta_upright_raw: float) -> np.ndarray:
    theta_raw = float(data.sensor("sens_theta").data[0])
    theta_dot = float(data.sensor("sens_theta_dot").data[0])
    theta_norm = wrap_to_pi(theta_raw - theta_upright_raw)
    return np.array([theta_norm, theta_dot], dtype=np.float32)


def sample_random_action(low: float, high: float, rng: np.random.Generator) -> float:
    return float(rng.uniform(low, high))


def main():
    parser = argparse.ArgumentParser(description="Collect MuJoCo transitions for surrogate training.")
    parser.add_argument("--xml", type=str, default="assets/pendulum.xml", help="Path to MuJoCo XML.")
    parser.add_argument("--out", type=str, default="data/pendulum/mujoco_feedback/mujoco_transitions.npz", help="Output NPZ path.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes.")
    parser.add_argument("--steps-per-episode", type=int, default=200, help="Steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--theta-max", type=float, default=np.pi, help="Initial |theta_norm| max.")
    parser.add_argument("--thetadot-max", type=float, default=8.0, help="Initial |theta_dot| max.")
    parser.add_argument("--control-dt", type=float, default=0.01, help="Controller update period used for transition label.")
    parser.add_argument("--torque-scale", type=float, default=1.0, help="Scale factor applied to sampled torque before sending to MuJoCo ctrl.")
    args = parser.parse_args()

    xml_path = Path(args.xml)
    if not xml_path.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    ctrl_low = float(model.actuator_ctrlrange[0, 0])
    ctrl_high = float(model.actuator_ctrlrange[0, 1])
    sim_dt = float(model.opt.timestep)

    n_substeps = max(1, int(round(args.control_dt / sim_dt)))
    effective_dt = n_substeps * sim_dt

    theta_upright_raw = np.pi
    rng = np.random.default_rng(args.seed)

    total_samples = args.episodes * args.steps_per_episode
    x = np.empty((total_samples, 2), dtype=np.float32)
    u = np.empty((total_samples, 1), dtype=np.float32)
    x_next = np.empty((total_samples, 2), dtype=np.float32)
    episode_id = np.empty((total_samples,), dtype=np.int32)
    step_id = np.empty((total_samples,), dtype=np.int32)

    k = 0

    for ep in range(args.episodes):
        theta0 = float(rng.uniform(-args.theta_max, args.theta_max))
        theta_dot0 = float(rng.uniform(-args.thetadot_max, args.thetadot_max))

        data.qpos[0] = theta_upright_raw + theta0
        data.qvel[0] = theta_dot0
        mujoco.mj_forward(model, data)

        for t in range(args.steps_per_episode):
            x_t = get_train_state(data, theta_upright_raw)

            u_cmd = sample_random_action(ctrl_low, ctrl_high, rng)
            u_applied = float(np.clip(u_cmd * args.torque_scale, ctrl_low, ctrl_high))

            for _ in range(n_substeps):
                data.ctrl[0] = u_applied
                mujoco.mj_step(model, data)

            x_tp1 = get_train_state(data, theta_upright_raw)

            x[k] = x_t
            u[k, 0] = u_applied
            x_next[k] = x_tp1
            episode_id[k] = ep
            step_id[k] = t
            k += 1

    if k != total_samples:
        x = x[:k]
        u = u[:k]
        x_next = x_next[:k]
        episode_id = episode_id[:k]
        step_id = step_id[:k]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x=x,
        u=u,
        x_next=x_next,
        episode_id=episode_id,
        step_id=step_id,
        control_dt=np.array([effective_dt], dtype=np.float32),
        sim_dt=np.array([sim_dt], dtype=np.float32),
    )

    print(f"Saved dataset: {out_path}")
    print(f"Samples: {x.shape[0]}")
    print(f"control_dt effective: {effective_dt:.6f}, sim_dt: {sim_dt:.6f}, substeps: {n_substeps}")


if __name__ == "__main__":
    main()
