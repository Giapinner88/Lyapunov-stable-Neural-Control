import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    model_path = PROJECT_ROOT / "assets" / "cartpole.xml"
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Start near a visible pose so the shape is easy to inspect.
    data.qpos[0] = 0.0
    data.qpos[1] = np.pi + 0.15
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wide_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wide_fixed")
        if wide_cam_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = wide_cam_id

        print(f"Loaded model: {model_path}")
        print("Close the viewer window to exit.")

        while viewer.is_running():
            data.ctrl[0] = 0.0
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep))


if __name__ == "__main__":
    sys.exit(main())