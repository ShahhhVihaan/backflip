import time
import mujoco
import mujoco.viewer
import numpy as np

XML_PATH = "humanoid.xml"
CSV_PATH = "humanoid_backflip.csv"
Z_OFFSET = 0.0  # tweak if feet clip into ground

# Set this to the fps of the original motion (from PKL)
FPS = 16.0
FRAME_DURATION = 1.0 / FPS


def csv_frame_to_qpos(frame: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    """
    Convert one CSV row into a full qpos vector for this humanoid.

    CSV layout: [x, y, z, qx, qy, qz, qw, joint_0, joint_1, ...]
    MuJoCo qpos: [x, y, z, qw, qx, qy, qz, joint_0, joint_1, ...]
    """
    nq = model.nq
    if frame.shape[0] != nq:
        raise ValueError(
            f"CSV frame has {frame.shape[0]} values but model.nq={nq}. "
            "Make sure the CSV was generated for this humanoid.xml."
        )

    qpos = np.zeros(nq, dtype=np.float64)

    # Root position
    qpos[0:3] = frame[0:3]

    # Quaternion: CSV is [qx, qy, qz, qw] → MuJoCo expects [qw, qx, qy, qz]
    qx, qy, qz, qw = frame[3:7]
    qpos[3:7] = np.array([qw, qx, qy, qz], dtype=np.float64)

    # Remaining joints
    qpos[7:] = frame[7:]

    return qpos


def main():
    # Load CSV trajectory
    traj = np.loadtxt(CSV_PATH, delimiter=",")
    if traj.ndim == 1:
        traj = traj[None, :]  # handle single-line case

    num_frames, num_cols = traj.shape
    print(f"Loaded CSV: {num_frames} frames, {num_cols} columns")

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    print(f"Model nq: {model.nq}")

    if num_cols != model.nq:
        raise RuntimeError(
            f"CSV has {num_cols} columns but model.nq={model.nq}. "
            "Check pkl_to_csv joint layout vs humanoid.xml."
        )

    print(f"FPS: {FPS}")
    print(f"Motion length (seconds): {(num_frames - 1) / FPS:.3f}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Playing CSV motion...")

        while viewer.is_running():
            for i in range(num_frames):
                step_start = time.perf_counter()

                frame = traj[i]
                qpos = csv_frame_to_qpos(frame, model)

                # Optional vertical offset tweak
                qpos[2] += Z_OFFSET

                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                viewer.sync()

                process_time = time.perf_counter() - step_start
                sleep_time = FRAME_DURATION - process_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if not viewer.is_running():
                    break


if __name__ == "__main__":
    main()
