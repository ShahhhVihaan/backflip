import time
import pickle
import mujoco
import mujoco.viewer
import numpy as np

XML_PATH = "humanoid.xml"
PKL_PATH_IN = "humanoid_backflip.pkl"
PKL_PATH_OUT = "humanoid_backflip_mirrored_legs.pkl"
Z_OFFSET = 0.0

# frames in flight phase
EDIT_FRAMES = [i for i in range(11, 29)]


def rotvec_to_quat_mujoco(v):
    angle = np.linalg.norm(v)
    if angle < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = v / angle
    half_angle = angle / 2.0
    s = np.sin(half_angle)
    c = np.cos(half_angle)
    return np.array([c, s * axis[0], s * axis[1], s * axis[2]])


def mirror_legs_in_frame(q34: np.ndarray) -> np.ndarray:
    """Mirror right leg joints onto left leg, in 34D PKL qpos format."""
    q = q34.copy()
    assert q.shape[0] == 34, f"Expected 34D qpos, got {q.shape[0]}"

    # Indices in the 34D vector
    RHX, RHY, RHZ = 20, 21, 22
    RKNEE = 23
    RAX, RAY, RAZ = 24, 25, 26

    LHX, LHY, LHZ = 27, 28, 29
    LKNEE = 30
    LAX, LAY, LAZ = 31, 32, 33

    # Right <- Left, mirrored in sagittal plane

    q[RHX] = -q[LHX]          # hip_x (pitch) same
    q[RHY] = q[LHY]         # hip_y (roll) flipped
    q[RHZ] = -q[LHZ]         # hip_z (yaw) flipped

    q[RKNEE] = q[LKNEE]      # knee (pitch) same

    q[RAX] = -q[LAX]          # ankle_x (pitch) same
    q[RAY] = q[LAY]         # ankle_y (roll) flipped
    q[RAZ] = -q[LAZ]         # ankle_z (yaw) flipped

    return q


def main():
    with open(PKL_PATH_IN, "rb") as f:
        data = pickle.load(f)

    fps = data.get("fps")
    frame_duration = 1.0 / fps
    raw_frames = data["frames"]

    print(f"FPS: {fps}")
    print("Num frames:", len(raw_frames))
    print("Motion length (seconds):", (len(raw_frames) - 1) / fps)

    if isinstance(raw_frames[0], dict):
        raw_traj = np.array([f["qpos"] for f in raw_frames])
        frames_are_dicts = True
    else:
        raw_traj = np.array(raw_frames)
        frames_are_dicts = False

    assert raw_traj.ndim == 2
    assert raw_traj.shape[1] in (34, 35), f"Unexpected qpos dim {raw_traj.shape[1]}"

    if raw_traj.shape[1] != 34:
        raise ValueError(
            f"Expected 34D PKL qpos (pos+rotvec+28 joints), got {raw_traj.shape[1]}"
        )

    for fi in EDIT_FRAMES:
        if 0 <= fi < raw_traj.shape[0]:
            print(f"Mirroring legs in frame {fi}")
            raw_traj[fi] = mirror_legs_in_frame(raw_traj[fi])
        else:
            print(f"WARNING: frame {fi} is out of range (0..{raw_traj.shape[0]-1})")

    if frames_are_dicts:
        for i, frame in enumerate(raw_frames):
            frame["qpos"] = raw_traj[i]
        data["frames"] = raw_frames
    else:
        data["frames"] = [q.copy() for q in raw_traj]

    with open(PKL_PATH_OUT, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved modified motion to {PKL_PATH_OUT}")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data_mj = mujoco.MjData(model)

    print(f"Model DoF: {model.nq} | Data DoF (PKL): {raw_traj.shape[1]}")

    with mujoco.viewer.launch_passive(model, data_mj) as viewer:
        print("Playing modified motion...")

        while viewer.is_running():
            for i in range(raw_traj.shape[0]):
                step_start = time.perf_counter()
                frame_data = raw_traj[i]

                root_pos = frame_data[0:3]
                root_rot_vec = frame_data[3:6]
                joints = frame_data[6:]
                quat = rotvec_to_quat_mujoco(root_rot_vec)
                current_pose = np.concatenate([root_pos, quat, joints])

                current_pose[2] += Z_OFFSET
                data_mj.qpos[:] = current_pose
                mujoco.mj_forward(model, data_mj)
                viewer.sync()

                process_time = time.perf_counter() - step_start
                sleep_time = frame_duration - process_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if not viewer.is_running():
                    break


if __name__ == "__main__":
    main()
