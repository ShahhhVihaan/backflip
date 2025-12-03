#!/usr/bin/env python3
import pickle
import numpy as np

PKL_PATH_IN = "humanoid_backflip.pkl"
OUTPUT_PATH = "humanoid_keyframes.xml"   # output file with <keyframe>...</keyframe>


def rotvec_to_quat_mujoco(v: np.ndarray) -> np.ndarray:
    """Convert axis-angle (rotvec) to MuJoCo-style quat [w, x, y, z]."""
    angle = np.linalg.norm(v)
    if angle < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = v / angle
    half_angle = angle / 2.0
    s = np.sin(half_angle)
    c = np.cos(half_angle)
    return np.array([c, s * axis[0], s * axis[1], s * axis[2]])


def main():
    # --- Load PKL ---
    with open(PKL_PATH_IN, "rb") as f:
        data = pickle.load(f)

    raw_frames = data["frames"]

    # Extract qpos array
    if isinstance(raw_frames[0], dict):
        raw_traj = np.array([f["qpos"] for f in raw_frames])
    else:
        raw_traj = np.array(raw_frames)

    assert raw_traj.ndim == 2
    num_frames, dim = raw_traj.shape
    print(f"Loaded {num_frames} frames, dim={dim}")

    if dim not in (34, 35):
        raise ValueError(f"Expected qpos dim 34 or 35, got {dim}")

    lines = []
    lines.append("<keyframe>")

    for i in range(num_frames):
        q = raw_traj[i]

        if dim == 34:
            # PKL format: [0:3] root pos, [3:6] root rotvec, [6:] joints
            root_pos = q[0:3]
            root_rotvec = q[3:6]
            joints = q[6:]
            quat = rotvec_to_quat_mujoco(root_rotvec)
            q_mj = np.concatenate([root_pos, quat, joints])
        else:
            # Already full MuJoCo qpos
            q_mj = q

        # Format like your example: one long qpos string
        qpos_str = " ".join(f"{x:.8g}" for x in q_mj)
        key_name = f"frame{i:02d}"  # frame00, frame01, ...

        lines.append(f'  <key name="{key_name}" qpos="{qpos_str}"/>')

    lines.append("</keyframe>")

    xml_block = "\n".join(lines)

    with open(OUTPUT_PATH, "w") as f:
        f.write(xml_block)

    print(f"\nWrote keyframe block with {num_frames} keys to {OUTPUT_PATH}")
    print("Copy-paste that whole <keyframe>...</keyframe> into humanoid.xml.")


if __name__ == "__main__":
    main()
