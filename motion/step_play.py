import time
import pickle
import mujoco
import mujoco.viewer
import numpy as np
import sys

XML_PATH = "humanoid.xml"
PKL_PATH = "humanoid_backflip.pkl"
Z_OFFSET = 0  # might have to adjust if feet clip into ground

def rotvec_to_quat_mujoco(v):
    # Converts an axis-angle to a 4D quaternion
    angle = np.linalg.norm(v)
    if angle < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    axis = v / angle
    half_angle = angle / 2
    s = np.sin(half_angle)
    c = np.cos(half_angle)
    
    # Return [w, x, y, z]
    return np.array([c, s*axis[0], s*axis[1], s*axis[2]])

def main():
    print(f"Loading {PKL_PATH}...")
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)

    raw_frames = data['frames']
    if isinstance(raw_frames[0], dict):
        raw_traj = np.array([f['qpos'] for f in raw_frames])
    else:
        raw_traj = np.array(raw_frames)

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data_mj = mujoco.MjData(model)
    
    n_frames = raw_traj.shape[0]
    print(f"Loaded {n_frames} frames.")
    print("-" * 50)
    print("CONTROLS (Type in terminal):")
    print("  [Enter]    : Next Frame")
    print("  'b' + Enter: Previous Frame")
    print("  'r' + Enter: Reset to Frame 0")
    print("  'q' + Enter: Quit")
    print("-" * 50)

    with mujoco.viewer.launch_passive(model, data_mj) as viewer:
        i = 0
        
        while viewer.is_running():
            if i >= n_frames: i = n_frames - 1
            if i < 0: i = 0

            frame_data = raw_traj[i]
            
            if frame_data.shape[0] == 34 and model.nq == 35:
                root_pos = frame_data[0:3]
                root_rot_vec = frame_data[3:6]
                joints = frame_data[6:]
                quat = rotvec_to_quat_mujoco(root_rot_vec)
                current_pose = np.concatenate([root_pos, quat, joints])
            else:
                current_pose = frame_data.copy()

            current_pose[2] += Z_OFFSET
            data_mj.qpos[:] = current_pose

            if i == 0:
                print(f"first qpos: {data_mj.qpos}")
            
            mujoco.mj_forward(model, data_mj)
            viewer.sync()

            command = input(f"Frame {i}/{n_frames-1} >> ").strip().lower()
            # command = input(f"Frame {i}/{n_frames-1} | height: {data_mj.qpos[2]}>> ").strip().lower()

            if command == 'q':
                print("Quitting...")
                break
            elif command == 'b':
                i -= 1
            elif command == 'r':
                i = 0
            elif command.isdigit():
                target = int(command)
                if 0 <= target < n_frames:
                    i = target
                else:
                    print(f"Frame {target} out of range.")
            else:
                i += 1

if __name__ == "__main__":
    main()