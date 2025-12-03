import time
import pickle
import mujoco
import mujoco.viewer
import numpy as np

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
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)

    fps = data.get('fps', 30)
    print(f"fps: {fps}")
    frame_duration = 1.0 / fps
    print(f"Motion loaded. FPS: {fps} (Time per frame: {frame_duration:.4f}s)")

    print("FPS:", data["fps"])
    print("Num frames:", len(data["frames"]))
    print("Motion length (seconds):", (len(data["frames"]) - 1) / data["fps"])

    raw_frames = data['frames']
    if isinstance(raw_frames[0], dict):
        raw_traj = np.array([f['qpos'] for f in raw_frames])
    else:
        raw_traj = np.array(raw_frames)

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data_mj = mujoco.MjData(model)
    
    print(f"Model DoF: {model.nq} | Data DoF: {raw_traj.shape[1]}")

    with mujoco.viewer.launch_passive(model, data_mj) as viewer:
        print("Playing...")
        
        while viewer.is_running():
            for i in range(raw_traj.shape[0]):
                step_start = time.perf_counter()
                frame_data = raw_traj[i]
                
                if frame_data.shape[0] == 34 and model.nq == 35:
                    # Convert 34D to 35D
                    root_pos = frame_data[0:3]
                    root_rot_vec = frame_data[3:6]
                    joints = frame_data[6:]
                    
                    # Convert rotation
                    quat = rotvec_to_quat_mujoco(root_rot_vec)
                    
                    # Reassemble - Pos(3), Quat(4), Joints(28) = 35
                    current_pose = np.concatenate([root_pos, quat, joints])
                else:
                    current_pose = frame_data.copy()

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