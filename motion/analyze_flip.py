import pickle
import mujoco
import numpy as np

XML_PATH = "humanoid.xml"
PKL_PATH = "humanoid_backflip.pkl"
FLIGHT_THRESHOLD = 0.05 

def rotvec_to_quat_mujoco(v):
    angle = np.linalg.norm(v)
    if angle < 1e-6: return np.array([1., 0., 0., 0.])
    axis = v / angle
    half_angle = angle / 2
    s, c = np.sin(half_angle), np.cos(half_angle)
    return np.array([c, s*axis[0], s*axis[1], s*axis[2]])

def main():
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)

    fps = data.get('fps', 30)
    raw_frames = data['frames']
    
    if isinstance(raw_frames[0], dict):
        qpos_traj = np.array([f['qpos'] for f in raw_frames])
    else:
        qpos_traj = np.array(raw_frames)

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data_mj = mujoco.MjData(model)

    right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
    left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")

    n_frames = qpos_traj.shape[0]
    is_flying = False
    takeoff_frame = None
    landing_frame = None

    print(f"Analyzing {n_frames} frames at {fps} FPS")

    for i in range(n_frames):
        frame_data = qpos_traj[i]
        
        if frame_data.shape[0] == 34 and model.nq == 35:
            root_pos = frame_data[0:3]
            root_rot = rotvec_to_quat_mujoco(frame_data[3:6])
            joints = frame_data[6:]
            current_pose = np.concatenate([root_pos, root_rot, joints])
        else:
            current_pose = frame_data

        data_mj.qpos[:] = current_pose
        
        mujoco.mj_forward(model, data_mj)

        r_foot_z = data_mj.xpos[right_foot_id][2]
        l_foot_z = data_mj.xpos[left_foot_id][2]
        
        min_foot_height = min(r_foot_z, l_foot_z)

        if not is_flying and min_foot_height > FLIGHT_THRESHOLD:
            is_flying = True
            if takeoff_frame is None:
                takeoff_frame = i
        
        elif is_flying and min_foot_height <= FLIGHT_THRESHOLD:
            is_flying = False
            if landing_frame is None:
                landing_frame = i

    total_time = n_frames / fps
    
    if takeoff_frame is not None:
        stand_time = takeoff_frame / fps
        
        if landing_frame is not None:
            flight_time = (landing_frame - takeoff_frame) / fps
        else:
            flight_time = (n_frames - takeoff_frame) / fps # Never landed
            landing_frame = n_frames
            
    else:
        stand_time = total_time
        flight_time = 0.0

    print(f"ANALYSIS REPORT")
    print(f"Stand Phase Time:   {stand_time:.4f} s (Frames 0 to {takeoff_frame})")
    print(f"Flight Phase Time:  {flight_time:.4f} s (Frames {takeoff_frame} to {landing_frame})")
    print(f"Total Flip Time:    {total_time:.4f} s")
    
    print("\n CONFIGURATIONS (Rotations fixed to Quaternions)")
    
    # We re-calculate to ensure we get the quaternion version
    data_mj.qpos[:] = qpos_traj[0] if qpos_traj[0].shape[0] == 35 else np.concatenate([qpos_traj[0][0:3], rotvec_to_quat_mujoco(qpos_traj[0][3:6]), qpos_traj[0][6:]])
    start_config = data_mj.qpos.copy()
    
    data_mj.qpos[:] = qpos_traj[-1] if qpos_traj[-1].shape[0] == 35 else np.concatenate([qpos_traj[-1][0:3], rotvec_to_quat_mujoco(qpos_traj[-1][3:6]), qpos_traj[-1][6:]])
    end_config = data_mj.qpos.copy()

    print(f"Start Config (Shape {start_config.shape}):\n{np.array2string(start_config, precision=3, separator=', ')}")
    print(f"\n End Config (Shape {end_config.shape}):\n{np.array2string(end_config, precision=3, separator=', ')}")

    np.save("config_start.npy", start_config)
    np.save("config_end.npy", end_config)

if __name__ == "__main__":
    main()