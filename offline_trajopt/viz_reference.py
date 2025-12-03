import sys
import time
import pickle
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

MJCF_PATH = "urdf/humanoid/humanoid.xml"
PKL_PATH = "motion/humanoid_backflip.pkl"
Z_OFFSET = 0.0 

def rotvec_to_quat_pinocchio(v):
    rot_matrix = pin.exp3(v)
    
    quat = pin.Quaternion(rot_matrix)

    return np.array([quat.x, quat.y, quat.z, quat.w])

def main():
    model = pin.buildModelFromMJCF(MJCF_PATH)
    data = model.createData()

    try:
        visual_model = pin.buildGeomFromMJCF(model, MJCF_PATH, pin.GeometryType.VISUAL)
        collision_model = pin.buildGeomFromMJCF(model, MJCF_PATH, pin.GeometryType.COLLISION)
    except AttributeError:
        print("Error: Pinocchio version usually needs to be v2.6.9+ or v3.x for MJCF geom.")
        sys.exit(1)

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=False) 
    viz.loadViewerModel()
    
    print(f"Loading motion from {PKL_PATH}...")
    with open(PKL_PATH, 'rb') as f:
        motion_data = pickle.load(f)

    fps = motion_data.get('fps', 30)
    frame_duration = 1.0 / fps
    
    raw_frames = motion_data['frames']
    if len(raw_frames) > 0 and isinstance(raw_frames[0], dict):
        raw_traj = np.array([f['qpos'] for f in raw_frames])
    else:
        raw_traj = np.array(raw_frames)

    print(f"Model nq: {model.nq} | Trajectory Data Dim: {raw_traj.shape[1]}")
    time.sleep(3)

    while True:
        for i in range(raw_traj.shape[0]):
            step_start = time.perf_counter()
            
            frame_data = raw_traj[i]
            
            # If data is 34D (pos+rotvec+joints) and model is 35D (pos+quat+joints)
            if frame_data.shape[0] == 34 and model.nq == 35:
                root_pos = frame_data[0:3]
                root_rot_vec = frame_data[3:6]
                joints = frame_data[6:]
                
                # Convert to [x, y, z, w]
                quat = rotvec_to_quat_pinocchio(root_rot_vec)
                root_pos[2] += Z_OFFSET
                q = np.concatenate([root_pos, quat, joints])
            else:
                q = frame_data.copy()
                q[2] += Z_OFFSET
            viz.display(q)

            process_time = time.perf_counter() - step_start
            sleep_time = frame_duration - process_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("Replaying...")
        time.sleep(0.5)

if __name__ == "__main__":
    main()