import pickle
import mujoco
import numpy as np

# --- SETTINGS ---
XML_PATH = "humanoid.xml"
PKL_PATH = "humanoid_backflip.pkl"

CONTACT_THRESHOLD = 0.005 

def rotvec_to_quat_mujoco(v):
    angle = np.linalg.norm(v)
    if angle < 1e-6: return np.array([1., 0., 0., 0.])
    axis = v / angle
    half_angle = angle / 2
    s, c = np.sin(half_angle), np.cos(half_angle)
    return np.array([c, s*axis[0], s*axis[1], s*axis[2]])

def get_foot_contacts(model, data, foot_body_id):
    points = []
    
    for i in range(data.ncon):
        contact = data.contact[i]
        
        # Check if the contact is "active" (actually touching or penetrating)
        if contact.dist > CONTACT_THRESHOLD:
            continue

        # Get the IDs of the two geometries involved in the collision
        g1 = contact.geom1
        g2 = contact.geom2
        
        # Map Geometry IDs to Body IDs
        b1 = model.geom_bodyid[g1]
        b2 = model.geom_bodyid[g2]
        
        # Check if the foot is one of the bodies involved
        if b1 == foot_body_id or b2 == foot_body_id:
            points.append(contact.pos.copy())
            
    return np.array(points)

def main():
    print(f"Loading {PKL_PATH}...")
    with open(PKL_PATH, 'rb') as f:
        traj_data = pickle.load(f)

    fps = traj_data.get('fps', 30)
    raw_frames = traj_data['frames']
    
    if isinstance(raw_frames[0], dict):
        qpos_traj = np.array([f['qpos'] for f in raw_frames])
    else:
        qpos_traj = np.array(raw_frames)

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    try:
        r_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
        l_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
    except ValueError:
        print("Error: Could not find bodies named 'right_foot' or 'left_foot' in XML.")
        return

    n_frames = qpos_traj.shape[0]
    print(f"Processing {n_frames} frames...")
    print("-" * 60)
    print(f"{'Time (s)':<10} | {'Right Foot Contacts':<25} | {'Left Foot Contacts':<25}")
    print("-" * 60)

    for i in range(n_frames):
        frame_data = qpos_traj[i]
        
        if frame_data.shape[0] == 34 and model.nq == 35:
            root_pos = frame_data[0:3]
            root_rot = rotvec_to_quat_mujoco(frame_data[3:6])
            joints = frame_data[6:]
            current_pose = np.concatenate([root_pos, root_rot, joints])
        else:
            current_pose = frame_data

        data.qpos[:] = current_pose
        
        mujoco.mj_forward(model, data)

        # Extract Contacts
        r_points = get_foot_contacts(model, data, r_foot_id)
        l_points = get_foot_contacts(model, data, l_foot_id)

        time_sec = i / fps
        
        if len(r_points) > 0:
            r_str = f"{len(r_points)} pts (Z~{np.mean(r_points[:,2]):.3f})"
        else:
            r_str = "Air"

        if len(l_points) > 0:
            l_str = f"{len(l_points)} pts (Z~{np.mean(l_points[:,2]):.3f})"
        else:
            l_str = "Air"

        print(f"{time_sec:<10.2f} | {r_str:<25} | {l_str:<25}")

        # to see the specific XYZ of the first contact point
        if len(r_points) > 0:
            # print(f"   -> R-Contact 0: {np.round(r_points[0], 3)}")
            pass

if __name__ == "__main__":
    main()