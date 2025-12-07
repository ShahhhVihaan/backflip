import pinocchio as pin
import casadi as cas
import numpy as np
from pinocchio import casadi as cpin
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from pathlib import Path
from pinocchio.robot_wrapper import RobotWrapper
import time
import os
import pickle
# Force Qt to use X11 (xcb) instead of Wayland
os.environ["QT_QPA_PLATFORM"] = "xcb"
import matplotlib.pyplot as plt

# --- 1. Helper: Setup CasADi functions from Pinocchio ---
def create_casadi_functions(model):
    """
    Creates CasADi symbolic functions for Pinocchio kinematics and dynamics.
    """
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    cq = cas.SX.sym("q", model.nq)
    cdq = cas.SX.sym("dq", model.nv)
    cv = cas.SX.sym("v", model.nv)
    
    # Forward Kinematics (updates cdata)
    cpin.framesForwardKinematics(cmodel, cdata, cq)
    cpin.computeCentroidalMomentum(cmodel, cdata, cq, cv)
    cpin.centerOfMass(cmodel, cdata, cq, cv)

    q_next = cpin.integrate(cmodel, cq, cdq)
    integrate_func = cas.Function("integrate", [cq, cdq], [q_next])

    # Function 1: Get Frame Position (Translation)
    # A function generator to get specific frame IDs later
    def get_corner_pos_func(frame_id, local_offset_np):
        frame_trans = cdata.oMf[frame_id].translation
        frame_rot = cdata.oMf[frame_id].rotation

        local_offset = cas.SX(local_offset_np)

        corner_pos = frame_trans + cas.mtimes(frame_rot, local_offset)
        
        safe_suffix = str(local_offset_np[0]).replace('.', '_').replace('-', 'm')
        func_name = f"pos_{frame_id}_off_{safe_suffix}"

        return cas.Function(func_name, [cq], [corner_pos])

    # Function 2: Get Centroidal Momentum (Linear & Angular)
    # Returns [hx, hy, hz, Lx, Ly, Lz] (Pinocchio standard)
    h_ag = cdata.hg
    centroidal_mom = cas.Function("centroidal_mom", [cq, cv], [h_ag.angular, h_ag.linear])
    
    # Function 3: Center of Mass
    com_pos = cas.Function("com_pos", [cq], [cdata.com[0]]) #[0] gives the whole com

    # Function 4: Frame Rotation Matrix (For Orientation Constraints)
    def get_frame_rot_func(frame_id):
        return cas.Function(f"rot_{frame_id}", [cq], [cdata.oMf[frame_id].rotation])

    return cmodel, cdata, get_corner_pos_func, get_frame_rot_func, centroidal_mom, com_pos, integrate_func


def solve_pinocchio():
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    urdf_path = str(project_root /"urdf/g1_description/g1_23dof.urdf")
    package_dir = str(Path(urdf_path).parent) # used to get meshes

    # Load Model
    try:
        robot = RobotWrapper.BuildFromURDF(urdf_path, [package_dir], root_joint=pin.JointModelFreeFlyer())
        model = robot.model
        data = robot.data # Pinocchio data
    except Exception as e:
        print(f"Error loading URDF: {e}")
        return
    
    # Initialize Visualizer (Standard Pinocchio Visualizer)
    try:
        robot.initViewer(loadModel=True)
        viz = robot.viz
    except ImportError:
        print("Meshcat not installed or failed to launch.")
        viz = None

    def set_q(q_vec, joint_name, val):
        """
        Finds the index of 'joint_name' in the q vector and sets it to 'val'. Helper function
        """
        if not model.existJointName(joint_name):
            print(f"WARNING: Joint '{joint_name}' not found in URDF.")
            return
        
        # 1. Get the Joint ID (e.g., 2, 3, ...)
        joint_id = model.getJointId(joint_name)
        
        # 2. Get the index in the configuration vector q
        # model.idx_qs gives the starting index for this joint
        # (It automatically accounts for the 7 floating base vars)
        q_idx = model.idx_qs[joint_id]
        
        # 3. Set the value
        q_vec[q_idx] = val

    # get 
    q_nom = pin.neutral(model)
    q_nom[2] = 0.78  # Height of pelvis for standing
    
    # --- Setup CasADi Optimization ---
    opti = cas.Opti()

    # Generate Symbolic Functions
    cmodel, cdata, make_corner_func, make_rot_func, f_mom, f_com, f_int= create_casadi_functions(model)

    # Robot Constants
    total_mass = pin.computeTotalMass(model)
    gravity = 9.81
    mu = 0.5  # Friction
    weight = total_mass * gravity

    corner_offsets = [
        np.array([-0.05, 0.025, -0.03]),
        np.array([-0.05, -0.025, -0.03]),
        np.array([0.12, 0.03, -0.03]),
        np.array([0.12, -0.03, -0.03])
    ]
    # Frame IDs (Update names based on your URDF)
    foot_names = ["left_ankle_roll_link", "right_ankle_roll_link"] 
    foot_ids = [model.getFrameId(n) for n in foot_names]
    body_id = model.getFrameId("pelvis")


    # 3. Create Functions for all 8 points (2 feet * 4 corners)
    # Structure: corner_funcs[foot_index][corner_index]
    corner_funcs = []
    for fid in foot_ids:
        funcs_for_this_foot = []
        for offset in corner_offsets:
            funcs_for_this_foot.append(make_corner_func(fid, offset))
        corner_funcs.append(funcs_for_this_foot)
    
    body_rot_func = make_rot_func(body_id)
    pin.framesForwardKinematics(model, data, q_nom)
    pin.updateFramePlacements(model, data)
    
    # Find height of the first corner of the first foot
    corner_0_pos = data.oMf[foot_ids[0]].translation + data.oMf[foot_ids[0]].rotation @ corner_offsets[0]
    height_error = corner_0_pos[2]
    print(f"Initial Guess Foot Height Error: {height_error:.4f}m. Adjusting...")
    
    # Shift base down/up by the error
    q_nom[2] -= height_error

    # height goal for crouch 
    z_crouch = 0.30
    #crouch pose guess
    q_crouch_pose = q_nom
    q_crouch_pose[2] = q_nom[2] - z_crouch

    try:
        set_q(q_crouch_pose, "left_knee_joint", 1.2)
        set_q(q_crouch_pose, "right_knee_joint", 1.2) 
        set_q(q_crouch_pose, "left_hip_pitch_joint", -0.6)
        set_q(q_crouch_pose, "right_hip_pitch_joint", -0.6)
        set_q("left_ankle_pitch_joint", -0.6)
        set_q("right_ankle_pitch_joint", -0.6)
    except:
        pass
    
    # get COM location 
    com_stand_pos = f_com(q_nom).full().flatten()
    
    # get position goals
    z_com_stand = com_stand_pos[2]
    z_com_crouch = z_com_stand - z_crouch
    min_com_height = z_crouch - 0.10
    z_com_apex = z_com_stand + 0.4

    delta_h = z_com_apex - z_com_stand
    
    if delta_h < 0:
        print("Error: Target apex is lower than standing height!")
        return

    v_z_required = np.sqrt(2 * 9.81 * delta_h)
    
    print(f"To reach {z_com_apex}m, Launch Velocity must be: {v_z_required:.3f} m/s")
    
    # Time, Knot points and conctact schedule 
    T_total = 1.2
    N = 80

    N_push   = 30         # Stand -> Crouch -> Launch
    N_flight = 20         # Air time
    N_land   = 30         # Impact -> Stand 

    assert (N_push + N_flight + N_land) == N
    
    k_liftoff   = N_push
    k_touchdown = N_push + N_flight

    contact_schedule = np.ones((2, N))
    contact_schedule[:, :k_liftoff]   = 1.0  # Phase 1: Push
    contact_schedule[:, k_liftoff:k_touchdown] = 0.0  # Phase 2: Flight
    contact_schedule[:, k_touchdown:] = 1.0  # Phase 3: Land

    # --- Decision Variables ---
    # Time step (variable)
    dt = opti.variable(N-1) 
    opti.subject_to(dt >= 0.5 * T_total / N)
    opti.subject_to(dt <= 2 * T_total / N)
    opti.subject_to(cas.sum1(dt) == T_total)

    fs = []
    for k in range(N-1):
        forces_at_k = []
        for f_idx in range(2): # 2 Feet
            foot_corner_forces = []
            for c_idx in range(4): # 4 Corners
                foot_corner_forces.append(opti.variable(3))
            forces_at_k.append(foot_corner_forces)
        fs.append(forces_at_k)

    # state space decision variable
    qs = [opti.variable(model.nq) for _ in range(N)]
    vs = [opti.variable(model.nv) for _ in range(N)]

    v_max = model.velocityLimit.copy() # get velocity limit from model
    v_max[:6] = 15.0
    
    k_apex = int(k_liftoff + (N_flight/2))

    min_x = com_stand_pos[0] - 0.10 
    max_x = com_stand_pos[0] + 0.10
    min_y = com_stand_pos[1] - 0.10
    max_y = com_stand_pos[1] + 0.10
    
    opti.subject_to(vs[0] == 0)
    #opti.subject_to(qs[0] == q_nom)
    #opti.subject_to(qs[-1][2] == q_nom[2])
    opti.subject_to(vs[-1] == 0)

    for k in range(N):
        # 1. Joint Limits
        opti.subject_to(opti.bounded(model.lowerPositionLimit, qs[k], model.upperPositionLimit))
        opti.subject_to(opti.bounded(-v_max, vs[k], v_max)) 

        # 2. Unit Quaternion Constraint (Pinocchio: [x,y,z, w])
        quat = qs[k][3:7] 
        opti.subject_to(cas.sumsqr(quat) == 1)

        qx = qs[k][3]
        qy = qs[k][4]
        qz = qs[k][5]
        
        opti.subject_to(qx == 0)
        #opti.subject_to(qy == 0)
        opti.subject_to(qz == 0)
        
        com_k = f_com(qs[k])
        
        opti.subject_to(com_k[2] >= min_com_height)
        is_stance_com = contact_schedule[0, k]
        if is_stance_com == 1:
            opti.subject_to(opti.bounded(min_x, com_k[0], max_x))
            opti.subject_to(opti.bounded(min_y, com_k[1], max_y))
        if k == k_apex:
            opti.subject_to(com_k[2] >= z_com_apex)
        if k == 0 or k == N:
            opti.subject_to(com_k[2] == z_com_stand)
            opti.subject_to(com_k[0] == com_stand_pos[0])
            opti.subject_to(com_k[1] == com_stand_pos[1])

        if k < N - 1:
            h = dt[k]
            
            #A. Integration (Backward Euler for stability)
            # q_{k+1} = integrate(q_k, v_{k+1} * h)
            q_next_integrated = f_int(qs[k], vs[k+1] * h)
            opti.subject_to(qs[k+1] == q_next_integrated)

            # Get current COM and Momentum
            ang_mom_k, lin_mom_k = f_mom(qs[k], vs[k])
            ang_mom_next, lin_mom_next = f_mom(qs[k+1], vs[k+1])
            

            # Finite Difference Derivatives
            lin_force_total = cas.vertcat(0,0,0)
            ang_torque_total = cas.vertcat(0,0,0)

            for f_idx in range(2):
                is_stance = contact_schedule[f_idx, k]

                for c_idx in range(4):
                    f_corner = fs[k][f_idx][c_idx]
                    p_corner_world = corner_funcs[f_idx][c_idx](qs[k])
                    # Accumulate for dynamics
                    lin_force_total += f_corner
                    # Angular momentum contribution: cross(p_foot - com, f)
                    ang_torque_total += cas.cross(p_corner_world - com_k, f_corner)
                    
                    if is_stance == 1:
                    # 2. Friction Cone
                    # applied individually to each corner
                        opti.subject_to(f_corner[2] >= 0) 
                        opti.subject_to(f_corner[0] <= mu * f_corner[2])
                        opti.subject_to(-f_corner[0] <= mu * f_corner[2])
                        opti.subject_to(f_corner[1] <= mu * f_corner[2])
                        opti.subject_to(-f_corner[1] <= mu * f_corner[2])

                        #Kinematics Constraint for foot
                        p_next = corner_funcs[f_idx][c_idx](qs[k+1])
                        opti.subject_to(p_corner_world == p_next)
                        opti.subject_to(p_corner_world[2] == 0) # On ground
                    else:
                        opti.subject_to(f_corner == 0)
                        opti.subject_to(p_corner_world[2] >= 0.0)

            # Linear Dynamics Constraint
            # mv_next - mv_curr = h * (forces + mg)
            # Note: Pinocchio Linear Momentum is mass * velocity
            delta_lin_mom = lin_mom_next - lin_mom_k
            gravity_vec = np.array([0, 0, -9.81 * total_mass])
            opti.subject_to(delta_lin_mom == h * (lin_force_total + gravity_vec))

            # Angular Dynamics Constraint
            delta_ang_mom = ang_mom_next - ang_mom_k
            opti.subject_to(delta_ang_mom == h * ang_torque_total)

            opti.subject_to(opti.bounded(-1.0, ang_mom_k[1], 1.0))
                

    cost = 0 
    # Cost for base location 
    w_com_pos_x = 15.0   
    w_com_pos_y = 30.0
    w_com_pos_z = 50.0 

    # cost matrix for joints
    n_actuated = model.nq - 7
    w_joints_vector = np.ones(n_actuated) * 1.0 # Default weight
    for joint_id in range(2, model.njoints):
        name = model.names[joint_id]
        idx_in_cost_vector = model.idx_qs[joint_id] - 7
        if "ankle_pitch_joint" in name.lower():
            w_joints_vector[idx_in_cost_vector] = 0.2
        elif "knee_joint" in name.lower():
            w_joints_vector[idx_in_cost_vector] = 0.01
        elif "hip_pitch" in name.lower():
            w_joints_vector[idx_in_cost_vector] = 1
        else:
            w_joints_vector[idx_in_cost_vector] = 30.0
    # cost for weights
    W_joints = cas.MX(w_joints_vector)        
    w_base_rot = 30.0 
    w_force = 1e-3
    w_force_rate = 1e-4
    # Minimize joint velocities (smoothness)
    for k in range(N):
        cost += 1e-5 * cas.sumsqr(vs[k])    

    # Minimize forces (efficiency)
    for k in range(N-1):
        for f_idx in range(2):      # Loop over 2 feet
            for c_idx in range(4):  # Loop over 4 corners
                # access the specific MX variable
                f_corner_current = fs[k][f_idx][c_idx] 
                cost += w_force * cas.sumsqr(f_corner_current)
                


    for k in range(N):
        
        if k < N - 1:
            acc = (vs[k+1] - vs[k]) / (dt[k] + 1e-5)
            w_acc = 0.01 
            cost += w_acc * cas.sumsqr(acc)

        target_com_z = z_com_stand
        q_target_joints = q_nom[7:]
        current_w_z = w_com_pos_z
        k_bottom = k_liftoff / 2.0
        if k<k_liftoff:  
            if k < k_bottom:
                progress = k / k_bottom
                factor = (1.0 - np.cos(progress * np.pi)) / 2.0
                target_com_z = z_com_stand + (z_com_crouch - z_com_stand) * factor
            else:
                progress = (k - k_bottom) / (k_liftoff - k_bottom)
                factor = (1.0 - np.cos(progress * np.pi)) / 2.0
                target_com_z = z_com_crouch + (z_com_stand - z_com_crouch) * factor
        elif k < k_touchdown:
            current_w_z = 0.0  
            target_com_z = 0.0
        else:
            target_com_z = z_com_stand

        cost += w_com_pos_x * cas.sumsqr(qs[k][0] - q_nom[0]) # Keep base near origin
        cost += w_com_pos_y * cas.sumsqr(qs[k][1] - q_nom[1]) # Keep base near origin
        cost += current_w_z * cas.sumsqr(qs[k][2] - target_com_z) # Keep base near target height
        q_diff = qs[k][7:] - q_target_joints
        cost += cas.sum1(W_joints * q_diff**2)
        cost += w_base_rot * cas.sumsqr(qs[k][3:7] - q_nom[3:7])

    opti.minimize(cost)

    # --- Solve ---
    opts = {
        "ipopt.print_level": 5, 
        "ipopt.max_iter": 300, 
        "ipopt.tol": 1e-3  # changed for now
    }
    opti.solver("ipopt", opts)
    k_bottom = int(k_liftoff/2)
    # Initial Guess
    z_stand = q_nom[2]
    z_apex = q_nom[2] + delta_h
    for k in range(N):
        if k < k_bottom:
            progress = k / k_bottom
            z_val = z_stand + (z_crouch - z_stand) * np.sin(progress * np.pi / 2)
            # Create a guess that moves linearly
            q_k = pin.interpolate(model, q_nom, q_crouch_pose, progress)
        elif k < k_liftoff:
            progress = (k - k_bottom) / (k_liftoff - k_bottom)
            q_k = pin.interpolate(model, q_crouch_pose, q_nom, progress)
            z_val = z_crouch + (z_stand - z_crouch) * progress
        elif k < k_touchdown:
            progress = (k - k_liftoff) / N_flight
            z_val = z_crouch + (z_apex - z_crouch) * 4 * progress * (1 - progress)
            q_k = q_crouch_pose.copy()
        else: 
            progress = (k - k_touchdown) / (N - k_touchdown)
            z_val = z_crouch + (z_stand - z_crouch) * progress
            q_k = pin.interpolate(model, q_crouch_pose, q_nom, progress)

        q_k[2] = z_val
        opti.set_initial(qs[k], q_k)

        if k < N-1:
            if k_bottom < k < k_liftoff:
                 # Guess upward velocity during thrust
                 opti.set_initial(vs[k], np.zeros(model.nv)) 
                 opti.set_initial(vs[k][2], 2.0) # Guess 2 m/s upwards
            else:
                 opti.set_initial(vs[k], np.zeros(model.nv))
        
    for k in range(N-1):    
        opti.set_initial(dt[k], T_total/N)
    
    for k in range(N-1):
        for f_idx in range(2):
            for c_idx in range(4):
                # Guess [0, 0, weight/8]
                if k < k_liftoff:
                    #push off
                    opti.set_initial(fs[k][f_idx][c_idx], [0, 0, weight/4])
                elif k < k_touchdown:
                    #flight phase
                    opti.set_initial(fs[k][f_idx][c_idx], np.array([0, 0, 0]))
                else:
                    # Impact absorption
                    opti.set_initial(fs[k][f_idx][c_idx], [0, 0, weight/4])

    
    try:
        sol = opti.solve()
    except:
        print("Solver failed, retrieving debug values...")
        sol = opti.debug
    
    # --- Extract & Animate ---
    q_sol = []
    v_sol = np.zeros((N, model.nv))
    for k in range(N):
        q_sol.append(sol.value(qs[k]))

    for k in range(N):
        v_sol[k, :] = sol.value(vs[k])

    # --- 2. COMPUTE WORLD FRAME VELOCITY  ---
    v_world_sol = np.zeros((N, 3))
    
    for k in range(N):
        # Get orientation from q_sol
        q_k = q_sol[k]
        # Pinocchio quaternion is [x,y,z,w]
        quat = pin.Quaternion(q_k[6], q_k[3], q_k[4], q_k[5]) 
        R_body = quat.matrix()
        
        # v_local is the first 3 elements of v
        v_local = v_sol[k, :3]
        
        # Rotate to World Frame
        v_world_sol[k, :] = R_body @ v_local

    # --- 3. PLOT ---

    # Create a directory for results if it doesn't exist
    results_dir = Path(project_root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {results_dir}")

    dt_val = sol.value(dt)

    t_array_full = np.zeros(N)
    t_array_full[1:] = np.cumsum(dt_val)

    plt.figure(figsize=(10, 6))
    
    # Plot World Z Velocity (Vertical Jump Speed)
    plt.plot(t_array_full, v_world_sol[:, 2], label='World Z (Vertical)', color='blue', linewidth=2)
    
    # Plot World X Velocity (Forward Speed)
    plt.plot(t_array_full, v_world_sol[:, 0], label='World X (Forward)', color='green', linestyle='--')
    

    plt.title("Base Translation Velocity (World Frame)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid(True)    
    plt.savefig(results_dir / "velocity_profile.png", dpi=300)

    force_data = np.zeros((N-1, 2, 4, 3)) 
    
    for k in range(N-1):
        for f_idx in range(2):      # Left, Right
            for c_idx in range(4):  # 4 Corners
                # Extract value from CasADi variable
                val = sol.value(fs[k][f_idx][c_idx])
                force_data[k, f_idx, c_idx, :] = val

    # Create Time Vector for plotting
    t_array_forces = t_array_full[:-1]
    
    # Plot 2: Total Vertical Force (Z) per Foot
    # Summing the 4 corners to see total weight on each leg
    total_force_left = np.sum(force_data[:, 0, :, 2], axis=1)
    total_force_right = np.sum(force_data[:, 1, :, 2], axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(t_array_forces, total_force_left, label='Left Foot Total Fz', color='blue', linewidth=5)
    plt.plot(t_array_forces, total_force_right, label='Right Foot Total Fz', color='red', linewidth=2)
    
    # Reference line for Robot Weight
    weight = 9.81 * pin.computeTotalMass(model)
    plt.axhline(y=weight, color='k', linestyle='--', label='Total Robot Weight')
    plt.axhline(y=weight/2, color='gray', linestyle=':', label='Half Weight')
    
    plt.title("Total Vertical Reaction Force per Foot")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / "total_forces.png", dpi=300)

    # Plot 3: Individual Corner Forces (Left Foot Only)
    # This helps debug if the foot is "tipping" (e.g. heel lifting off)
    plt.figure(figsize=(10, 6))
    colors = ['r', 'g', 'b', 'orange']
    labels = ['Front-Left', 'Front-Right', 'Back-Left', 'Back-Right']
    
    for c_idx in range(4):
        # Plotting Z-component only
        forces = force_data[:, 0, c_idx, 2] 
        plt.plot(t_array_forces, forces, label=f'Left Foot - {labels[c_idx]}', color=colors[c_idx])

    plt.title("Left Foot - Individual Corner Forces (Z)")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / "corner_forces.png", dpi=300)


    
    
    data_to_save = {
        "t": t_array_full,
        "q": np.array(q_sol),       # Shape: (N, nq)
        "v": v_sol,                 # Shape: (N, nv)
        "dt": dt_val,               # Shape: (N-1,)
        "f": force_data,            # Shape: (N-1, 2, 4, 3)
        "mass": total_mass,         # Helpful metadata
        "T_total": T_total
    }

    pkl_filename = "trajectory_solution.pkl"
    if 'results_dir' in locals():
        save_path = results_dir / pkl_filename
    else:
        save_path = Path(pkl_filename)

    with open(save_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print(f"Decision variables saved to: {save_path.absolute()}")

    # Simple Meshcat Animation
    
    fps = 30.0             # Target Frames Per Second
    playback_speed = 0.1 # How many seconds to spend moving between step k and k+1
        
    while True:
        for k in range(N - 1):
            q_start = q_sol[k]
            q_end = q_sol[k+1]
            dt_k = dt_val[k]
            # Calculate how many sub-frames we need for this step
            duration_for_step = dt_k / playback_speed
            n_substeps = int(duration_for_step * fps)
            
            for i in range(n_substeps):
                alpha = i / n_substeps # Goes from 0.0 to 1.0
                
                # Pinocchio handles quaternion interpolation (SLERP) automatically
                q_interp = pin.interpolate(model, q_start, q_end, alpha)
                
                viz.display(q_interp)
                time.sleep(1.0 / fps)
        
        time.sleep(2) # Pause before restarting
        
        
    return q_sol

if __name__ == "__main__":
    solve_pinocchio()
    print("\nScript finished. Keeping visualizer alive...")
    print("Press Ctrl+C to close.")
    while True:
        time.sleep(1.0) # Keep the process running