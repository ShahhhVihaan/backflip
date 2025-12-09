#!/usr/bin/env python3
import os
import time
import pickle
from pathlib import Path

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import matplotlib.pyplot as plt

# Force Qt to use X11 (xcb) instead of Wayland, same as your main script
os.environ["QT_QPA_PLATFORM"] = "xcb"


# ------------------------------------------------------------
#  Torque limits per joint (from your MJCF actuatorfrcrange)
#  We only care about actuated 1-DoF joints here.
# ------------------------------------------------------------

JOINT_TAU_LIMITS = {
    # Left leg
    "left_hip_pitch_joint": 88.0,
    "left_hip_roll_joint": 88.0,
    "left_hip_yaw_joint": 88.0,
    "left_knee_joint": 139.0,
    "left_ankle_pitch_joint": 50.0,
    "left_ankle_roll_joint": 50.0,
    # Right leg
    "right_hip_pitch_joint": 88.0,
    "right_hip_roll_joint": 88.0,
    "right_hip_yaw_joint": 88.0,
    "right_knee_joint": 139.0,
    "right_ankle_pitch_joint": 50.0,
    "right_ankle_roll_joint": 50.0,
    # Waist
    "waist_yaw_joint": 88.0,
    # Left arm
    "left_shoulder_pitch_joint": 25.0,
    "left_shoulder_roll_joint": 25.0,
    "left_shoulder_yaw_joint": 25.0,
    "left_elbow_joint": 25.0,
    "left_wrist_roll_joint": 25.0,
    # Right arm
    "right_shoulder_pitch_joint": 25.0,
    "right_shoulder_roll_joint": 25.0,
    "right_shoulder_yaw_joint": 25.0,
    "right_elbow_joint": 25.0,
    "right_wrist_roll_joint": 25.0,
}


def load_robot():
    """
    Load the G1 robot with a free-flyer base, same as in solve_pinocchio().
    """
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    urdf_path = project_root / "urdf/g1_description/g1_23dof.urdf"
    package_dir = urdf_path.parent

    robot = RobotWrapper.BuildFromURDF(
        str(urdf_path),
        [str(package_dir)],
        root_joint=pin.JointModelFreeFlyer(),
    )
    robot.initViewer(loadModel=True)
    return robot


def load_trajectory():
    """
    Load the trajectory_solution.pkl you saved in solve_pinocchio().
    """
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    results_dir = project_root / "results"

    pkl_path = results_dir / "trajectory_solution_flip.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Could not find {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Expecting the dict structure you saved:
    # "t": (N,), "q": (N, nq), "v": (N, nv), "dt": (N-1,), "f": (N-1, 2, 4, 3), etc.
    t = np.asarray(data["t"])
    q = np.asarray(data["q"])
    v = np.asarray(data["v"])
    dt = np.asarray(data["dt"])
    f = np.asarray(data["f"])

    return t, q, v, dt, f, data


def compute_inverse_dynamics(robot, t, q, v, dt, f):
    """
    Compute whole-body inverse dynamics torques tau[k] along the trajectory.

    tau has shape (N-1, nv) because we use finite-difference acceleration
    a_k = (v[k+1] - v[k]) / dt[k].
    """
    model = robot.model
    data = model.createData()

    N = q.shape[0]
    nv = model.nv

    # Same corner offsets and foot frame names as in the optimizer
    corner_offsets = [
        np.array([-0.05,  0.025, -0.03]),
        np.array([-0.05, -0.025, -0.03]),
        np.array([ 0.12,  0.03,  -0.03]),
        np.array([ 0.12, -0.03,  -0.03]),
    ]
    foot_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    foot_ids = [model.getFrameId(n) for n in foot_names]
    # Parent joint for each foot frame
    foot_joint_ids = [model.frames[fid].parentJoint for fid in foot_ids]

    tau_all = np.zeros((N - 1, nv))

    for k in range(N - 1):
        qk = q[k]
        vk = v[k]
        vk1 = v[k + 1]
        h = float(dt[k])

        if h <= 0.0:
            # Fallback: zero accel if dt is degenerate
            ak = np.zeros_like(vk)
        else:
            ak = (vk1 - vk) / h

        # Forward kinematics for this configuration
        pin.forwardKinematics(model, data, qk, vk, ak)
        pin.updateFramePlacements(model, data)

        # External forces: one spatial wrench per joint (in LOCAL joint frame)
        fext = [pin.Force.Zero() for _ in range(model.njoints)]

        # f[k, foot_idx, corner_idx, :] is the 3D force at each corner in WORLD frame
        for foot_idx, (frame_id, joint_id) in enumerate(zip(foot_ids, foot_joint_ids)):
            # World pose of the foot frame
            oMf = data.oMf[frame_id]
            p_frame = oMf.translation       # world position of foot frame origin
            Rwf = oMf.rotation              # world-to-foot rotation matrix (world R frame)

            # Accumulate net force + moment at the frame origin, in WORLD coords
            f_total_world = np.zeros(3)
            tau_total_world = np.zeros(3)

            for c_idx, offset in enumerate(corner_offsets):
                f_corner = f[k, foot_idx, c_idx, :]  # 3D force in WORLD
                # Corner position in WORLD
                p_corner = p_frame + Rwf @ offset

                f_total_world += f_corner
                tau_total_world += np.cross(p_corner - p_frame, f_corner)

            # Express this wrench in the LOCAL frame of the foot
            # Force = (linear, angular) in Pinocchio.
            f_local_linear = Rwf.T @ f_total_world
            f_local_angular = Rwf.T @ tau_total_world

            fext[joint_id] = pin.Force(f_local_linear, f_local_angular)

        # Inverse dynamics with external forces
        tau = pin.rnea(model, data, qk, vk, ak, fext)
        tau_all[k, :] = tau

    return tau_all


# ------------------------------------------------------------
#  Helpers to check torque limits in tau_all
# ------------------------------------------------------------

def collect_joint_dof_indices(model):
    """
    For each 1-DoF actuated joint in JOINT_TAU_LIMITS, find its DOF index in v/tau.

    Returns:
        joint_to_idx : dict { joint_name -> dof_index }
    """
    joint_to_idx = {}

    for j_id in range(model.njoints):
        name = model.names[j_id]
        if name in JOINT_TAU_LIMITS:
            # idx_vs gives starting index of this joint's motion subspace in v/tau
            idx_v = model.idx_vs[j_id]
            nj = model.joints[j_id].nv
            if nj != 1:
                print(f"WARNING: joint {name} has nv={nj}, skipping for limit check")
                continue
            joint_to_idx[name] = idx_v

    return joint_to_idx


def check_torque_limits(robot, tau_all):
    """
    Compare |tau| against the per-joint torque limits and print a small report.
    """
    model = robot.model
    joint_to_idx = collect_joint_dof_indices(model)

    print("\n=== Torque limit check (inverse dynamics) ===")
    print(f"tau_all shape: {tau_all.shape}  (N-1, nv={model.nv})")
    print(f"Checking {len(joint_to_idx)} joints with limits:\n")

    header = "{:<30s} {:>10s} {:>10s} {:>10s} {:>10s}"
    row    = "{:<30s} {:>10.2f} {:>10.2f} {:>10.2f} {:>10s}"
    print(header.format("joint", "max|τ|", "limit", "ratio", "violated"))

    any_violated = False

    for j_name, limit in JOINT_TAU_LIMITS.items():
        if j_name not in joint_to_idx:
            print(f"(skip) joint {j_name} not found in model or not 1-DoF")
            continue

        idx = joint_to_idx[j_name]   # DOF index in tau
        tau_j = tau_all[:, idx]      # (N-1,)
        max_abs = float(np.max(np.abs(tau_j)))
        ratio = max_abs / limit if limit > 0 else np.inf
        violated = ratio > 1.0 + 1e-3

        if violated:
            any_violated = True

        print(row.format(j_name, max_abs, limit, ratio, "YES" if violated else "no"))

    if not any_violated:
        print("\nNo torque limits violated (within the checked joints).")
    else:
        print("\nSome joints exceed their torque limits. See 'YES' rows above.")


# ------------------------------------------------------------
#  Simple visualization (unchanged)
# ------------------------------------------------------------

def play_trajectory(robot, t, q, fps=60.0, loop=True):
    """
    Visualize the trajectory using constant FPS, but respecting the fact
    that the original dt are *not* evenly spaced.
    """
    model = robot.model
    viz = robot.viz

    N = len(t)
    if N < 2:
        raise ValueError("Need at least 2 states in the trajectory to animate.")

    T_total = float(t[-1])
    dt_frame = 1.0 / fps

    # Uniform visualization time grid
    t_vis = np.arange(0.0, T_total, dt_frame)

    print(f"Total duration: {T_total:.3f} s, frames: {len(t_vis)}, FPS: {fps}")

    while True:
        k = 0  # current segment index
        for tv in t_vis:
            while k < N - 2 and tv > t[k + 1]:
                k += 1

            t0 = float(t[k])
            t1 = float(t[k + 1])
            if abs(t1 - t0) < 1e-9:
                alpha = 0.0
            else:
                alpha = (tv - t0) / (t1 - t0)
                alpha = max(0.0, min(1.0, alpha))

            q_k = q[k]
            q_kp1 = q[k + 1]
            q_interp = pin.interpolate(model, q_k, q_kp1, float(alpha))

            viz.display(q_interp)
            time.sleep(dt_frame)

        if not loop:
            break
        time.sleep(1.0)


def main():
    robot = load_robot()
    t, q, v, dt, f, meta = load_trajectory()

    # 1) Compute whole-body torques along the trajectory
    tau_all = compute_inverse_dynamics(robot, t, q, v, dt, f)
    print("tau_all shape:", tau_all.shape)

    # 1b) Check torque limits
    check_torque_limits(robot, tau_all)

    # Save torques for later use
    script_dir = Path(__file__).parent.resolve()
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    tau_path = results_dir / "inverse_dynamics_tau.npy"
    np.save(tau_path, tau_all)
    print(f"\nSaved inverse dynamics torques to: {tau_path}")

    # Quick debug plot for a couple of actuated joints
    time_knots = t[:-1]  # tau is defined for k = 0..N-2
    plt.figure(figsize=(10, 6))
    # First actuated DOF is usually at index 6 if free-flyer has 6 DOFs
    plt.plot(time_knots, tau_all[:, 6 + 0], label="DOF 6 (first actuated)")
    if tau_all.shape[1] > 7:
        plt.plot(time_knots, tau_all[:, 6 + 1], label="DOF 7")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.title("Inverse Dynamics Torques (example DOFs)")
    plt.grid(True)
    plt.legend()
    plt.savefig(results_dir / "inverse_dynamics_tau_example.png", dpi=300)
    plt.close()

    # 2) Optionally animate
    play_trajectory(robot, t, q, fps=60.0, loop=True)


if __name__ == "__main__":
    main()
