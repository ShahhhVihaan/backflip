#!/usr/bin/env python3
import os
import time
import pickle
from pathlib import Path

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import matplotlib.pyplot as plt

# Force Qt to use X11 (xcb) instead of Wayland (for remote / some Linux setups)
os.environ["QT_QPA_PLATFORM"] = "xcb"

SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
URDF_PATH    = PROJECT_ROOT / "urdf" / "g1_description" / "g1_23dof.urdf"
RESULTS_DIR  = PROJECT_ROOT / "results"
PKL_PATH     = RESULTS_DIR / "trajectory_solution.pkl"
TAU_PATH     = RESULTS_DIR / "inverse_dynamics_tau.npy"


# ------------------------
# Robot & Trajectory I/O
# ------------------------

def load_robot():
    """Load the G1 robot with a free-flyer base and init viewer."""
    robot = RobotWrapper.BuildFromURDF(
        str(URDF_PATH),
        [str(URDF_PATH.parent)],
        root_joint=pin.JointModelFreeFlyer(),
    )
    robot.initViewer(loadModel=True)
    return robot


def load_traj_and_tau():
    """Load optimizer trajectory and inverse-dynamics torques."""
    if not PKL_PATH.exists():
        raise FileNotFoundError(f"Could not find {PKL_PATH}")
    if not TAU_PATH.exists():
        raise FileNotFoundError(f"Could not find {TAU_PATH}")

    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    t  = np.asarray(data["t"])        # (N,)
    q  = np.asarray(data["q"])        # (N, nq)
    v  = np.asarray(data["v"])        # (N, nv)
    dt = np.asarray(data["dt"])       # (N-1,)
    f  = np.asarray(data["f"])        # (N-1, 2, 4, 3) corner GRFs

    tau = np.load(TAU_PATH)           # (N-1, nv)

    assert len(t)  == q.shape[0] == v.shape[0], "t/q/v mismatch"
    assert len(dt) == len(t) - 1, "dt length must be N-1"
    assert tau.shape[0] == len(t) - 1, "tau must be N-1 x nv"

    return t, q, v, dt, f, tau, data


# ------------------------
# External wrenches (from reference)
# ------------------------

CORNER_OFFSETS = [
    np.array([-0.05,  0.025, -0.03]),
    np.array([-0.05, -0.025, -0.03]),
    np.array([ 0.12,  0.03,  -0.03]),
    np.array([ 0.12, -0.03,  -0.03]),
]
FOOT_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def build_external_wrenches_from_ref(model, data, q_ref_k, f_k):
    """
    Build external wrenches for ABA from corner forces at a timestep.

    q_ref_k : reference configuration (nq,)
    f_k     : (2, 4, 3) – [foot_idx, corner_idx, xyz] in WORLD frame

    Returns:
        fext: list of pin.Force in LOCAL joint frames (len = model.njoints).
    """
    foot_ids       = [model.getFrameId(n) for n in FOOT_NAMES]
    foot_joint_ids = [model.frames[fid].parentJoint for fid in foot_ids]

    pin.forwardKinematics(model, data, q_ref_k)
    pin.updateFramePlacements(model, data)

    fext = [pin.Force.Zero() for _ in range(model.njoints)]

    for foot_idx, (frame_id, joint_id) in enumerate(zip(foot_ids, foot_joint_ids)):
        oMf     = data.oMf[frame_id]
        p_frame = oMf.translation
        Rwf     = oMf.rotation  # world R frame

        f_total_world   = np.zeros(3)
        tau_total_world = np.zeros(3)

        for c_idx, offset in enumerate(CORNER_OFFSETS):
            f_corner = f_k[foot_idx, c_idx, :]   # WORLD force
            p_corner = p_frame + Rwf @ offset    # WORLD position of corner

            f_total_world   += f_corner
            tau_total_world += np.cross(p_corner - p_frame, f_corner)

        # Convert to local (foot) frame
        f_local_linear  = Rwf.T @ f_total_world
        f_local_angular = Rwf.T @ tau_total_world

        fext[joint_id] = pin.Force(f_local_linear, f_local_angular)

    return fext


# ------------------------
# Consistency check: RNEA vs ABA on the *reference*
# ------------------------

def check_inverse_dynamics_consistency(model, t, q, v, dt, f_corner, tau_ff_all):
    """
    For each knot k, compare:
      a_fd   = (v[k+1] - v[k]) / dt[k]
      a_aba  = aba(q[k], v[k], tau_ff_all[k], fext_from_ref(k))

    If this is small, τ_ff and GRFs are consistent with the reference dynamics.
    """
    data = model.createData()
    N    = q.shape[0]

    diffs = []

    print("\n=== Checking inverse-dynamics consistency (ABA vs finite diff) ===")
    for k in range(N - 1):
        h = float(dt[k])
        if h <= 0.0:
            continue

        qk  = q[k]
        vk  = v[k]
        vk1 = v[k + 1]

        a_fd = (vk1 - vk) / h

        fext_k = build_external_wrenches_from_ref(model, data, qk, f_corner[k])
        a_aba  = pin.aba(model, data, qk, vk, tau_ff_all[k], fext_k)

        diff = a_aba - a_fd
        norm = np.linalg.norm(diff)
        diffs.append(norm)

        if k < 5 or k == N - 2:
            print(f"  k={k:2d}: ||a_aba - a_fd|| = {norm:.3e}")

    diffs = np.array(diffs)
    print(f"\n  max ||a_aba - a_fd|| over knots: {np.max(diffs):.3e}")
    print(f"  mean||a_aba - a_fd|| over knots: {np.mean(diffs):.3e}")
    print("=== Done consistency check ===\n")


# ------------------------
# Forward dynamics with τ_ff (+ optional PD) + GRFs, with logging
# ------------------------

def simulate_forward_with_pd(
    robot,
    t,
    q_ref,
    v_ref,
    dt,
    f_corner,
    tau_ff_all,
    substeps=4,
    Kp_scalar=0.0,
    Kd_scalar=0.0,
):
    """
    Single pass forward simulation (one backflip attempt):

        τ = τ_ff[k]          (feedforward)
          + PD(q_ref[k]-q)   (if Kp_scalar>0)
          + PD(v_ref[k]-v)   (if Kd_scalar>0)

    External forces are built from the *reference* pose and contact forces.

    We **only** clamp joint torques by effort limits (indices 6:),
    and leave the floating base wrench (0:6) unconstrained.

    Logs:
      - t_log      : list of times
      - qerr_log   : ||q_ref_joints - q_joints|| over time
      - verr_log   : ||v_ref_joints - v_joints|| over time
      - vnorm_log  : ||v|| over time
      - q_log      : list of q over time
      - v_log      : list of v over time
    """
    model = robot.model
    data  = model.createData()
    viz   = robot.viz

    N = q_ref.shape[0]
    assert tau_ff_all.shape[0] == N - 1
    assert f_corner.shape[0]   == N - 1

    # Indices for actuated joints (after free-flyer)
    idx_q_joints = slice(7, model.nq)   # positions of actuated joints
    idx_v_joints = slice(6, model.nv)   # velocities of actuated joints

    # Effort limits: per DOF; keep only positive magnitudes
    if hasattr(model, "effortLimit") and model.effortLimit.shape[0] == model.nv:
        tau_limit_full = np.array(model.effortLimit).copy()
        tau_limit_full[tau_limit_full < 0] = 0.0
    else:
        tau_limit_full = np.ones(model.nv) * 1e3

    tau_limit_joints = tau_limit_full[idx_v_joints]

    # Initial state = reference start
    q = q_ref[0].copy()
    v = v_ref[0].copy()

    viz.display(q)
    time.sleep(0.2)

    print("\n=== Forward simulation in Pinocchio (ABA + τ_ff + PD + GRFs) ===")
    print(f"N knots: {N}, total time: {t[-1]:.3f} s, substeps per knot: {substeps}")
    print(f"Kp_scalar = {Kp_scalar}, Kd_scalar = {Kd_scalar}")

    # Logs
    t_log      = []
    qerr_log   = []
    verr_log   = []
    vnorm_log  = []
    q_log      = []
    v_log      = []

    t_curr = 0.0

    for k in range(N - 1):
        h = float(dt[k])
        if h <= 0.0:
            continue

        h_sub = h / float(substeps)

        qk_ref = q_ref[k]
        vk_ref = v_ref[k]
        f_k    = f_corner[k]
        tau_ff = tau_ff_all[k]

        for s in range(substeps):
            # External wrenches from reference pose
            fext_k = build_external_wrenches_from_ref(model, data, qk_ref, f_k)

            # Start from feedforward τ
            tau = np.array(tau_ff, copy=True)

            if Kp_scalar != 0.0 or Kd_scalar != 0.0:
                # Joint-space errors only (skip floating base)
                q_err = np.zeros(model.nq)
                v_err = np.zeros(model.nv)
                q_err[idx_q_joints] = qk_ref[idx_q_joints] - q[idx_q_joints]
                v_err[idx_v_joints] = vk_ref[idx_v_joints] - v[idx_v_joints]

                tau_joints = (
                    tau[idx_v_joints]
                    + Kp_scalar * q_err[idx_q_joints]
                    + Kd_scalar * v_err[idx_v_joints]
                )

                # Clamp to effort limits
                tau_joints = np.clip(tau_joints, -tau_limit_joints, tau_limit_joints)
                tau[idx_v_joints] = tau_joints
                # Floating base 0:6 remains as in tau_ff

            # Forward dynamics
            a = pin.aba(model, data, q, v, tau, fext_k)

            # Semi-implicit Euler
            v = v + a * h_sub
            q = pin.integrate(model, q, v * h_sub)
            t_curr += h_sub

            # --- logging ---
            q_err_j = qk_ref[idx_q_joints] - q[idx_q_joints]
            v_err_j = vk_ref[idx_v_joints] - v[idx_v_joints]

            t_log.append(t_curr)
            qerr_log.append(np.linalg.norm(q_err_j))
            verr_log.append(np.linalg.norm(v_err_j))
            vnorm_log.append(np.linalg.norm(v))
            q_log.append(q.copy())
            v_log.append(v.copy())

            # Basic sanity checks
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
                print(f"\n[ABA] NaN/Inf at knot {k}, substep {s}")
                print("  |q[0:3]|:", np.linalg.norm(q[0:3]))
                print("  |v[0:6]|:", np.linalg.norm(v[0:6]))
                return t_log, qerr_log, verr_log, vnorm_log, q_log, v_log

            if np.linalg.norm(v) > 1e4:
                print(f"\n[ABA] Velocity exploded at knot {k}, substep {s}")
                print("  |v|:", np.linalg.norm(v))
                return t_log, qerr_log, verr_log, vnorm_log, q_log, v_log

        # One viz update per knot
        viz.display(q)
        viz_slowdown = 5.0
        time.sleep(min(h * viz_slowdown, 0.1))

    print("\n[ABA] Forward simulation finished normally.")
    return t_log, qerr_log, verr_log, vnorm_log, q_log, v_log


# ------------------------
# Plot helpers
# ------------------------

def save_debug_plots(robot, t, v_ref_full, t_log, qerr_log, verr_log, vnorm_log, v_log):
    """Save debug plots using the logs from a single forward pass."""
    t_log     = np.asarray(t_log)
    qerr_log  = np.asarray(qerr_log)
    verr_log  = np.asarray(verr_log)
    vnorm_log = np.asarray(vnorm_log)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if t_log.size == 0:
        print("No logs to plot (t_log is empty).")
        return

    # 1) Norms of q_err and v_err
    plt.figure()
    plt.plot(t_log, qerr_log, label="||q_err|| (joints)")
    plt.plot(t_log, verr_log, label="||v_err|| (joints)")
    plt.xlabel("time [s]")
    plt.ylabel("error norm")
    plt.legend()
    plt.grid(True)
    plt.title("Tracking error vs time")
    plt.savefig(RESULTS_DIR / "forward_tracking_error.png", dpi=200)
    plt.close()

    # 2) Norm of full velocity vector
    plt.figure()
    plt.plot(t_log, vnorm_log)
    plt.xlabel("time [s]")
    plt.ylabel("||v||")
    plt.grid(True)
    plt.title("Velocity norm vs time")
    plt.savefig(RESULTS_DIR / "forward_velocity_norm.png", dpi=200)
    plt.close()

    # 3) Example joint: left_hip_pitch_joint velocity vs reference
    try:
        model = robot.model
        j_id = model.getJointId("left_hip_pitch_joint")  # Pinocchio joint id
        vj   = model.idx_v[j_id]                        # joint velocity index

        v_forward = np.vstack(v_log)  # (T, nv)
        # Reference velocities are defined at knot times t[0..N-1]; we use t[:-1] to match tau
        v_ref_joint = v_ref_full[:-1, vj]
        # Interpolate reference onto t_log
        v_ref_interp = np.interp(t_log, t[:-1], v_ref_joint)

        plt.figure()
        plt.plot(t_log, v_forward[:, vj], label="forward v_left_hip_pitch")
        plt.plot(t_log, v_ref_interp, "--", label="ref v_left_hip_pitch (interp)")
        plt.xlabel("time [s]")
        plt.ylabel("joint velocity")
        plt.legend()
        plt.grid(True)
        plt.title("Left hip pitch velocity: forward vs reference")
        plt.savefig(RESULTS_DIR / "forward_vs_ref_left_hip_vel.png", dpi=200)
        plt.close()
    except Exception as e:
        print("Could not plot joint-specific velocity comparison:", e)


# ------------------------
# Main
# ------------------------

def main():
    robot = load_robot()
    t, q, v, dt, f_corner, tau_ff_all, meta = load_traj_and_tau()

    print("Trajectory shapes:")
    print("  t:", t.shape)
    print("  q:", q.shape)
    print("  v:", v.shape)
    print("  dt:", dt.shape)
    print("  f:", f_corner.shape)
    print("  tau_ff_all:", tau_ff_all.shape)

    # 1) Numeric sanity check (no viz)
    print("\n--- Checking ABA vs finite-diff accelerations on reference ---")
    check_inverse_dynamics_consistency(
        robot.model, t, q, v, dt, f_corner, tau_ff_all
    )

    # Give you a chance to open the Meshcat viewer before sim starts
    input("\nViewer is up. Open the URL in your browser, then press <Enter> to start forward sims...")

    print("\nReplaying forward dynamics (ABA + τ_ff + PD + GRFs) in a loop.\n")

    first_pass = True

    while True:
        (t_log,
         qerr_log,
         verr_log,
         vnorm_log,
         q_log,
         v_log) = simulate_forward_with_pd(
            robot,
            t,
            q_ref=q,
            v_ref=v,
            dt=dt,
            f_corner=f_corner,
            tau_ff_all=tau_ff_all,
            substeps=4,
            Kp_scalar=0.0,   # still pure feedforward here
            Kd_scalar=0.0,
        )

        # On the first pass, dump debug plots so we can inspect exploding behavior
        if first_pass and len(t_log) > 0:
            save_debug_plots(
                robot,
                t=t,
                v_ref_full=v,
                t_log=t_log,
                qerr_log=qerr_log,
                verr_log=verr_log,
                vnorm_log=vnorm_log,
                v_log=v_log,
            )
            print("Saved debug plots to:", RESULTS_DIR)
            first_pass = False

        print("One forward sim pass finished. Replaying again in 1 second...\n")
        time.sleep(1.0)


if __name__ == "__main__":
    main()
