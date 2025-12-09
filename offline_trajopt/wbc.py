#!/usr/bin/env python3
import os
import time
import pickle
from pathlib import Path

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import matplotlib.pyplot as plt

import osqp
import scipy.sparse as sp

# Use X11 for Pinocchio viewer on some Linux / remote setups
os.environ["QT_QPA_PLATFORM"] = "xcb"

SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
URDF_PATH    = PROJECT_ROOT / "urdf" / "g1_description" / "g1_23dof.urdf"
RESULTS_DIR  = PROJECT_ROOT / "results"
PKL_PATH     = RESULTS_DIR / "trajectory_solution.pkl"

# Foot frames and corner offsets must match your optimizer
FOOT_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
CORNER_OFFSETS = [
    np.array([-0.05,  0.025, -0.03]),
    np.array([-0.05, -0.025, -0.03]),
    np.array([ 0.12,  0.03,  -0.03]),
    np.array([ 0.12, -0.03,  -0.03]),
]

MU_FRICTION = 0.5


# ------------------------------------------------------------
# Load robot and trajectory
# ------------------------------------------------------------

def load_robot():
    robot = RobotWrapper.BuildFromURDF(
        str(URDF_PATH),
        [str(URDF_PATH.parent)],
        root_joint=pin.JointModelFreeFlyer(),
    )
    robot.initViewer(loadModel=True)
    return robot


def load_traj():
    if not PKL_PATH.exists():
        raise FileNotFoundError(f"Could not find {PKL_PATH}")

    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    t  = np.asarray(data["t"])        # (N,)
    q  = np.asarray(data["q"])        # (N, nq)
    v  = np.asarray(data["v"])        # (N, nv)
    dt = np.asarray(data["dt"])       # (N-1,)
    f  = np.asarray(data["f"])        # (N-1, 2, 4, 3) corner GRFs in WORLD

    assert len(t)  == q.shape[0] == v.shape[0], "t/q/v mismatch"
    assert len(dt) == len(t) - 1, "dt length must be N-1"
    assert f.shape[0] == len(t) - 1, "f must be N-1 x 2 x 4 x 3"

    return t, q, v, dt, f, data


# ------------------------------------------------------------
# Helper: compute foot wrenches from corner forces
# ------------------------------------------------------------

def aggregate_foot_wrenches(model, data, q_k, f_corner_k, foot_ids):
    """
    Given:
      q_k           : (nq,)
      f_corner_k    : (2,4,3)  in WORLD frame (same as in your trajopt)
      foot_ids      : [id_left, id_right]
    Returns:
      f_ref_foot    : (2,6) in LOCAL_WORLD_ALIGNED frame of each foot:
                      [Fx, Fy, Fz, Tx, Ty, Tz]
    """
    pin.forwardKinematics(model, data, q_k)
    pin.updateFramePlacements(model, data)

    f_ref_foot = np.zeros((2, 6))

    for foot_idx, frame_id in enumerate(foot_ids):
        oMf = data.oMf[frame_id]
        p_frame = oMf.translation              # world position
        Rwf = oMf.rotation                     # world R frame ( ^W R_F )

        f_total_world   = np.zeros(3)
        tau_total_world = np.zeros(3)

        for c_idx, offset in enumerate(CORNER_OFFSETS):
            f_corner = f_corner_k[foot_idx, c_idx, :]      # WORLD
            p_corner = p_frame + Rwf @ offset              # WORLD corner pos

            f_total_world   += f_corner
            tau_total_world += np.cross(p_corner - p_frame, f_corner)

        # Convert to LOCAL_WORLD_ALIGNED (same orientation as J in LWA)
        f_lin_local  = Rwf.T @ f_total_world
        f_ang_local  = Rwf.T @ tau_total_world

        f_ref_foot[foot_idx, :3] = f_lin_local
        f_ref_foot[foot_idx, 3:] = f_ang_local

    return f_ref_foot  # shape (2,6)


# ------------------------------------------------------------
# WBIC QP solve per step (stance: push or landing)
# ------------------------------------------------------------

def wbic_solve_step(
    model,
    data,
    q,
    v,
    q_ref,
    v_ref,
    dt_k,
    f_corner_k,
    foot_ids,
    w_ddq_base=1e-4,
    w_ddq_joints=1e-2,
    w_f_lin=1e-5,
    w_f_ang=1e-7,
    phase_name="WBIC",
    use_force_tracking=True,
):
    """
    Solve a minimal WBIC QP for one timestep in a stance phase (push or landing).

    Variables x = [ddq (nv); fL (6); fR (6)].

    Cost:
      1/2 * ||W_ddq (ddq - ddq_des)||^2
    + (if use_force_tracking)
      1/2 * ||W_f (f - f_ref)||^2

    Constraints:
      - Floating base dynamics (first 6 rows of A ddq + h - Jc^T f = 0)
      - Contact kinematics: Jc ddq = -Jdotv (approx Jdotv = 0)
      - Friction + unilateral constraints on fL, F_r.
    """
    nv = model.nv
    nc = 2        # 2 feet
    nf = 6 * nc   # 6D wrench per foot

    # Compute dynamics terms
    pin.computeAllTerms(model, data, q, v)
    M = data.M.copy()
    h_vec = pin.nonLinearEffects(model, data, q, v)  # coriolis + gravity

    # Build stacked contact Jacobian Jc (12 x nv) in LOCAL_WORLD_ALIGNED
    J_list = []
    Jdotv_list = []
    for frame_id in foot_ids:
        J = pin.computeFrameJacobian(
            model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J_list.append(J)
        # For now, approximate Jdot * v as zero (typical WBIC simplification).
        Jdotv_list.append(np.zeros(6))

    Jc = np.vstack(J_list)           # (12, nv)
    Jdotv = np.concatenate(Jdotv_list)  # (12,)

    # Selection matrix for floating base (rows 0..5)
    S_f = np.zeros((6, nv))
    S_f[np.arange(6), np.arange(6)] = 1.0

    # Joint-space desired accelerations via PD on joints only
    qj      = q[7:]       # actuated positions
    vj      = v[6:]       # actuated velocities
    qj_ref  = q_ref[7:]
    vj_ref  = v_ref[6:]

    # Simple PD on joints for ddq_des
    Kp = 100.0
    Kd = 20.0
    ddq_des = np.zeros(nv)
    ddq_des[6:] = Kp * (qj_ref - qj) + Kd * (vj_ref - vj)

    # Weights for cost
    w_ddq = np.ones(nv) * w_ddq_joints
    w_ddq[:6] = w_ddq_base  # smaller weight on base accelerations

    # Force tracking weights / reference
    if use_force_tracking and f_corner_k is not None:
        f_ref_foot = aggregate_foot_wrenches(model, data, q_ref, f_corner_k, foot_ids)
        f_ref = f_ref_foot.reshape(-1)   # (12,)

        w_f = np.zeros(nf)
        for foot_idx in range(nc):
            base = 6 * foot_idx
            # linear forces
            w_f[base + 0] = w_f_lin
            w_f[base + 1] = w_f_lin
            w_f[base + 2] = w_f_lin
            # angular torques
            w_f[base + 3] = w_f_ang
            w_f[base + 4] = w_f_ang
            w_f[base + 5] = w_f_ang
    else:
        # No force tracking: zero reference and zero weights
        f_ref = np.zeros(nf)
        w_f   = np.zeros(nf)

    # Quadratic cost:
    #   1/2 * (ddq - ddq_des)^T W_ddq (ddq - ddq_des)
    # + 1/2 * (f   - f_ref   )^T W_f   (f   - f_ref)
    P_diag = np.concatenate([w_ddq, w_f])
    P = sp.diags(P_diag)

    q_cost = np.concatenate([-w_ddq * ddq_des, -w_f * f_ref])

    # --- Equality constraints ---
    # 1) Floating base dynamics:
    #    S_f (M ddq + h - Jc^T f) = 0
    A_dyn = np.hstack([S_f @ M, -S_f @ Jc.T])
    b_dyn = -S_f @ h_vec

    # 2) Contact kinematics:
    #    Jc ddq + Jdotv = 0   (approx)
    A_kin = np.hstack([Jc, np.zeros((Jc.shape[0], nf))])
    b_kin = -Jdotv

    A_eq = np.vstack([A_dyn, A_kin])
    b_eq = np.concatenate([b_dyn, b_kin])

    # --- Inequality constraints: friction + unilateral per foot ---
    A_ineq_rows = []
    u_ineq = []

    for foot_idx in range(nc):
        base = nv + 6 * foot_idx  # index in x where this foot's wrench starts
        Fx_idx = base + 0
        Fy_idx = base + 1
        Fz_idx = base + 2

        # 1) -Fz <= 0        (Fz >= 0)
        row = np.zeros(nv + nf)
        row[Fz_idx] = -1.0
        A_ineq_rows.append(row)
        u_ineq.append(0.0)

        # 2) Fx - mu Fz <= 0
        row = np.zeros(nv + nf)
        row[Fx_idx] = 1.0
        row[Fz_idx] = -MU_FRICTION
        A_ineq_rows.append(row)
        u_ineq.append(0.0)

        # 3) -Fx - mu Fz <= 0
        row = np.zeros(nv + nf)
        row[Fx_idx] = -1.0
        row[Fz_idx] = -MU_FRICTION
        A_ineq_rows.append(row)
        u_ineq.append(0.0)

        # 4) Fy - mu Fz <= 0
        row = np.zeros(nv + nf)
        row[Fy_idx] = 1.0
        row[Fz_idx] = -MU_FRICTION
        A_ineq_rows.append(row)
        u_ineq.append(0.0)

        # 5) -Fy - mu Fz <= 0
        row = np.zeros(nv + nf)
        row[Fy_idx] = -1.0
        row[Fz_idx] = -MU_FRICTION
        A_ineq_rows.append(row)
        u_ineq.append(0.0)

    if len(A_ineq_rows) > 0:
        A_ineq = np.vstack(A_ineq_rows)
        u_ineq = np.array(u_ineq)
        l_ineq = -np.inf * np.ones_like(u_ineq)
    else:
        A_ineq = np.zeros((0, nv + nf))
        u_ineq = np.zeros(0)
        l_ineq = np.zeros(0)

    # Stack equalities and inequalities
    A = np.vstack([A_eq, A_ineq])
    l = np.concatenate([b_eq, l_ineq])
    u = np.concatenate([b_eq, u_ineq])

    # Solve QP with OSQP
    prob = osqp.OSQP()
    prob.setup(
        P=P.tocsc(),
        q=q_cost,
        A=sp.csc_matrix(A),
        l=l,
        u=u,
        verbose=False,
        polish=True,
        eps_abs=1e-5,
        eps_rel=1e-5,
        max_iter=20000,
    )

    t0 = time.perf_counter()
    res = prob.solve()
    solve_time = time.perf_counter() - t0

    # Continuous QP debug printout
    print(
        f"[{phase_name} QP] status={res.info.status}, "
        f"iters={res.info.iter}, "
        f"time={solve_time*1e3:.2f} ms"
    )

    if res.info.status_val not in (
        osqp.constant("OSQP_SOLVED"),
        osqp.constant("OSQP_SOLVED_INACCURATE"),
    ):
        print(f"[{phase_name}] OSQP failed with status: {res.info.status}")
        # Fallback: just use ddq_des, f_ref
        ddq = ddq_des
        f_contact = f_ref.copy()
    else:
        x_opt = res.x
        ddq = x_opt[:nv]
        f_contact = x_opt[nv:]

    return ddq, f_contact


# ------------------------------------------------------------
# Flight phase: exact reference playback (no PD, no QP)
# ------------------------------------------------------------

def flight_ddq_ref_only(v_ref_k, v_ref_k_next, dt_k):
    """
    Compute reference generalized acceleration from reference velocities.

    ddq_ref = (v_ref_{k+1} - v_ref_k) / dt_k

    If we integrate once per knot with this ddq, starting from the
    correct initial v, we exactly reproduce the reference v sequence.
    """
    if dt_k > 0.0:
        return (v_ref_k_next - v_ref_k) / dt_k
    else:
        return np.zeros_like(v_ref_k)


# ------------------------------------------------------------
# Main: forward simulate push + flight + landing
# ------------------------------------------------------------

def main():
    robot = load_robot()
    model = robot.model
    data  = model.createData()

    t, q_ref_all, v_ref_all, dt_all, f_corner_all, meta = load_traj()

    N = q_ref_all.shape[0]
    print("Trajectory shapes:")
    print("  t:", t.shape)
    print("  q:", q_ref_all.shape)
    print("  v:", v_ref_all.shape)
    print("  dt:", dt_all.shape)
    print("  f:", f_corner_all.shape)

    # Phase split must match your offline optimizer
    N_push   = 30
    N_flight = 20
    N_land   = N - (N_push + N_flight)
    if N_land <= 0:
        raise RuntimeError(
            f"Invalid phase split: N={N}, N_push={N_push}, N_flight={N_flight}, "
            f"N_land={N_land}"
        )

    liftoff_idx   = N_push             # state index where flight starts
    touchdown_idx = N_push + N_flight  # state index where landing starts

    # Durations per phase (for info)
    T_push   = float(np.sum(dt_all[:N_push]))
    T_flight = float(np.sum(dt_all[N_push:N_push + N_flight]))
    T_land   = float(np.sum(dt_all[N_push + N_flight:]))

    print(
        f"\nPhase split:"
        f"\n  push:   states 0..{N_push-1},   dt indices 0..{N_push-1},   T={T_push:.4f} s"
        f"\n  flight: states {liftoff_idx}..{touchdown_idx-1}, "
        f"dt indices {N_push}..{N_push+N_flight-1}, T={T_flight:.4f} s"
        f"\n  land:   states {touchdown_idx}..{N-1}, "
        f"dt indices {N_push+N_flight}..{N-2}, T={T_land:.4f} s"
    )

    # Precompute foot ids
    foot_ids = [model.getFrameId(n) for n in FOOT_NAMES]

    # Integration substeps
    substeps_wbic   = 4   # for stance (push + landing)
    substeps_flight = 1   # match optimizer for flight

    cycle = 0
    while True:
        print(f"\n=== Starting forward simulation cycle {cycle} (push + flight + landing) ===")

        # Reset sim state to the reference at k=0 for each cycle
        q = q_ref_all[0].copy()
        v = v_ref_all[0].copy()

        robot.viz.display(q)
        time.sleep(1.0)

        # Logs for debugging / COM plotting (for this cycle)
        times_log = [0.0]
        com_z_log = []
        t_sim = 0.0

        # ------------------------------------------------------------
        # Single loop over all dt indices, with phase-based behavior
        # ------------------------------------------------------------
        for k in range(N - 1):
            h = float(dt_all[k])
            if h <= 0.0:
                continue

            q_ref_k      = q_ref_all[k]
            v_ref_k      = v_ref_all[k]
            q_ref_k_next = q_ref_all[k + 1]
            v_ref_k_next = v_ref_all[k + 1]

            # Determine phase
            if k < N_push:
                phase = "PUSH"
            elif k < N_push + N_flight:
                phase = "FLIGHT"
            else:
                phase = "LAND"

            # --------------------------------------------------------
            # PUSH PHASE: WBIC QP with contact force tracking
            # --------------------------------------------------------
            if phase == "PUSH":
                print(
                    f"[WBIC-PUSH] cycle={cycle}, k={k}/{N-2}, "
                    f"h={h:.4f}, t_sim={t_sim:.4f}"
                )

                f_corner_k = f_corner_all[k]  # (2,4,3)

                ddq, f_contact = wbic_solve_step(
                    model,
                    data,
                    q,
                    v,
                    q_ref_k,
                    v_ref_k,
                    h,
                    f_corner_k,
                    foot_ids,
                    phase_name="WBIC-PUSH",
                    use_force_tracking=True,
                )

                h_sub = h / substeps_wbic
                for s in range(substeps_wbic):
                    v = v + ddq * h_sub
                    q = pin.integrate(model, q, v * h_sub)
                    t_sim += h_sub

                    # Log COM height
                    pin.centerOfMass(model, data, q)
                    com_z = float(data.com[0][2])
                    com_z_log.append(com_z)
                    times_log.append(t_sim)

                    # Basic blow-up check
                    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
                        print(f"\n[SIM] NaN/Inf encountered in push at t={t_sim:.4f}")
                        print("  |q[0:3]|:", np.linalg.norm(q[0:3]))
                        print("  |v|:", np.linalg.norm(v))
                        robot.viz.display(q)
                        break

                robot.viz.display(q)
                time.sleep(min(h * 5.0, 0.1))

                if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
                    break

            # --------------------------------------------------------
            # FLIGHT PHASE: exact reference ballistic playback (no QP)
            # --------------------------------------------------------
            elif phase == "FLIGHT":
                print(
                    f"[FLIGHT] cycle={cycle}, k={k}/{N-2}, "
                    f"h={h:.4f}, t_sim={t_sim:.4f}"
                )

                ddq_ref = flight_ddq_ref_only(v_ref_k, v_ref_k_next, h)

                # Single-step semi-implicit Euler to match optimizer
                v = v + ddq_ref * h
                q = pin.integrate(model, q, v * h)
                t_sim += h

                pin.centerOfMass(model, data, q)
                com_z = float(data.com[0][2])
                com_z_log.append(com_z)
                times_log.append(t_sim)

                if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
                    print(f"\n[SIM] NaN/Inf encountered in flight at t={t_sim:.4f}")
                    print("  |q[0:3]|:", np.linalg.norm(q[0:3]))
                    print("  |v|:", np.linalg.norm(v))
                    robot.viz.display(q)
                    break

                robot.viz.display(q)
                time.sleep(min(h * 5.0, 0.1))

            # --------------------------------------------------------
            # LANDING PHASE: WBIC QP with NO force tracking
            # --------------------------------------------------------
            else:  # phase == "LAND"
                print(
                    f"[WBIC-LAND] cycle={cycle}, k={k}/{N-2}, "
                    f"h={h:.4f}, t_sim={t_sim:.4f}"
                )

                # We still pass f_corner_k, but wbic_solve_step will ignore it
                # when use_force_tracking=False
                f_corner_k = f_corner_all[k]

                ddq, f_contact = wbic_solve_step(
                    model,
                    data,
                    q,
                    v,
                    q_ref_k,
                    v_ref_k,
                    h,
                    f_corner_k,
                    foot_ids,
                    phase_name="WBIC-LAND",
                    use_force_tracking=False,  # <--- key change for landing
                )

                h_sub = h / substeps_wbic
                for s in range(substeps_wbic):
                    v = v + ddq * h_sub
                    q = pin.integrate(model, q, v * h_sub)
                    t_sim += h_sub

                    # Log COM height
                    pin.centerOfMass(model, data, q)
                    com_z = float(data.com[0][2])
                    com_z_log.append(com_z)
                    times_log.append(t_sim)

                    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
                        print(f"\n[SIM] NaN/Inf encountered in landing at t={t_sim:.4f}")
                        print("  |q[0:3]|:", np.linalg.norm(q[0:3]))
                        print("  |v|:", np.linalg.norm(v))
                        robot.viz.display(q)
                        break

                robot.viz.display(q)
                time.sleep(min(h * 5.0, 0.1))

                if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
                    break

        print("\n[SIM] Full (push + flight + landing) forward simulation finished (or stopped).")

        # ------------------------------------------------------------
        # Save COM plot (only for first cycle) and loop
        # ------------------------------------------------------------
        if cycle == 0 and len(com_z_log) > 0:
            times_log = np.array(times_log)
            com_z_log = np.array(com_z_log)

            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            plt.figure()
            plt.plot(times_log[:len(com_z_log)], com_z_log)
            plt.xlabel("time [s]")
            plt.ylabel("COM z [m]")
            plt.grid(True)
            plt.title("COM height: WBIC push + ref flight + WBIC landing (no GRF tracking)")
            plt.savefig(RESULTS_DIR / "wbic_push_flight_land_com_z.png", dpi=200)
            plt.close()

            print("Saved WBIC COM plot to:", RESULTS_DIR / "wbic_push_flight_land_com_z.png")

        cycle += 1
        # loop back and repeat the full sequence again


if __name__ == "__main__":
    main()
