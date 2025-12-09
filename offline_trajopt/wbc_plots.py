#!/usr/bin/env python3
import os
import time
import pickle
from pathlib import Path

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

import osqp
import scipy.sparse as sp
import matplotlib.pyplot as plt  # NEW: for plotting

# ------------------------------------------------------------------
# Paths (from your project layout)
# ------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
URDF_PATH    = PROJECT_ROOT / "urdf" / "g1_description" / "g1_23dof.urdf"
RESULTS_DIR  = PROJECT_ROOT / "results"
PKL_PATH     = RESULTS_DIR / "trajectory_solution.pkl"

# Make Qt use X11 on some remote setups
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Foot frames and corner offsets must match your optimizer
FOOT_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
CORNER_OFFSETS = [
    np.array([-0.05,  0.025, -0.03]),
    np.array([-0.05, -0.025, -0.03]),
    np.array([ 0.12,  0.03,  -0.03]),
    np.array([ 0.12, -0.03,  -0.03]),
]

MU_FRICTION = 0.5


# ================================================================
# LOAD ROBOT + TRAJ
# ================================================================
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

    assert len(t)  == q.shape[0] == v.shape[0]
    assert len(dt) == len(t) - 1
    assert f.shape[0] == len(t) - 1

    return t, q, v, dt, f, data


# ================================================================
# HELPER: CORNER FORCES → FOOT WRENCHES
# ================================================================
def aggregate_foot_wrenches(model, data, q_k, f_corner_k, foot_ids):
    """
    Given:
      q_k           : (nq,)
      f_corner_k    : (2,4,3) in WORLD frame
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

        # Convert to LOCAL_WORLD_ALIGNED
        f_lin_local  = Rwf.T @ f_total_world
        f_ang_local  = Rwf.T @ tau_total_world

        f_ref_foot[foot_idx, :3] = f_lin_local
        f_ref_foot[foot_idx, 3:] = f_ang_local

    return f_ref_foot  # shape (2,6)


# ================================================================
# HELPER: FINITE-DIFFERENCE ddq_ref
# ================================================================
def generalized_ddq_ref(v_k, v_kp1, h, ddq_max=40.0):
    """
    Simple finite-diff:
        ddq_ref = (v_{k+1} - v_k) / h
    with clipping to keep things sane.
    """
    if h <= 0.0:
        return np.zeros_like(v_k)
    ddq = (v_kp1 - v_k) / h
    ddq = np.clip(ddq, -ddq_max, ddq_max)
    return ddq


# ================================================================
# WBIC QP (STANCE)
# ================================================================
def wbic_solve_step(
    model,
    data,
    q,
    v,
    q_ref_k,
    v_ref_k,
    ddq_ref_k,
    f_corner_k,
    foot_ids,
    # ddq weights
    w_ddq_base=1e-4,
    w_ddq_joints=1e-2,
    # force tracking weights (axis-specific)
    w_fx=5e-2,
    w_fy=5e-2,
    w_fz=1e-3,
    w_tau=1e-4,
    phase_name="WBIC",
    use_force_tracking=True,
):
    """
    WBIC-style QP:
      variables: x = [ddq (nv); fL (6); fR (6)]
    """
    nv = model.nv
    nc = 2
    nf = 6 * nc

    # Dynamics terms
    pin.computeAllTerms(model, data, q, v)
    M = data.M.copy()
    h_vec = pin.nonLinearEffects(model, data, q, v)

    # Contact Jacobians
    J_list = []
    Jdotv_list = []
    for frame_id in foot_ids:
        J = pin.computeFrameJacobian(
            model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J_list.append(J)
        Jdotv_list.append(np.zeros(6))  # approximate Jdot*v ≈ 0

    Jc = np.vstack(J_list)              # (12, nv)
    Jdotv = np.concatenate(Jdotv_list)  # (12,)

    # Floating-base selector (first 6 dofs)
    S_f = np.zeros((6, nv))
    S_f[np.arange(6), np.arange(6)] = 1.0

    # ------------------------------------------------------------------
    # State-feedback augmented ddq_des (joints only)
    # ------------------------------------------------------------------
    ddq_des = ddq_ref_k.copy()

    # joint slices
    qj     = q[7:]
    vj     = v[6:]
    qj_ref = q_ref_k[7:]
    vj_ref = v_ref_k[6:]

    # stance gains
    Kp = 600.0
    Kd = 80.0

    ddq_des[6:] += Kp * (qj_ref - qj) + Kd * (vj_ref - vj)

    # Cost weights on ddq
    w_ddq = np.ones(nv) * w_ddq_joints
    w_ddq[:6] = w_ddq_base

    # Force tracking from corner forces
    if use_force_tracking and (f_corner_k is not None):
        f_ref_foot = aggregate_foot_wrenches(model, data, q_ref_k, f_corner_k, foot_ids)
        f_ref = f_ref_foot.reshape(-1)   # (12,)

        w_f = np.zeros(nf)
        for foot_idx in range(nc):
            base = 6 * foot_idx
            # linear forces: Fx, Fy, Fz
            w_f[base + 0] = w_fx
            w_f[base + 1] = w_fy
            w_f[base + 2] = w_fz
            # torques: Tx, Ty, Tz
            w_f[base + 3] = w_tau
            w_f[base + 4] = w_tau
            w_f[base + 5] = w_tau
    else:
        f_ref_foot = np.zeros((nc, 6))
        f_ref = np.zeros(nf)
        w_f   = np.zeros(nf)

    # Quadratic cost: 0.5 x^T P x + q^T x
    P_diag = np.concatenate([w_ddq, w_f])
    P = sp.diags(P_diag)
    q_cost = np.concatenate([-w_ddq * ddq_des, -w_f * f_ref])

    # ------------------------------------------------------------------
    # Equalities:
    #   1) floating-base dynamics: S_f (M ddq + h - Jc^T f) = 0
    #   2) contact kinematics:     Jc ddq + Jdotv = 0
    # ------------------------------------------------------------------
    A_dyn = np.hstack([S_f @ M, -S_f @ Jc.T])
    b_dyn = -S_f @ h_vec

    A_kin = np.hstack([Jc, np.zeros((Jc.shape[0], nf))])
    b_kin = -Jdotv

    A_eq = np.vstack([A_dyn, A_kin])
    b_eq = np.concatenate([b_dyn, b_kin])

    # ------------------------------------------------------------------
    # Inequalities: friction + unilateral
    # ------------------------------------------------------------------
    A_ineq_rows = []
    u_ineq = []

    for foot_idx in range(nc):
        base = nv + 6 * foot_idx
        Fx_idx = base + 0
        Fy_idx = base + 1
        Fz_idx = base + 2

        # Fz >= 0  →  -Fz <= 0
        row = np.zeros(nv + nf)
        row[Fz_idx] = -1.0
        A_ineq_rows.append(row)
        u_ineq.append(0.0)

        # |Fx| <= mu Fz
        row = np.zeros(nv + nf)
        row[Fx_idx] = 1.0
        row[Fz_idx] = -MU_FRICTION
        A_ineq_rows.append(row)
        u_ineq.append(0.0)

        row = np.zeros(nv + nf)
        row[Fx_idx] = -1.0
        row[Fz_idx] = -MU_FRICTION
        A_ineq_rows.append(row)
        u_ineq.append(0.0)

        # |Fy| <= mu Fz
        row = np.zeros(nv + nf)
        row[Fy_idx] = 1.0
        row[Fz_idx] = -MU_FRICTION
        A_ineq_rows.append(row)
        u_ineq.append(0.0)

        row = np.zeros(nv + nf)
        row[Fy_idx] = -1.0
        row[Fz_idx] = -MU_FRICTION
        A_ineq_rows.append(row)
        u_ineq.append(0.0)

    if A_ineq_rows:
        A_ineq = np.vstack(A_ineq_rows)
        u_ineq = np.array(u_ineq)
        l_ineq = -np.inf * np.ones_like(u_ineq)
    else:
        A_ineq = np.zeros((0, nv + nf))
        u_ineq = np.zeros(0)
        l_ineq = np.zeros(0)

    # Stack all constraints
    A = np.vstack([A_eq, A_ineq])
    l = np.concatenate([b_eq, l_ineq])
    u = np.concatenate([b_eq, u_ineq])

    # Solve with OSQP
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
    res = prob.solve()

    print(
        f"[{phase_name} QP] status={res.info.status}, "
        f"iters={res.info.iter}, "
        f"time={res.info.run_time * 1e3:.2f} ms"
    )

    if res.info.status_val not in (
        osqp.constant("OSQP_SOLVED"),
        osqp.constant("OSQP_SOLVED_INACCURATE"),
    ):
        print(f"[{phase_name}] OSQP failed; falling back to ddq_des")
        ddq = ddq_des
        f_contact = f_ref.copy()
    else:
        x_opt = res.x
        ddq = x_opt[:nv]
        f_contact = x_opt[nv:]

    return ddq, f_contact, f_ref_foot  # NOTE: return f_ref_foot explicitly


# ================================================================
# PLOTTING
# ================================================================
def plot_results(t_log, com_ref_log, com_sim_log, f_ref_log, f_qp_log, N_push):
    t_log = np.array(t_log)
    com_ref_log = np.array(com_ref_log)  # (T,3)
    com_sim_log = np.array(com_sim_log)  # (T,3)
    f_ref_log = np.array(f_ref_log)      # (T,2,6)
    f_qp_log  = np.array(f_qp_log)       # (T,2,6)

    t_push = t_log[N_push-1]

    # --- COM position ---
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    labels = ['x', 'y', 'z']
    for i in range(3):
        axes[i].plot(t_log, com_ref_log[:, i], label='ref')
        axes[i].plot(t_log, com_sim_log[:, i], '--', label='sim')
        axes[i].axvline(t_push, linestyle=':', alpha=0.7)
        axes[i].set_ylabel(f'COM {labels[i]} [m]')
    axes[0].set_title('COM position: reference vs simulated (stance + flight)')
    axes[-1].set_xlabel('time [s]')
    axes[0].legend()
    fig.tight_layout()
    plt.savefig(RESULTS_DIR / "com_position.png")

    # --- Forces (Fx, Fy, Fz) per foot ---
    comp_names = ['Fx', 'Fy', 'Fz']
    for foot_idx, foot_name in enumerate(FOOT_NAMES):
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
        for i in range(3):
            axes[i].plot(t_log, f_ref_log[:, foot_idx, i], label='ref')
            axes[i].plot(t_log, f_qp_log[:, foot_idx, i], '--', label='QP')
            axes[i].axvline(t_push, linestyle=':', alpha=0.7)
            axes[i].set_ylabel(f'{comp_names[i]} [N]')
        axes[0].set_title(f'{foot_name}: force tracking (LOCAL_WORLD_ALIGNED)')
        axes[-1].set_xlabel('time [s]')
        axes[0].legend()
        fig.tight_layout()

    plt.savefig(RESULTS_DIR / "forces.png")
    plt.close()


# ================================================================
# MAIN: TAKEOFF WBIC → FLIGHT (ref ddq + joint PD)
# ================================================================
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

    # Phase split
    N_push   = 30
    N_flight = 20
    N_total  = N_push + N_flight

    if N <= N_total:
        raise RuntimeError(f"Trajectory too short: N={N}, need > {N_total}")

    # Foot frame IDs
    foot_ids = [model.getFrameId(n) for n in FOOT_NAMES]

    # Flight PD gains (joints only, milder than stance)
    Kp_flight = 150.0
    Kd_flight = 25.0

    # --- Logging containers ---
    t_log = []
    com_ref_log = []
    com_sim_log = []
    f_ref_log = []   # (T,2,6)
    f_qp_log  = []   # (T,2,6)

    cycle = 0
    print(f"\n=== cycle {cycle} : TAKEOFF (WBIC, ddq_ref+PD+force) → FLIGHT (ddq_ref+joint PD, 20 knots) ===\n")

    # Start at reference initial state
    q = q_ref_all[0].copy()
    v = v_ref_all[0].copy()
    robot.viz.display(q)
    time.sleep(1.0)

    t_sim = 0.0

    for k in range(N_total):
        h = float(dt_all[k])
        if h <= 0.0:
            continue

        q_ref_k = q_ref_all[k]
        v_ref_k = v_ref_all[k]
        if k < N - 1:
            v_ref_kp1 = v_ref_all[k + 1]
        else:
            v_ref_kp1 = v_ref_all[k]

        speed = np.linalg.norm(v)

        # precompute ref COM for this knot (for logging later)
        pin.centerOfMass(model, data, q_ref_k, v_ref_k)
        com_ref = data.com[0].copy()

        # -------------------- PUSH / TAKEOFF --------------------
        if k < N_push:
            print(
                f"[WBIC-PUSH] cycle={cycle}, k={k}/{N_total-1}, "
                f"h={h:.4f}, t_sim={t_sim:.4f}, |v|={speed:.3f}"
            )

            ddq_ref_k = generalized_ddq_ref(v_ref_k, v_ref_kp1, h, ddq_max=40.0)
            f_corner_k = f_corner_all[k]

            ddq, f_contact, f_ref_foot_k = wbic_solve_step(
                model,
                data,
                q,
                v,
                q_ref_k,
                v_ref_k,
                ddq_ref_k,
                f_corner_k,
                foot_ids,
                phase_name="WBIC-PUSH",
            )
            f_qp_foot_k = f_contact.reshape(2, 6)

            if k == N_push - 1:
                speed_ref = np.linalg.norm(v_ref_all[k])
                print(f"[END PUSH] |v_sim|={speed:.3f}, |v_ref|={speed_ref:.3f}")
        # ------------------------ FLIGHT ------------------------
        else:
            print(
                f"[FLIGHT-REF] cycle={cycle}, k={k}/{N_total-1}, "
                f"h={h:.4f}, t_sim={t_sim:.4f}, |v|={speed:.3f}"
            )

            ddq = generalized_ddq_ref(v_ref_k, v_ref_kp1, h, ddq_max=40.0)

            # joint PD to stick shape/orientation closer to ref in flight
            qj     = q[7:]
            vj     = v[6:]
            qj_ref = q_ref_k[7:]
            vj_ref = v_ref_k[6:]

            ddq[6:] += Kp_flight * (qj_ref - qj) + Kd_flight * (vj_ref - vj)

            # no QP forces in flight; keep NaNs for plotting
            f_ref_foot_k = np.full((2, 6), np.nan)
            f_qp_foot_k  = np.full((2, 6), np.nan)

        # Integrate (semi-implicit Euler-like)
        v = v + ddq * h
        q = pin.integrate(model, q, v * h)
        t_sim += h

        # compute simulated COM
        pin.centerOfMass(model, data, q, v)
        com_sim = data.com[0].copy()

        # log everything
        t_log.append(t_sim)
        com_ref_log.append(com_ref)
        com_sim_log.append(com_sim)
        f_ref_log.append(f_ref_foot_k)
        f_qp_log.append(f_qp_foot_k)

        # Safety check
        if (not np.all(np.isfinite(q))) or (not np.all(np.isfinite(v))):
            print(f"[SIM] NaN/Inf at k={k}, t_sim={t_sim:.4f}; breaking.")
            robot.viz.display(q)
            break

        robot.viz.display(q)
        time.sleep(min(h * 4.0, 0.05))

    print(f"[SIM] Finished cycle {cycle} (or broke early).")

    # --- Plot diagnostics ---
    plot_results(t_log, com_ref_log, com_sim_log, f_ref_log, f_qp_log, N_push)


if __name__ == "__main__":
    main()
