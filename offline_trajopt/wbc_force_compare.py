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
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Paths (from your project layout)
# ------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
URDF_PATH    = PROJECT_ROOT / "urdf" / "g1_description" / "g1_23dof.urdf"
RESULTS_DIR  = PROJECT_ROOT / "results"

# NOTE: change this to "trajectory_solution_flip.pkl" if you want
# to use that file instead.
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

# Extra penalty on horizontal forces (Fx) during stance
FX_PENALTY = 2e-1  # tune this as needed


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
    w_ddq_base=1e-4,
    w_ddq_joints=5e-2,   # stronger joint ddq tracking
    w_f_lin=1e-4,        # small wrench tracking
    w_f_ang=1e-5,
    phase_name="WBIC",
    use_force_tracking=True,
):
    """
    WBIC-style QP:
      variables: x = [ddq (nv); fL (6); fR (6)]

    Cost:
      - track ddq_des = ddq_ref_k + joint-PD(q,v)
      - optionally track net foot wrenches from optimizer
      - EXTRA penalty on Fx magnitude to reduce horizontal drift

    Constraints:
      - floating-base dynamics
      - contact kinematics: Jc ddq + Jdotv = 0
      - friction pyramid + unilateral
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

    # stance gains (stronger)
    Kp = 600.0
    Kd = 80.0

    ddq_des[6:] += Kp * (qj_ref - qj) + Kd * (vj_ref - vj)

    # Cost weights
    w_ddq = np.ones(nv) * w_ddq_joints
    w_ddq[:6] = w_ddq_base

    # Force tracking from corner forces
    if use_force_tracking and (f_corner_k is not None):
        f_ref_foot = aggregate_foot_wrenches(model, data, q_ref_k, f_corner_k, foot_ids)
        f_ref = f_ref_foot.reshape(-1)   # (12,)
        w_f = np.zeros(nf)
        for foot_idx in range(nc):
            base = 6 * foot_idx
            # linear components
            w_f[base + 0] = w_f_lin   # Fx
            w_f[base + 1] = w_f_lin   # Fy
            w_f[base + 2] = w_f_lin   # Fz
            # angular components
            w_f[base + 3] = w_f_ang
            w_f[base + 4] = w_f_ang
            w_f[base + 5] = w_f_ang
        # zero out Fx reference so we don't try to match nonzero Fx
        for foot_idx in range(nc):
            idx_fx = 6 * foot_idx + 0
            f_ref[idx_fx] = 0.0
    else:
        f_ref = np.zeros(nf)
        w_f   = np.zeros(nf)

    # Quadratic cost
    P_diag = np.concatenate([w_ddq, w_f])

    # EXTRA penalty on Fx components of contact wrenches
    for foot_idx in range(nc):
        fx_var_idx = nv + 6 * foot_idx + 0  # Fx index for this foot in x
        P_diag[fx_var_idx] += FX_PENALTY

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

    return ddq, f_contact


# ================================================================
# MAIN: STANCE PHASE + FORCE PLOTS (FIRST ITERATION)
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

    # Stance length we care about
    N_push = 30
    if N <= N_push:
        raise RuntimeError(f"Trajectory too short: N={N}, need > {N_push}")

    # Foot frame IDs
    foot_ids = [model.getFrameId(n) for n in FOOT_NAMES]

    # Storage for plotting
    time_push = []

    # [foot][component] -> list
    qp_forces = {
        0: {"Fx": [], "Fy": [], "Fz": []},  # left
        1: {"Fx": [], "Fy": [], "Fz": []},  # right
    }
    ref_forces = {
        0: {"Fx": [], "Fy": [], "Fz": []},
        1: {"Fx": [], "Fy": [], "Fz": []},
    }

    # Initial sim state
    q = q_ref_all[0].copy()
    v = v_ref_all[0].copy()
    robot.viz.display(q)
    time.sleep(1.0)

    t_sim = 0.0

    print("\n=== First iteration: STANCE ONLY (0 .. N_push-1) ===\n")

    for k in range(N_push):
        h = float(dt_all[k])
        if h <= 0.0:
            continue

        q_ref_k = q_ref_all[k]
        v_ref_k = v_ref_all[k]
        if k < N - 1:
            v_ref_kp1 = v_ref_all[k + 1]
        else:
            v_ref_kp1 = v_ref_all[k]

        ddq_ref_k = generalized_ddq_ref(v_ref_k, v_ref_kp1, h, ddq_max=40.0)
        f_corner_k = f_corner_all[k]

        print(
            f"[WBIC-PUSH] k={k}/{N_push-1}, h={h:.4f}, t_sim={t_sim:.4f}, "
            f"|v|={np.linalg.norm(v):.3f}"
        )

        ddq, f_contact = wbic_solve_step(
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

        # ---- LOG FORCES (QP) ----
        f_qp = f_contact.reshape(2, 6)  # (2 feet, 6 components)
        # ---- LOG REFERENCE FORCES (trajectory) ----
        f_ref_foot = aggregate_foot_wrenches(model, data, q_ref_k, f_corner_k, foot_ids)

        # use cumulative sim time as x-axis
        time_push.append(t_sim)

        for foot_idx in [0, 1]:
            Fx_qp, Fy_qp, Fz_qp = f_qp[foot_idx, 0:3]
            Fx_ref, Fy_ref, Fz_ref = f_ref_foot[foot_idx, 0:3]

            qp_forces[foot_idx]["Fx"].append(Fx_qp)
            qp_forces[foot_idx]["Fy"].append(Fy_qp)
            qp_forces[foot_idx]["Fz"].append(Fz_qp)

            ref_forces[foot_idx]["Fx"].append(Fx_ref)
            ref_forces[foot_idx]["Fy"].append(Fy_ref)
            ref_forces[foot_idx]["Fz"].append(Fz_ref)

        # Integrate (semi-implicit Euler-like)
        v = v + ddq * h
        q = pin.integrate(model, q, v * h)
        t_sim += h

        # Safety check
        if (not np.all(np.isfinite(q))) or (not np.all(np.isfinite(v))):
            print(f"[SIM] NaN/Inf at k={k}, t_sim={t_sim:.4f}; breaking.")
            robot.viz.display(q)
            break

        robot.viz.display(q)
        time.sleep(min(h * 4.0, 0.05))

    print(f"\n[SIM] Finished first stance iteration up to k={k}.\n")

    # ============================================================
    # PLOTTING: QP vs REFERENCE, PER FOOT
    # ============================================================
    time_push_arr = np.array(time_push)

    foot_labels = {0: "Left foot", 1: "Right foot"}
    components = ["Fx", "Fy", "Fz"]

    for foot_idx in [0, 1]:
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
        fig.suptitle(f"Foot {foot_labels[foot_idx]} – push phase forces")

        for i, comp in enumerate(components):
            ax = axs[i]
            ax.plot(time_push_arr, qp_forces[foot_idx][comp], label=f"QP {comp}")
            ax.plot(time_push_arr, ref_forces[foot_idx][comp], linestyle="--", label=f"Ref {comp}")
            ax.set_ylabel(comp)
            ax.grid(True)
            ax.legend(loc="best")

        axs[-1].set_xlabel("time [s]")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"wbic_push_foot{foot_idx}_forces.png")
        plt.close()


if __name__ == "__main__":
    main()
