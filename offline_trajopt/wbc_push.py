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

SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
URDF_PATH    = PROJECT_ROOT / "urdf" / "g1_description" / "g1_23dof.urdf"
RESULTS_DIR  = PROJECT_ROOT / "results"
PKL_PATH     = RESULTS_DIR / "trajectory_solution.pkl"

FOOT_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
CORNER_OFFSETS = [
    np.array([-0.05,  0.025, -0.03]),
    np.array([-0.05, -0.025, -0.03]),
    np.array([ 0.12,  0.03,  -0.03]),
    np.array([ 0.12, -0.03,  -0.03]),
]

MU_FRICTION = 0.5

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
        # world frame
        p_frame = oMf.translation
        # world R frame ( ^W R_F )
        Rwf = oMf.rotation          

        f_total_world   = np.zeros(3)
        tau_total_world = np.zeros(3)

        for c_idx, offset in enumerate(CORNER_OFFSETS):
            f_corner = f_corner_k[foot_idx, c_idx, :]
            p_corner = p_frame + Rwf @ offset

            f_total_world   += f_corner
            tau_total_world += np.cross(p_corner - p_frame, f_corner)

        # Convert to LOCAL_WORLD_ALIGNED it is the same orientation as J in LWA
        f_lin_local  = Rwf.T @ f_total_world
        f_ang_local  = Rwf.T @ tau_total_world

        f_ref_foot[foot_idx, :3] = f_lin_local
        f_ref_foot[foot_idx, 3:] = f_ang_local

    return f_ref_foot

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
):
    nv = model.nv
    nc = 2
    nf = 6 * nc   # 6D wrench per foot

    pin.computeAllTerms(model, data, q, v)
    M = data.M.copy()
    h = pin.nonLinearEffects(model, data, q, v)  # coriolis + gravity

    # Build stacked contact Jacobian Jc (12 x nv) in LOCAL_WORLD_ALIGNED
    J_list = []
    Jdotv_list = []
    for frame_id in foot_ids:
        J = pin.computeFrameJacobian(
            model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J_list.append(J)
        # For now, approximate Jdot * v as zero (WBIC simplification for slow motions)
        Jdotv_list.append(np.zeros(6))

    Jc = np.vstack(J_list)           # (12, nv)
    Jdotv = np.concatenate(Jdotv_list)  # (12,)

    # Selection matrix for floating base (rows 0..5)
    S_f = np.zeros((6, nv))
    S_f[np.arange(6), np.arange(6)] = 1.0

    # Aggregate foot wrenches from corner forces
    f_ref_foot = aggregate_foot_wrenches(model, data, q_ref, f_corner_k, foot_ids)
    f_ref = f_ref_foot.reshape(-1)   # (12,)

    # Joint-space desired accelerations via PD on joints only
    qj      = q[7:]
    vj      = v[6:]
    qj_ref  = q_ref[7:]
    vj_ref  = v_ref[6:]

    # Simple PD on joints for ddq_des
    Kp = 100.0
    Kd = 20.0
    ddq_des = np.zeros(nv)
    ddq_des[6:] = Kp * (qj_ref - qj) + Kd * (vj_ref - vj)

    w_ddq = np.ones(nv) * w_ddq_joints
    w_ddq[:6] = w_ddq_base  # small weight on base accelerations

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

    # Quadratic cost:
    #   1/2 * (ddq - ddq_des)^T W_ddq (ddq - ddq_des)
    # + 1/2 * (f   - f_ref   )^T W_f   (f   - f_ref)
    # => P = diag([W_ddq, W_f])
    #    q_vec = [-W_ddq * ddq_des; -W_f * f_ref]
    P_diag = np.concatenate([w_ddq, w_f])
    P = sp.diags(P_diag)

    q_cost = np.concatenate([-w_ddq * ddq_des, -w_f * f_ref])

    # --- Equality constraints ---
    # Floating base dynamics:
    #    S_f (M ddq + h - Jc^T f) = 0
    # => [ S_f M   -S_f Jc^T ] [ddq; f] = - S_f h
    A_dyn = np.hstack([S_f @ M, -S_f @ Jc.T])
    b_dyn = -S_f @ h

    # Contact kinematics:
    #    Jc ddq + Jdotv = 0   (approx)
    # => [ Jc   0 ] [ddq; f] = -Jdotv
    A_kin = np.hstack([Jc, np.zeros((Jc.shape[0], nf))])
    b_kin = -Jdotv

    A_eq = np.vstack([A_dyn, A_kin])
    b_eq = np.concatenate([b_dyn, b_kin])

    # Inequality constraints: friction + unilateral per foot
    # For each foot, impose:
    #  -Fz <= 0          (Fz >= 0)
    #   Fx - mu Fz <= 0
    #  -Fx - mu Fz <= 0
    #   Fy - mu Fz <= 0
    #  -Fy - mu Fz <= 0
    A_ineq_rows = []
    u_ineq = []

    for foot_idx in range(nc):
        base = nv + 6 * foot_idx  # index in x where this foot's wrench starts
        Fx_idx = base + 0
        Fy_idx = base + 1
        Fz_idx = base + 2

        # -Fz <= 0
        row = np.zeros(nv + nf)
        row[Fz_idx] = -1.0
        A_ineq_rows.append(row)
        u_ineq.append(0.0)

        # Fx - mu Fz <= 0
        row = np.zeros(nv + nf)
        row[Fx_idx] = 1.0
        row[Fz_idx] = -MU_FRICTION
        A_ineq_rows.append(row)
        u_ineq.append(0.0)

        # -Fx - mu Fz <= 0
        row = np.zeros(nv + nf)
        row[Fx_idx] = -1.0
        row[Fz_idx] = -MU_FRICTION
        A_ineq_rows.append(row)
        u_ineq.append(0.0)

        # Fy - mu Fz <= 0
        row = np.zeros(nv + nf)
        row[Fy_idx] = 1.0
        row[Fz_idx] = -MU_FRICTION
        A_ineq_rows.append(row)
        u_ineq.append(0.0)

        # -Fy - mu Fz <= 0
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
        f"[WBIC QP] status={res.info.status}, "
        f"iters={res.info.iter}, "
        f"time={solve_time*1e3:.2f} ms"
    )

    if res.info.status_val not in (
        osqp.constant("OSQP_SOLVED"),
        osqp.constant("OSQP_SOLVED_INACCURATE"),
    ):
        print(f"[WBIC] OSQP failed with status: {res.info.status}")
        # Fallback: just use ddq_des, f_ref
        ddq = ddq_des
        f_contact = f_ref.copy()
    else:
        x_opt = res.x
        ddq = x_opt[:nv]
        f_contact = x_opt[nv:]

    return ddq, f_contact



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

    # push phase is first 30 knots change if increased
    N_push = 30

    push_indices = np.arange(N_push)
    t_push = t[push_indices]
    dt_push = dt_all[:N_push-1]

    print(f"\nPush phase: knots 0 -> {N_push-1}, duration: {t_push[-1] - t_push[0]:.6f} s")

    foot_ids = [model.getFrameId(n) for n in FOOT_NAMES]

    # Number of integration substeps per knot
    substeps = 4

    cycle = 0
    while True:
        print(f"\n=== Starting forward simulation cycle {cycle} (push phase WBIC QP) ===")

        # Reset sim state to the reference at k=0 for each cycle
        q = q_ref_all[0].copy()
        v = v_ref_all[0].copy()

        robot.viz.display(q)
        time.sleep(1.0)

        # Logs for debugging / COM plotting (for this cycle)
        times_log = [0.0]
        com_z_log = []
        t_sim = 0.0

        for k in range(N_push - 1):
            h = float(dt_push[k])
            if h <= 0.0:
                continue
            h_sub = h / substeps

            q_ref_k = q_ref_all[k]
            v_ref_k = v_ref_all[k]
            f_corner_k = f_corner_all[k]  # (2,4,3)

            # Per-knot debug print
            print(
                f"[WBIC] cycle={cycle}, knot={k}/{N_push-1}, "
                f"h={h:.4f}, t_sim={t_sim:.4f}"
            )

            # Solve WBIC QP at this knot
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
            )

            # Integrate with semi-implicit Euler using ddq from WBIC
            for s in range(substeps):
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
                    print(f"\n[SIM] NaN/Inf encountered at t={t_sim:.4f}")
                    print("  |q[0:3]|:", np.linalg.norm(q[0:3]))
                    print("  |v|:", np.linalg.norm(v))
                    robot.viz.display(q)
                    break

            robot.viz.display(q)
            # Slow down viz so we can see it
            time.sleep(min(h * 5.0, 0.1))

            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
                break

        print("\n[SIM] Push phase forward simulation finished (or stopped).")

        # Save COM plot only for the first cycle to avoid spamming the disk
        if cycle == 0 and len(com_z_log) > 0:
            times_log = np.array(times_log)
            com_z_log = np.array(com_z_log)

            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            plt.figure()
            plt.plot(times_log[:len(com_z_log)], com_z_log)
            plt.xlabel("time [s]")
            plt.ylabel("COM z [m]")
            plt.grid(True)
            plt.title("COM height during WBIC push phase")
            plt.savefig(RESULTS_DIR / "wbic_push_com_z.png", dpi=200)
            plt.close()

            print("Saved WBIC COM plot to:", RESULTS_DIR / "wbic_push_com_z.png")

        cycle += 1
        # loop back and repeat the push phase again



if __name__ == "__main__":
    main()
