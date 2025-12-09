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
GRAVITY = 9.81


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
# HELPER: COM PITCH TORQUE FROM CONTACT WRENCHES
# ================================================================
def compute_pitch_torque_about_com(model, data, q, v, f_contact, foot_ids):
    """
    Compute net sagittal-plane torque about the COM from the *actual* contact
    wrenches in f_contact (shape (12,): [FL(6), FR(6)] in LOCAL_WORLD_ALIGNED).

    Returns:
        tau_y_com : scalar, net pitch torque about COM in WORLD frame
    """
    # Update kinematics + COM
    pin.forwardKinematics(model, data, q, v)
    pin.updateFramePlacements(model, data)
    pin.centerOfMass(model, data, q, v)
    com = data.com[0].copy()   # (3,)

    tau_com_world = np.zeros(3)

    for foot_idx, frame_id in enumerate(foot_ids):
        oMf = data.oMf[frame_id]
        p_foot = oMf.translation       # world position of foot frame
        Rwf = oMf.rotation             # world R foot

        base = 6 * foot_idx
        f_lin_local = f_contact[base : base + 3]
        f_ang_local = f_contact[base + 3 : base + 6]

        # Convert to world
        F_world = Rwf @ f_lin_local
        tau_foot_world = Rwf @ f_ang_local

        # torque about COM: tau_foot + (r × F)
        r = p_foot - com
        tau_world = tau_foot_world + np.cross(r, F_world)

        tau_com_world += tau_world

    # Return the pitch (y) component
    return tau_com_world[1]


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
# LUMPED PARAMS + STATE EXTRACTION FOR LANDING MPC
# ================================================================
def estimate_lumped_params(model, q0):
    """
    Estimate:
      - total mass
      - effective pitch inertia Iy at COM in the nominal pose q0
    """
    m = pin.computeTotalMass(model)

    data = model.createData()
    v0 = np.zeros(model.nv)

    pin.centerOfMass(model, data, q0, v0)
    pin.ccrba(model, data, q0, v0)  # centroidal composite rigid body algo
    Ig = data.Ig  # spatial inertia at COM (6x6)

    # Effective pitch inertia (around y-axis in world frame) ~ element (1,1) of rotational block
    I_rot = Ig.inertia  # 3x3
    Iy = I_rot[1, 1]

    return m, Iy


def quat_to_pitch(q_base):
    """
    Extract a simple pitch angle (rotation about world y) from a base quaternion.
    q_base: [x, y, z, w]
    Using a standard XYZ convention; for small angles this is fine.
    """
    x, y, z, w = q_base
    # Rotation matrix
    R = pin.Quaternion(w, x, y, z).toRotationMatrix()
    # One common convention for pitch (about y):
    # pitch = atan2(-R[2,0], sqrt(R[0,0]^2 + R[1,0]^2))
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    return pitch


def extract_com_pitch_state(model, data, q, v):
    """
    Build a 6D lumped state:
        x = [ x_COM, z_COM, pitch, vx_COM, vz_COM, omega_y ]
    """
    # COM + COM velocity
    pin.centerOfMass(model, data, q, v)
    com = data.com[0].copy()
    vcom = data.vcom[0].copy()

    # Base quaternion is in q[3:7] for free-flyer
    q_base = q[3:7]
    pitch = quat_to_pitch(q_base)

    # Free-flyer twist: v[0:3] angular, v[3:6] linear in world frame
    omega_base = v[0:3]

    x = np.zeros(6)
    x[0] = com[0]           # x
    x[1] = com[2]           # z (height)
    x[2] = pitch            # pitch
    x[3] = vcom[0]          # vx
    x[4] = vcom[2]          # vz
    x[5] = omega_base[1]    # omega_y

    return x


# ================================================================
# MINI LANDING "MPC" QP (single-step)
# ================================================================
def landing_mpc_qp(x_cur, x_ref, m, Iy, h):
    """
    Single-step QP on lumped model:

      state: x = [x, z, pitch, vx, vz, omega_y]
      input: u = [Fx, Fz, Ty]

    Dynamics (discrete, forward Euler):
      x_next = x + h * [vx,
                        vz,
                        omega_y,
                        Fx/m,
                        Fz/m - g,
                        Ty/Iy]

    Cost:
      0.5 * (x_next - x_ref)^T Q (x_next - x_ref) + 0.5 * u^T R u

    Returns:
      u_opt = [Fx, Fz, Ty]
    """
    # Tuning
    Q = np.diag([
        1.0,   # x
        10.0,  # z
        50.0,  # pitch
        0.1,   # vx
        5.0,   # vz
        20.0,  # omega_y
    ])
    R = np.diag([
        1e-3,  # Fx
        1e-3,  # Fz
        1e-4,  # Ty
    ])

    # Build discrete A, B, and gravity bias
    A = np.eye(6)
    # x, z, pitch
    A[0, 3] = h      # x_next depends on vx
    A[1, 4] = h      # z_next depends on vz
    A[2, 5] = h      # pitch_next depends on omega_y
    # velocities are identity
    # A[3,3] = 1, A[4,4]=1, A[5,5]=1 already

    B = np.zeros((6, 3))
    B[3, 0] = h / m       # vx_next: Fx
    B[4, 1] = h / m       # vz_next: Fz
    B[5, 2] = h / Iy      # omega_next: Ty

    g_bias = np.zeros(6)
    g_bias[4] = -GRAVITY * h  # vz term

    # x_next - x_ref = (A x_cur + g_bias - x_ref) + B u
    e = A @ x_cur + g_bias - x_ref

    # QP: 0.5 u^T P u + q^T u
    P = B.T @ Q @ B + R
    q = B.T @ Q @ e

    # Bounds on u = [Fx, Fz, Ty]
    # (simple box constraints for now)
    Fz_max = 4.0 * m * GRAVITY
    Fx_max = MU_FRICTION * Fz_max * 0.5   # somewhat conservative
    Ty_max = 0.5 * m * GRAVITY * 0.5      # very rough guess

    u_lower = np.array([-Fx_max, 0.0,      -Ty_max])
    u_upper = np.array([ Fx_max,  Fz_max,   Ty_max])

    # Encode bounds as I * u between lower/upper
    A_ineq = np.eye(3)
    l_ineq = u_lower
    u_ineq = u_upper

    prob = osqp.OSQP()
    prob.setup(
        P=sp.csc_matrix(P),
        q=q,
        A=sp.csc_matrix(A_ineq),
        l=l_ineq,
        u=u_ineq,
        verbose=False,
        polish=True,
        eps_abs=1e-6,
        eps_rel=1e-6,
        max_iter=10000,
    )
    res = prob.solve()

    if res.info.status_val not in (
        osqp.constant("OSQP_SOLVED"),
        osqp.constant("OSQP_SOLVED_INACCURATE"),
    ):
        # Fallback: simple PD-like heuristic if QP fails
        # Try to kill velocities & pitch
        Fx = -50.0 * x_cur[3]   # brake vx
        Fz = max(0.0, m * GRAVITY - 50.0 * x_cur[4])
        Ty = -50.0 * x_cur[2] - 5.0 * x_cur[5]
        u_opt = np.array([Fx, Fz, Ty])
    else:
        u_opt = res.x

    return u_opt  # [Fx, Fz, Ty]


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
    w_ddq_joints=1e-2,   # slightly softer than before
    # force tracking weights (axis-specific)
    w_fx=5e-2,
    w_fy=5e-2,
    w_fz=1e-3,
    w_tau=1e-4,
    phase_name="WBIC",
    use_force_tracking=True,
    f_ref_override=None,
):
    """
    WBIC-style QP:
      variables: x = [ddq (nv); fL (6); fR (6)]

    Cost:
      - track ddq_des = ddq_ref_k + joint-PD(q,v)
      - track net foot wrenches (Fx, Fy, Fz, Torques)

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

    # Cost weights on ddq
    w_ddq = np.ones(nv) * w_ddq_joints

    # Base indices (free-flyer): 0:3 translational, 3:6 rotational
    w_base_trans = 5e2
    w_base_rot   = 5e4

    w_ddq[0:3] = w_base_trans
    w_ddq[3:6] = w_base_rot

    # ------------------------------------------------------------------
    # Force tracking target f_ref (12,)
    # ------------------------------------------------------------------
    if use_force_tracking:
        if f_ref_override is not None:
            # Use MPC-provided desired foot wrenches (2,6)
            f_ref_foot = f_ref_override  # shape (2,6)
        elif f_corner_k is not None:
            # Use offline centroidal optimizer corner forces
            f_ref_foot = aggregate_foot_wrenches(
                model, data, q_ref_k, f_corner_k, foot_ids
            )
        else:
            f_ref_foot = np.zeros((2, 6))
        f_ref = f_ref_foot.reshape(-1)

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

    return ddq, f_contact


# ================================================================
# MAIN: PUSH WBIC → FLIGHT (ref ddq + joint PD) → LAND (WBIC+MPC)
# ================================================================
def main():
    robot = load_robot()
    model = robot.model
    data  = model.createData()

    t, q_ref_all, v_ref_all, dt_all, f_corner_all, meta = load_traj()
    N      = q_ref_all.shape[0]    # number of states
    N_dt   = dt_all.shape[0]       # = N - 1, number of integration steps

    print("Trajectory shapes:")
    print("  t:", t.shape)
    print("  q:", q_ref_all.shape)
    print("  v:", v_ref_all.shape)
    print("  dt:", dt_all.shape)

    # Lumped parameters for landing MPC
    m_lumped, Iy_lumped = estimate_lumped_params(model, q_ref_all[0])
    print(f"Lumped params: m={m_lumped:.3f}, Iy={Iy_lumped:.3f}")

    # ------------------------------------------------------------
    # Phase split: PUSH (stance) → FLIGHT → LAND (stance)
    # ------------------------------------------------------------
    N_push   = 30     # as before
    N_flight = 20     # as before

    if N_dt <= N_push + N_flight:
        raise RuntimeError(
            f"Not enough knots for landing: N_dt={N_dt}, "
            f"need > N_push+N_flight={N_push+N_flight}"
        )

    # Use *all remaining* dt steps for landing
    N_land  = N_dt - (N_push + N_flight)
    N_total = N_push + N_flight + N_land   # number of integration steps we simulate

    print(f"Phase lengths: push={N_push}, flight={N_flight}, land={N_land}, total={N_total}")

    # Foot frame IDs
    foot_ids = [model.getFrameId(n) for n in FOOT_NAMES]

    # Flight PD gains (joints only, milder than stance)
    Kp_flight = 150.0
    Kd_flight = 25.0

    cycle = 0
    while True:
        print(
            f"\n=== cycle {cycle} : "
            f"PUSH (WBIC, ddq_ref+PD+force) → "
            f"FLIGHT (ddq_ref+joint PD) → "
            f"LAND (WBIC+MPC, ddq_ref+PD+force) ===\n"
        )

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

            # ------------------------------------------------
            # 1) TAKEOFF / PUSH (stance WBIC)
            # ------------------------------------------------
            if k < N_push:
                print(
                    f"[WBIC-PUSH] cycle={cycle}, k={k}/{N_total-1}, "
                    f"h={h:.4f}, t_sim={t_sim:.4f}, |v|={speed:.3f}"
                )

                ddq_ref_k   = generalized_ddq_ref(v_ref_k, v_ref_kp1, h, ddq_max=40.0)
                f_corner_k  = f_corner_all[k]

                # phase-dependent weighting inside push
                if k < N_push - 5:
                    # early stance
                    w_ddq_joints = 1e-3
                    w_fx = 5e-2
                    w_fy = 5e-2
                    w_fz = 5e-3
                    w_tau = 5e-4
                else:
                    # last 5 push knots: sacrifice Fx/Fy tracking to avoid pitch impulse
                    w_ddq_joints = 1e-3
                    w_fx = 0.0
                    w_fy = 0.0
                    w_fz = 5e-3
                    w_tau = 1e-4

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
                    w_ddq_base=1e-4,
                    w_ddq_joints=w_ddq_joints,
                    w_fx=w_fx,
                    w_fy=w_fy,
                    w_fz=w_fz,
                    w_tau=w_tau,
                )

                # COM pitch torque diagnostic
                tau_y_com = compute_pitch_torque_about_com(
                    model, data, q, v, f_contact, foot_ids
                )
                print(f"[WBIC-PUSH] k={k}, tau_y_com={tau_y_com:.4f}")

                # Liftoff logging
                if k == N_push - 1:
                    speed_ref = np.linalg.norm(v_ref_all[k])

                    pin.centerOfMass(model, data, q, v)
                    com_sim  = data.com[0].copy()
                    vcom_sim = data.vcom[0].copy()

                    pin.centerOfMass(model, data, q_ref_all[k], v_ref_all[k])
                    com_ref  = data.com[0].copy()
                    vcom_ref = data.vcom[0].copy()

                    pin.computeCentroidalMomentum(model, data, q, v)
                    hG = data.hg

                    print(f"[END PUSH] |v_sim|   ={speed:.3f}, |v_ref|   ={speed_ref:.3f}")
                    print(f"[END PUSH] ΔCOM     ={com_sim - com_ref}")
                    print(f"[END PUSH] ΔvCOM    ={vcom_sim - vcom_ref}")
                    print(f"[END PUSH] tau_y_com_liftoff = {tau_y_com:.4f}")
                    print(f"[END PUSH] hG.angular_y      = {hG.angular[1]:.4f}")

            # ------------------------------------------------
            # 2) FLIGHT (no contact; ddq_ref + joint PD)
            # ------------------------------------------------
            elif k < N_push + N_flight:
                print(
                    f"[FLIGHT-REF] cycle={cycle}, k={k}/{N_total-1}, "
                    f"h={h:.4f}, t_sim={t_sim:.4f}, |v|={speed:.3f}"
                )

                ddq = generalized_ddq_ref(
                    v_ref_k, v_ref_kp1, h, ddq_max=40.0
                )

                # joint PD in flight
                qj     = q[7:]
                vj     = v[6:]
                qj_ref = q_ref_k[7:]
                vj_ref = v_ref_k[6:]

                ddq[6:] += Kp_flight * (qj_ref - qj) + Kd_flight * (vj_ref - vj)

                # Touchdown diagnostics at start of landing (next step)
                if k == N_push + N_flight - 1:
                    print("[FLIGHT] end of flight, next step will be LAND stance.")

            # ------------------------------------------------
            # 3) LANDING (stance WBIC + lumped-model MPC for f_ref)
            # ------------------------------------------------
            else:
                print(
                    f"[WBIC-LAND] cycle={cycle}, k={k}/{N_total-1}, "
                    f"h={h:.4f}, t_sim={t_sim:.4f}, |v|={speed:.3f}"
                )

                ddq_ref_k  = generalized_ddq_ref(v_ref_k, v_ref_kp1, h, ddq_max=40.0)

                # Build lumped state from *current* sim state
                x_cur = extract_com_pitch_state(model, data, q, v)
                # And from reference
                x_ref = extract_com_pitch_state(model, data, q_ref_k, v_ref_k)

                # Landing MPC: choose net Fx, Fz, Ty at COM
                Fx_net, Fz_net, Ty_net = landing_mpc_qp(
                    x_cur, x_ref, m_lumped, Iy_lumped, h
                )

                # Split net forces equally between two feet, ignore direct torques
                f_ref_foot = np.zeros((2, 6))
                for foot_idx in range(2):
                    f_ref_foot[foot_idx, 0] = Fx_net / 2.0   # Fx
                    f_ref_foot[foot_idx, 1] = 0.0            # Fy
                    f_ref_foot[foot_idx, 2] = Fz_net / 2.0   # Fz
                    # For now, no direct foot torques; WBIC + CoP will realize Ty_net indirectly.

                # Landing weighting: prioritize vertical support and pitch control
                w_ddq_joints = 1e-3
                w_fx = 1e-2
                w_fy = 1e-3
                w_fz = 1e-2
                w_tau = 1e-4

                ddq, f_contact = wbic_solve_step(
                    model,
                    data,
                    q,
                    v,
                    q_ref_k,
                    v_ref_k,
                    ddq_ref_k,
                    f_corner_k=None,            # ignore offline forces in landing
                    foot_ids=foot_ids,
                    phase_name="WBIC-LAND",
                    w_ddq_base=1e-4,
                    w_ddq_joints=w_ddq_joints,
                    w_fx=w_fx,
                    w_fy=w_fy,
                    w_fz=w_fz,
                    w_tau=w_tau,
                    use_force_tracking=True,
                    f_ref_override=f_ref_foot,  # MPC target
                )

                tau_y_com = compute_pitch_torque_about_com(
                    model, data, q, v, f_contact, foot_ids
                )
                print(f"[WBIC-LAND] k={k}, tau_y_com={tau_y_com:.4f}")

                # Optional: log "touchdown" metrics at first landing step
                if k == N_push + N_flight:
                    pin.computeCentroidalMomentum(model, data, q, v)
                    hG = data.hg
                    print(f"[START LAND] hG.angular_y = {hG.angular[1]:.4f}")

            # -------- integrate one step --------
            v = v + ddq * h
            q = pin.integrate(model, q, v * h)
            t_sim += h

            # safety check
            if (not np.all(np.isfinite(q))) or (not np.all(np.isfinite(v))):
                print(f"[SIM] NaN/Inf at k={k}, t_sim={t_sim:.4f}; breaking.")
                robot.viz.display(q)
                break

            robot.viz.display(q)
            time.sleep(min(h * 4.0, 0.05))

        print(f"[SIM] Finished cycle {cycle} (or broke early).")
        cycle += 1


if __name__ == "__main__":
    main()
