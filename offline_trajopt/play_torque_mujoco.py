#!/usr/bin/env python3
import os
import time
import pickle
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

# ---- Paths ----
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"

PKL_PATH = RESULTS_DIR / "trajectory_solution.pkl"
XML_PATH = PROJECT_ROOT / "urdf/g1_description/g1_23dof.xml"


# These are the *actuated* joints in the order used by Pinocchio q[7:], v[6:].
# (Free-flyer is q[0:7], v[0:6].)
JOINT_NAMES_PIN_ORDER = [
    # Left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist
    "waist_yaw_joint",
    # Left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    # Right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]
N_ACT_JOINTS = len(JOINT_NAMES_PIN_ORDER)  # should be 23


# ---------- Load trajectory (no tau) ----------

def load_traj():
    if not PKL_PATH.exists():
        raise FileNotFoundError(f"Missing trajectory file: {PKL_PATH}")

    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    # Expecting:
    # "t": (N,), "q": (N, nq_pin), "v": (N, nv_pin), "dt": (N-1,), ...
    t = np.asarray(data["t"])
    q = np.asarray(data["q"])
    v = np.asarray(data["v"])
    dt = np.asarray(data["dt"])

    assert len(t) == q.shape[0] == v.shape[0], "Inconsistent t/q/v lengths"
    assert len(dt) == len(t) - 1, "dt should be length N-1"

    # Quick sanity:
    assert q.shape[1] == 7 + N_ACT_JOINTS, "q dimension doesn't match expected joint count"
    assert v.shape[1] == 6 + N_ACT_JOINTS, "v dimension doesn't match expected joint count"

    return t, q, v, dt


# ---------- Mapping between Pinocchio and MuJoCo (name-based) ----------

def build_joint_maps(model: mujoco.MjModel):
    """
    Build maps from:
      - Pinocchio joint index (0..N_ACT_JOINTS-1)
      to
      - MuJoCo qpos index (for that hinge joint)
      - MuJoCo qvel index (for that hinge joint)
    using joint names.
    """
    from mujoco import mjtObj

    pin_to_mj_qpos = []
    pin_to_mj_qvel = []

    for j_name in JOINT_NAMES_PIN_ORDER:
        j_id = mujoco.mj_name2id(model, mjtObj.mjOBJ_JOINT, j_name)
        if j_id < 0:
            raise ValueError(f"Joint '{j_name}' not found in MuJoCo model")

        qpos_adr = model.jnt_qposadr[j_id]  # index in qpos
        qvel_adr = model.jnt_dofadr[j_id]   # index in qvel

        pin_to_mj_qpos.append(int(qpos_adr))
        pin_to_mj_qvel.append(int(qvel_adr))

    return np.array(pin_to_mj_qpos, dtype=int), np.array(pin_to_mj_qvel, dtype=int)


def pin_to_mj_q(q_pin: np.ndarray, model: mujoco.MjModel,
                pin_to_mj_qpos: np.ndarray) -> np.ndarray:
    """
    Map Pinocchio q -> MuJoCo qpos for free-flyer + all 1-DoF joints,
    using name-based mapping for joints.
    """
    q_mj = np.zeros(model.nq)

    # ---- Free base ----
    # Pinocchio: [x, y, z, qx, qy, qz, qw]
    x, y, z = q_pin[0:3]
    qx, qy, qz, qw = q_pin[3:7]

    # MuJoCo free qpos: [x, y, z, qw, qx, qy, qz]
    q_mj[0:7] = np.array([x, y, z, qw, qx, qy, qz])

    # ---- Joints: q_pin[7:] in JOINT_NAMES_PIN_ORDER ----
    q_joints_pin = q_pin[7:]  # shape (N_ACT_JOINTS,)

    assert q_joints_pin.shape[0] == N_ACT_JOINTS

    # For each actuated joint j, put q_joints_pin[j] into the correct qpos index
    for j in range(N_ACT_JOINTS):
        adr = pin_to_mj_qpos[j]
        q_mj[adr] = q_joints_pin[j]

    return q_mj


def pin_to_mj_v(v_pin: np.ndarray, model: mujoco.MjModel,
                pin_to_mj_qvel: np.ndarray) -> np.ndarray:
    """
    Map Pinocchio v -> MuJoCo qvel for free-flyer + all 1-DoF joints,
    using name-based mapping for joints.

    Pinocchio free velocity: [wx, wy, wz, vx, vy, vz]
    MuJoCo free qvel:        [vx, vy, vz, wx, wy, wz]
    """
    v_mj = np.zeros(model.nv)

    # ---- Free base ----
    wx, wy, wz, vx, vy, vz = v_pin[0:6]
    # MuJoCo: [vx, vy, vz, wx, wy, wz]
    v_mj[0:6] = np.array([vx, vy, vz, wx, wy, wz])

    # ---- Joints ----
    v_joints_pin = v_pin[6:]
    assert v_joints_pin.shape[0] == N_ACT_JOINTS

    for j in range(N_ACT_JOINTS):
        adr = pin_to_mj_qvel[j]
        v_mj[adr] = v_joints_pin[j]

    return v_mj


# ---------- Interpolation helpers ----------

def find_segment(t_knots, t_now, k_hint):
    """
    Incremental search: find index k such that t[k] <= t_now < t[k+1].
    k_hint is the previous k, used to avoid scanning from 0 every time.
    """
    N = len(t_knots)
    k = k_hint
    while k < N - 2 and t_now > t_knots[k + 1]:
        k += 1
    return k


def interpolate_vec(v0, v1, alpha):
    return (1.0 - alpha) * v0 + alpha * v1


# ---------- Main PD playback ----------

def main():
    # Load traj (Pinocchio frame)
    t_knots, q_pin, v_pin, dt_knots = load_traj()
    T_total = float(t_knots[-1])

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    sim_dt = model.opt.timestep  # MuJoCo integration dt
    print(f"MuJoCo dt: {sim_dt:.6f} s, traj duration: {T_total:.3f} s")

    # Build name-based maps
    pin_to_mj_qpos, pin_to_mj_qvel = build_joint_maps(model)

    # Map initial state
    q0_mj = pin_to_mj_q(q_pin[0], model, pin_to_mj_qpos)
    v0_mj = pin_to_mj_v(v_pin[0], model, pin_to_mj_qvel)

    print("Pin nq:", q_pin.shape[1])
    print("MJ  nq:", model.nq)
    print("Pin nv:", v_pin.shape[1])
    print("MJ  nv:", model.nv)
    print("MuJoCo nu (ctrl dim):", model.nu)

    if model.nu == 0:
        print("\nWARNING: model.nu == 0 ⇒ no actuators. "
              "Add <motor> actuators to your XML or controls will do nothing.\n")

    assert len(q0_mj) == model.nq, "q mapping mismatch with model.nq"
    assert len(v0_mj) == model.nv, "v mapping mismatch with model.nv"

    mujoco.mj_resetData(model, data)
    data.qpos[:] = q0_mj
    data.qvel[:] = v0_mj
    mujoco.mj_forward(model, data)

    N = len(t_knots)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_time = 0.0
        k = 0  # current segment index
        last_wall = time.time()

        # PD gains (can tune)
        Kp = 500.0
        Kd = 20.0

        while viewer.is_running():
            now = time.time()
            elapsed = now - last_wall
            last_wall = now

            # Step sim multiple times per GUI frame if needed
            n_steps = max(1, int(elapsed / sim_dt))
            for _ in range(n_steps):
                if sim_time < T_total:
                    # Find current segment [t_k, t_{k+1}]
                    k = find_segment(t_knots, sim_time, k)
                    t0 = float(t_knots[k])
                    t1 = float(t_knots[k + 1])

                    if t1 <= t0:
                        alpha = 0.0
                    else:
                        alpha = (sim_time - t0) / (t1 - t0)
                        alpha = max(0.0, min(1.0, alpha))

                    # Desired q, v (Pinocchio) interpolated in time
                    qd_pin = interpolate_vec(q_pin[k], q_pin[k + 1], alpha)
                    vd_pin = interpolate_vec(v_pin[k], v_pin[k + 1], alpha)

                    # Map to MuJoCo
                    qd_mj = pin_to_mj_q(qd_pin, model, pin_to_mj_qpos)
                    vd_mj = pin_to_mj_v(vd_pin, model, pin_to_mj_qvel)

                    # Current state
                    q_mj = data.qpos.copy()
                    v_mj = data.qvel.copy()

                    # PD tracking on joints (we won't touch free base here)
                    q_err = qd_mj - q_mj
                    v_err = vd_mj - v_mj

                    # Joint qpos/qvel indices for actuated joints:
                    #  - qpos indices: pin_to_mj_qpos
                    #  - qvel indices: pin_to_mj_qvel
                    q_err_joints = q_err[pin_to_mj_qpos]
                    v_err_joints = v_err[pin_to_mj_qvel]

                    # Control vector
                    u = np.zeros(model.nu)

                    # We assume 1 motor per actuated joint in the same order as JOINT_NAMES_PIN_ORDER
                    n_act = min(model.nu, N_ACT_JOINTS)
                    u[:n_act] = Kp * q_err_joints[:n_act] + Kd * v_err_joints[:n_act]

                    data.ctrl[:model.nu] = u

                mujoco.mj_step(model, data)
                sim_time += sim_dt

            viewer.sync()


if __name__ == "__main__":
    main()
