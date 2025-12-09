#!/usr/bin/env python3
import os
import time
import pickle
from pathlib import Path

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper


def load_robot():
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

def compute_phase_times(t, N_push=30, N_flight=20, N_land=30):
    N_total = len(t)
    assert N_total == N_push + N_flight + N_land, "Phase counts don't match total knots"

    idx_push_end = N_push - 1
    idx_flight_end = N_push + N_flight - 1
    idx_land_end = N_total - 1  # or N_push + N_flight + N_land - 1

    T_push = float(t[idx_push_end] - t[0])
    T_flight = float(t[idx_flight_end] - t[idx_push_end])
    T_land = float(t[idx_land_end] - t[idx_flight_end])
    T_total = float(t[idx_land_end] - t[0])

    print(f"Phase times (from actual non-uniform t):")
    print(f"Push (0 -> {idx_push_end}) : {T_push:.6f} s")
    print(f"Flight ({idx_push_end} -> {idx_flight_end}) : {T_flight:.6f} s")
    print(f"Land ({idx_flight_end} -> {idx_land_end}) : {T_land:.6f} s")
    print(f"Total : {T_total:.6f} s")

    return T_push, T_flight, T_land, T_total



def load_trajectory():
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    results_dir = project_root / "results"

    pkl_path = results_dir / "trajectory_solution.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Could not find {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # "t": (N,), "q": (N, nq), "v": (N, nv), "dt": (N-1,), ...
    t = np.asarray(data["t"])
    q = np.asarray(data["q"])
    dt = np.asarray(data["dt"])

    return t, q, dt


def play_trajectory_raw(robot, t, q, loop=True):
    model = robot.model
    viz = robot.viz

    t = np.asarray(t, dtype=float)
    t = t - t[0]
    N = len(t)

    while True:
        for k in range(N):
            viz.display(q[k])
            if k < N - 1:
                dt = max(0.0, float(t[k + 1] - t[k]))
                time.sleep(dt)
        if not loop:
            break
        time.sleep(1.0)


def play_trajectory(robot, t, q, fps=60.0, loop=True):
    # Visualize the trajectory with non-uniform knot times t:

    # For each segment [t[k], t[k+1]] we allocate a number of visual
    # sub-frames proportional to that segment's dt
    # within each segment we interpolate q between q[k] and q[k+1]
    # Overall average rate is ~fps, but per-segment sampling adapts to dt[k]
    model = robot.model
    viz = robot.viz

    # Ensure t starts at 0
    t = np.asarray(t, dtype=float)
    t = t - t[0]

    N = len(t)
    if N < 2:
        raise ValueError("Need at least 2 states in the trajectory to animate.")

    T_total = float(t[-1])
    print(f"Total duration: {T_total:.3f} s, knots: {N}, target FPS: {fps}")

    dt_target = 1.0 / fps

    while True:
        for k in range(N - 1):
            q_k = q[k]
            q_kp1 = q[k + 1]

            dt_seg = float(t[k + 1] - t[k])
            # How many frames to show in this segment?
            n_sub = max(1, int(round(dt_seg / dt_target)))

            # Interpolate within the segment
            for i in range(n_sub):
                alpha = (i + 1) / n_sub  # from (0, 1]
                q_interp = pin.interpolate(model, q_k, q_kp1, float(alpha))
                viz.display(q_interp)
                time.sleep(dt_target)

        if not loop:
            break

        time.sleep(1.0)

def main():
    robot = load_robot()
    t, q, dt = load_trajectory()
    compute_phase_times(t)

    play_trajectory(robot, t, q)
    # play_trajectory_raw(robot, t, q)

if __name__ == "__main__":
    main()
