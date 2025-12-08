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

    play_trajectory(robot, t, q)
    # play_trajectory_raw(robot, t, q)

if __name__ == "__main__":
    main()
