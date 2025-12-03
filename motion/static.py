#!/usr/bin/env python3
import time
import mujoco
import mujoco.viewer
import numpy as np

XML_PATH = "humanoid.xml"

def main():
    # Load model and create data
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # (Optional) tweak initial configuration here
    # data.qpos[:] = data.qpos  # default pose
    # e.g. tiny perturbation:
    # data.qpos[2] += 0.1

    # Compute derived quantities at this state
    # data.qpos[2] = 0.8805893739847979   # try ~0.8–1.0 and tune

    mujoco.mj_forward(model, data)
    

    # Launch passive viewer (no simulation stepping unless you do it)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer running. Close the window to exit.")
        while viewer.is_running():
            # If you *don’t* want it to move at all, just sync:
            viewer.sync()
            time.sleep(0.01)

            # If instead you want it to actually simulate, comment the two lines above
            # and use these:
            # mujoco.mj_step(model, data)
            print(f"pose: {data.qpos[2]}")  # print current pose
            viewer.sync()

if __name__ == "__main__":
    main()
