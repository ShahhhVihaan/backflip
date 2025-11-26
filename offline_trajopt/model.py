import sys
from pathlib import Path
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer


mjcf_path = "urdf/humanoid/humanoid.xml"
model = pin.buildModelFromMJCF(mjcf_path)
data = model.createData()

try:
    visual_model = pin.buildGeomFromMJCF(model, mjcf_path, pin.GeometryType.VISUAL)
    collision_model = pin.buildGeomFromMJCF(model, mjcf_path, pin.GeometryType.COLLISION)
except AttributeError:
    print("Error: Your Pinocchio version might not support MJCF geometry parsing directly.")
    sys.exit(1)

viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=False) 
viz.loadViewerModel()
q0 = pin.neutral(model) 
viz.display(q0)
print("Server is running at http://127.0.0.1:7000/static/")
print("Press Enter to exit...")
input()  # to keep running