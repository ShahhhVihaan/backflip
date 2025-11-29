from functools import partial
from IPython.display import SVG, display
import pydot
import time
import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    AddUnitQuaternionConstraintOnPlant,
    AutoDiffXd,
    BasicVector,
    DiagramBuilder,
    ExtractGradient,
    ExtractValue,
    InitializeAutoDiff,
    JacobianWrtVariable,
    JointIndex,
    LeafSystem,
    MathematicalProgram,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    OrientationConstraint,
    Parser,
    PidController,
    PiecewisePolynomial,
    PositionConstraint,
    RotationMatrix,
    Simulator,
    SnoptSolver,
    StartMeshcat,
    Solve,
    eq,
    namedview,
    HalfSpace,
    CoulombFriction
)

print("Successfully loaded all necessary imports.")