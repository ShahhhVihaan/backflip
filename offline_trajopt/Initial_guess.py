from pydrake.multibody.inverse_kinematics import InverseKinematics
import numpy as np
from pydrake.math import RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve

def solve_ik_pose(plant, q_seed, target_pelvis_z, foot_frames, foot_target_positions):
    """
    Solves IK to lower pelvis to target_z while keeping feet fixed at foot_target_positions.
    """
    ik = InverseKinematics(plant)
    q_var = ik.q()
    prog = ik.get_mutable_prog()
    
    # 1. Constrain Feet (Fixed to the positions passed in)
    for i, frame in enumerate(foot_frames):
        target_pos = foot_target_positions[i]
        
        # Position: Tight tolerance (2mm) to keep feet planted
        ik.AddPositionConstraint(
            frame, [0, 0, 0],
            plant.world_frame(),
            target_pos - 0.002, 
            target_pos + 0.002
        )
        
        # Orientation: Loose tolerance (0.1 rad) to allow slight ankle adjustment
        ik.AddOrientationConstraint(
            frame, RotationMatrix(),
            plant.world_frame(), RotationMatrix(),
            0.1 
        )

    # 2. Constrain Pelvis Height ONLY (Allow X/Y to float)
    pelvis_frame = plant.GetFrameByName("pelvis")
    
    # We only constrain Z. bounds: [-inf, -inf, z_min] <= p_WP <= [inf, inf, z_max]
    ik.AddPositionConstraint(
        pelvis_frame, [0, 0, 0],
        plant.world_frame(),
        [-np.inf, -np.inf, target_pelvis_z - 0.005],
        [np.inf, np.inf, target_pelvis_z + 0.005]
    )

    # 3. Seed with the previous configuration (Warm Start)
    prog.SetInitialGuess(q_var, q_seed)
    
    result = Solve(prog)
    
    if not result.is_success():
        print(f"Warning: IK failed for height {target_pelvis_z:.3f}")
        return q_seed # Return the seed as best effort
        
    return result.GetSolution(q_var)