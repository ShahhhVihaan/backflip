import numpy as np
from pathlib import Path

from pydrake.all import (
    DirectCollocation,
    JointIndex,
    MultibodyPlant,
    Parser,
    SceneGraph,
    Solve,
    MathematicalProgram,
)


def load_g1_robot(project_root: Path):
    """Load the G1 robot model into a MultibodyPlant."""
    robot_pkg = project_root / "urdf" / "g1_description"
    model_file = robot_pkg / "g1_23dof.urdf"
    
    # Create plant
    plant = MultibodyPlant(time_step=0.0)  # Continuous time
    scene_graph = SceneGraph()
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    
    # Create parser and configure package map
    parser = Parser(plant)
    pm = parser.package_map()
    pkg_xml = robot_pkg / "package.xml"
    if pkg_xml.exists():
        pm.AddPackageXml(str(pkg_xml))
    else:
        pm.Add("g1_description", str(robot_pkg))
    
    # Load model
    parser.AddModels(str(model_file))
    plant.Finalize()
    
    return plant


def setup_trajectory_optimization(
    plant: MultibodyPlant,
    num_knot_points: int = 50,
    duration: float = 1.0,
):
    """
    Set up a direct collocation trajectory optimization problem.
    
    Args:
        plant: MultibodyPlant with the robot model
        num_knot_points: Number of knot points in the trajectory
        duration: Total duration of the trajectory (seconds)
    
    Returns:
        dircol: DirectCollocation object with the optimization problem
    """
    # Create direct collocation problem
    dircol = DirectCollocation(
        plant,
        plant.CreateDefaultContext(),
        num_time_samples=num_knot_points,
        minimum_timestep=0.01,
        maximum_timestep=duration / (num_knot_points - 1),
    )
    
    # Get state and input dimensions
    num_states = plant.num_positions() + plant.num_velocities()
    num_inputs = plant.num_actuators()
    
    # Get state and input variables
    u = dircol.input()
    x = dircol.state()
    
    # Add dynamics constraints at each knot point
    # DirectCollocation automatically enforces dynamics constraints
    # using the collocation method (Hermite-Simpson or trapezoidal)
    
    # Add initial state constraints (example: start from a specific pose)
    # You can modify these based on your requirements
    initial_state = dircol.initial_state()
    # Example: fix initial positions and velocities
    # dircol.prog().AddBoundingBoxConstraint(initial_pos, initial_pos, initial_state[:num_positions])
    # dircol.prog().AddBoundingBoxConstraint(initial_vel, initial_vel, initial_state[num_positions:])
    
    # Add final state constraints (example: end at a specific pose)
    final_state = dircol.final_state()
    # Example: fix final positions and velocities
    # dircol.prog().AddBoundingBoxConstraint(final_pos, final_pos, final_state[:num_positions])
    # dircol.prog().AddBoundingBoxConstraint(final_vel, final_vel, final_state[num_positions:])
    
    # Add input constraints (torque limits)
    # Example: limit torques to reasonable values
    torque_limit = 100.0  # Adjust based on your robot
    dircol.prog().AddBoundingBoxConstraint(
        -torque_limit * np.ones(num_inputs),
        torque_limit * np.ones(num_inputs),
        u,
    )
    
    # Add state constraints (joint limits, etc.)
    # Try to get joint limits from the plant
    num_positions = plant.num_positions()
    num_velocities = plant.num_velocities()
    
    # Get joint limits if available
    pos_lower = np.full(num_positions, -np.inf)
    pos_upper = np.full(num_positions, np.inf)
    vel_lower = np.full(num_velocities, -np.inf)
    vel_upper = np.full(num_velocities, np.inf)
    
    # Extract joint limits from plant
    # Note: This is a simplified approach
    # For a full implementation, you'd iterate through all joints
    # and map them to position indices
    try:
        # Try to get limits from joints (this is a simplified version)
        # In practice, you'd need to map joint indices to position indices
        for joint_index in range(plant.num_joints()):
            joint = plant.get_joint(JointIndex(joint_index))
            if joint.num_positions() > 0 and joint.has_limits():
                # This is simplified - you'd need proper mapping
                pass
    except:
        # If extraction fails, use defaults
        pass
    
    # If no limits found, use reasonable defaults
    if np.all(np.isinf(pos_lower)) and np.all(np.isinf(pos_upper)):
        pos_lower = -10.0 * np.ones(num_positions)
        pos_upper = 10.0 * np.ones(num_positions)
    
    if np.all(np.isinf(vel_lower)) and np.all(np.isinf(vel_upper)):
        vel_lower = -20.0 * np.ones(num_velocities)
        vel_upper = 20.0 * np.ones(num_velocities)
    
    dircol.prog().AddBoundingBoxConstraint(
        pos_lower,
        pos_upper,
        x[:num_positions],
    )
    
    dircol.prog().AddBoundingBoxConstraint(
        vel_lower,
        vel_upper,
        x[num_positions:],
    )
    
    # Add cost function (example: minimize control effort)
    # You can customize this based on your objective
    dircol.AddRunningCost(u.dot(u))  # L2 norm of control inputs
    
    # Add final cost (example: reach a target pose)
    # You can add a cost to reach a specific final state
    # final_state_error = final_state - target_state
    # dircol.AddFinalCost(final_state_error.dot(final_state_error))
    
    # Add duration constraint
    dircol.AddDurationBounds(duration, duration)
    
    return dircol


def solve_trajectory_optimization(dircol: DirectCollocation):
    """
    Solve the trajectory optimization problem.
    
    Args:
        dircol: DirectCollocation object with the optimization problem
    
    Returns:
        result: SolutionResult
        traj_state: State trajectory
        traj_input: Input trajectory
        traj_time: Time points
    """
    # Set initial guess (optional, but can help convergence)
    # You can provide an initial trajectory guess here
    
    # Solve the optimization problem
    result = Solve(dircol.prog())
    
    if not result.is_success():
        print(f"Optimization failed: {result.get_solution_result()}")
        return result, None, None, None
    
    # Extract solution
    traj_state = dircol.ReconstructStateTrajectory(result)
    traj_input = dircol.ReconstructInputTrajectory(result)
    
    # Get time points
    traj_time = np.linspace(
        traj_state.start_time(),
        traj_state.end_time(),
        dircol.num_time_samples(),
    )
    
    return result, traj_state, traj_input, traj_time


def main():
    """Main function to set up and solve trajectory optimization."""
    # Get project root
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    
    # Load G1 robot
    print("Loading G1 robot model...")
    try:
        plant = load_g1_robot(project_root)
        print(f"Loaded robot with {plant.num_positions()} positions and {plant.num_actuators()} actuators")
    except Exception as e:
        print(f"Error loading robot: {e}")
        print("Note: G1 uses STL meshes which Drake doesn't support for dynamics.")
        print("You may need to use a simplified model or convert meshes to OBJ format.")
        return
    
    # Set up trajectory optimization
    print("Setting up trajectory optimization...")
    num_knot_points = 50
    duration = 1.0
    
    dircol = setup_trajectory_optimization(
        plant,
        num_knot_points=num_knot_points,
        duration=duration,
    )
    
    print(f"Optimization problem has {dircol.prog().num_vars()} variables")
    print(f"Optimization problem has {dircol.prog().num_constraints()} constraints")
    
    # Solve
    print("Solving trajectory optimization...")
    result, traj_state, traj_input, traj_time = solve_trajectory_optimization(dircol)
    
    if result.is_success():
        print("Optimization succeeded!")
        print(f"Optimal cost: {result.get_optimal_cost()}")
        print(f"Trajectory duration: {traj_state.end_time() - traj_state.start_time():.3f} s")
        
        # You can visualize or save the trajectory here
        # Example: visualize in meshcat, save to file, etc.
    else:
        print("Optimization failed. Try:")
        print("1. Adjusting initial/final state constraints")
        print("2. Relaxing bounds on inputs or states")
        print("3. Providing a better initial guess")
        print("4. Adjusting the number of knot points")


if __name__ == "__main__":
    main()

