from offline_trajopt.imports_setup import * 
from offline_trajopt.set_home_position import set_home
from pathlib import Path

"""
Helper Fuction to compare two AutoDiffs
"""

def autoDiffArrayEqual(a,b):
    return np.array_equal(a,b) and np.array_equal(
        ExtractGradient(a), ExtractGradient(b)
    )

"""
Function for Euler Integration Contraint for Kinematics
"""

def velocity_dynamics_constraint(vars, context_index, plant, ad_plant, context, ad_context):
    nq = plant.num_positions()
    nv = plant.num_velocities()
    h, q, v, qn = np.split(vars, [1, 1 + nq, 1 + nq + nv])
    
    # Check if we are using AutoDiff
    if isinstance(vars[0], AutoDiffXd):
        curr_context = ad_context[context_index]
        curr_plant = ad_plant
    else:
        curr_context = context[context_index]
        curr_plant = plant

    curr_plant.SetPositions(curr_context, q)
    
    dt = h if isinstance(h, AutoDiffXd) else max(h, 1e-6)
    qdot = (qn - q) / dt
    v_from_qdot = curr_plant.MapQDotToVelocity(curr_context, qdot)
    
    return v - v_from_qdot



"""
Main Optimization
"""
def Crouch():

    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    # Create a diagram
    builder = DiagramBuilder()
    # Add a plant 
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0005)
    X_WG = HalfSpace.MakePose(np.array([0,0, 1]), np.zeros(3,))
    plant.RegisterCollisionGeometry(
        plant.world_body(), 
        X_WG, HalfSpace(), 
        "collision", 
        CoulombFriction(1.0, 1.0)
    )
    parser = Parser(plant)
    (humanoid_g1,) = parser.AddModels(str(project_root /"urdf/g1_description/g1_23dof.urdf"))
    plant.Finalize()


    # Add the visualizer
    meshcat = StartMeshcat()
    vis_params = MeshcatVisualizerParams(publish_period=0.01)
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params=vis_params)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    set_home(plant, plant_context)
    diagram.ForcedPublish(context)

    body_frame = plant.GetFrameByName("pelvis")

    PositionView = namedview(
        "Positions", plant.GetPositionNames(humanoid_g1, always_add_suffix=False)
    )
    
    VelocityView = namedview(
        "Velocities",
        plant.GetVelocityNames(humanoid_g1, always_add_suffix=False),
    )
    
    q0 = plant.GetPositions(plant_context)
   
    mu = 1.0   # Friction Coefficient 

    total_mass = plant.CalcTotalMass(plant_context, [humanoid_g1])  
    gravity = plant.gravity_field().gravity_vector()

    foot_frame = [
        plant.GetFrameByName("left_ankle_roll_link"),
        plant.GetFrameByName("right_ankle_roll_link"),
    ]

    contact_points = [
        [-0.05, 0.025, -0.03],
        [-0.05, -0.025, -0.03],
        [0.12, 0.03, -0.03],
        [0.12, -0.03, -0.03]
    ]
    
    
    # No of Knot points
    N = 10
    
    # Total time (sec)
    T = 0.5

    # Legs on the ground (Change for jumping)
    in_stance = np.ones((2,N))

    # initiate the Mathematical Program
    prog = MathematicalProgram()

    # Time steps
    h = prog.NewContinuousVariables(N-1, "h")
    prog.AddBoundingBoxConstraint(0.5 * T / N, 2.0 * T / N, h)
    prog.AddLinearConstraint(sum(h) == T)

    # Create one context per time step (to maximize cache hits)
    context = [plant.CreateDefaultContext() for i in range(N)]
    # We could get rid of this by implementing a few more Jacobians in MultibodyPlant:
    ad_plant = plant.ToAutoDiffXd()

    nq = plant.num_positions()
    nv = plant.num_velocities()
    q = prog.NewContinuousVariables(nq, N, "q")
    v = prog.NewContinuousVariables(nv, N, "v")     

    q_cost = PositionView([1]*nq)
    v_cost = VelocityView([1]*nv)
    #print(VelocityView([1]*nv))
    q_cost.pelvis_qw = 0.0 
    q_cost.pelvis_qx = 0.0
    q_cost.pelvis_qy = 0.0
    q_cost.pelvis_qz = 0.0
    #q_cost.pelvis_x = 0.5
    #q_cost.pelvis_y = 1.0
    q_cost.pelvis_z = 0.0
    q_cost.left_hip_roll_joint = 10.0
    q_cost.right_hip_roll_joint = 10.0
    q_cost.left_hip_yaw_joint = 10.0
    q_cost.right_hip_yaw_joint = 10.0
    q_cost.left_ankle_pitch_joint = 10.0
    q_cost.right_ankle_pitch_joint = 10.0
    q_cost.left_ankle_roll_joint = 10.0
    q_cost.right_ankle_roll_joint = 10.0

    q_crouch_vec = np.copy(q0)
    q_crouch = PositionView(q_crouch_vec)

    q_crouch.pelvis_z = 0.58
    q_crouch.left_knee_joint = 0.7
    q_crouch.right_knee_joint = 0.7

    q_crouch.left_hip_pitch_joint = -0.45
    q_crouch.right_hip_pitch_joint = -0.45

    q_crouch.left_ankle_pitch_joint = -0.25
    q_crouch.right_ankle_pitch_joint = -0.25

    y_axis_vector = [0, 1, 0]

    for n in range(N):
        # Joint limits
        prog.AddBoundingBoxConstraint(
            plant.GetPositionLowerLimits(),
            plant.GetPositionUpperLimits(),
            q[:, n],
        )
        # Joint velocity limits
        prog.AddBoundingBoxConstraint(
            plant.GetVelocityLowerLimits(),
            plant.GetVelocityUpperLimits(),
            v[:, n],
        )

        # Body orientation
        prog.AddConstraint(
            AngleBetweenVectorsConstraint(
            plant,
            body_frame,           # Frame A (Body)
            y_axis_vector,        # Vector a_A: The Y-axis expressed in Body frame
            plant.world_frame(),  # Frame B (World)
            y_axis_vector,        # Vector b_B: The Y-axis expressed in World frame
            0.0,                  # Minimum angle (radians)
            0.01,                  # Maximum angle (radians) - The tolerance
            context[n]
        ),
        q[:, n]
        )

        alpha = n / (N - 1)
        q_guess = (1 - alpha) * q0 + alpha * q_crouch_vec
        prog.SetInitialGuess(q[:, n], q_guess)
    
        prog.SetInitialGuess(v[:, n], [0]*nv)
        #prog.SetInitialGuess(q[:, n], q0)
        #AddUnitQuaternionConstraintOnPlant(plant, q[:, n], prog)

        # Running costs:

        prog.AddQuadraticErrorCost(np.diag(q_cost), q0, q[:,n])
        prog.AddQuadraticErrorCost(np.diag(v_cost), [0] * nv, v[:, n])
        ### need to add cost for the forces 

    
    quat_target = np.array([1.0, 0.0, 0.0, 0.0])  # Unit quaternion
    for n in range(1,N-1):
    # Soft penalty to keep quaternion near unit norm
        
        prog.AddQuadraticErrorCost(
            Q=10.0 * np.eye(4),
            x_desired=quat_target,
            vars=q[0:4, n]
        )
        def unit_norm_cost(quat):
            qw, qx, qy, qz = quat
            norm_sq = qw**2 + qx**2 + qy**2 + qz**2
            weight = 100.0  # Adjust as needed
            return weight * (norm_sq - 1.0)**2
    
        prog.AddCost(unit_norm_cost, vars=q[0:4, n])
    
    prog.AddQuadraticErrorCost(
            Q=1000.0 * np.eye(4),
            x_desired=quat_target,
            vars=q[0:4, 0]
            )
    
            
    
    
    context_list = [plant.CreateDefaultContext() for _ in range(N)]
    ad_context_list = [ad_plant.CreateDefaultContext() for _ in range(N)]


   

    """
    Contact forces
    """

    contact_force = [
        prog.NewContinuousVariables(3, N - 1, f"foot{c_s}_contact_force")
        for c_s in range(8)
    ]

    # Set initial guess for contact forces based on stance schedule
    for c_s in range(8):
        for n in range(N-1):
            foot_idx = c_s // 4  # 0 for left foot (contacts 0-3), 1 for right foot (contacts 4-7)
            if in_stance[foot_idx, n]:
                # Distribute weight evenly across contact points
                force_guess = np.array([0, 0, total_mass * abs(gravity[2]) / 8])
            else:
                force_guess = np.zeros(3)
            prog.SetInitialGuess(contact_force[c_s][:, n], force_guess)

    """
    Friction Cone Constraint 
    """         
    for n in range(N - 1):
        for c_s in range(8):
            foot_idx = c_s // 4
            # Linear friction cone
            if in_stance[foot_idx,n] == 1:
                # normal force >=0, normal_force == 0 if not in_stance
                fz_min = 0.05 * total_mass * abs(gravity[-1])     # minimum load
                fz_max = 4.0  * total_mass * abs(gravity[-1])     # maximum load

                prog.AddLinearConstraint(contact_force[c_s][2, n] >=  fz_min)
                prog.AddLinearConstraint(contact_force[c_s][2, n] <=  fz_max)

                prog.AddLinearConstraint(
                    contact_force[c_s][0, n] <= mu * contact_force[c_s][2, n]
                )
                prog.AddLinearConstraint(
                    -contact_force[c_s][0, n] <= mu * contact_force[c_s][2, n]
                )
                prog.AddLinearConstraint(
                    contact_force[c_s][1, n] <= mu * contact_force[c_s][2, n]
                )
                prog.AddLinearConstraint(
                    -contact_force[c_s][1, n] <= mu * contact_force[c_s][2, n]
                )
            else: 
                prog.AddLinearConstraint(contact_force[c_s][0, n] == 0)
                prog.AddLinearConstraint(contact_force[c_s][1, n] == 0)
                prog.AddLinearConstraint(contact_force[c_s][2, n] == 0)

    """
    Center of mass variables and constraints
    """
    com = prog.NewContinuousVariables(3, N, "com")
    comdot = prog.NewContinuousVariables(3, N, "comdot")
    comddot = prog.NewContinuousVariables(3, N -1, "comddot")

    plant.SetPositions(context[0], q0)
    com0 = plant.CalcCenterOfMassPositionInWorld(context[0], [humanoid_g1])
    com_start = com0
    com_final = com0 - np.array([0, 0, 0.2])

# Loop through all knot points
    for n in range(N):
        # Calculate interpolation factor alpha (0.0 at start, 1.0 at end)
        alpha = n / (N - 1)
    
        # Linear Interpolation (Lerp): (1 - alpha) * start + alpha * end
        com_n = (1 - alpha) * com_start + alpha * com_final
    
        # Set the guess
        prog.SetInitialGuess(com[:, n], com_n)
    # Initial CoM position - relaxed to allow some movement
    prog.AddBoundingBoxConstraint(com0[0]-0.02, com0[0]+0.02, com[0, 0])
    prog.AddBoundingBoxConstraint(com0[1]-0.02, com0[1]+0.02, com[1, 0])
    prog.AddBoundingBoxConstraint(com0[2]-0.02, com0[2]+0.02, com[2, 0])
    prog.AddBoundingBoxConstraint(com0[0]-0.02, com0[0]+0.02, com[0, 1:])
    prog.AddBoundingBoxConstraint(com0[1]-0.02, com0[1]+0.02, com[1, -1])
    # CoM x vel
    prog.AddBoundingBoxConstraint(-0.5, 0.5, comdot[0, :])
    # CoM y vel
    prog.AddBoundingBoxConstraint(-0.3, 0.3, comdot[1, :])
    # Initial CoM z vel 
    prog.AddBoundingBoxConstraint(-0.2, 0.2, comdot[2, 0])
    # CoM final z position 
    prog.AddBoundingBoxConstraint(com0[2] - 0.2, com0[2] - 0.2, com[2, -1])
    prog.AddBoundingBoxConstraint(-0.5, 0.5, comdot[2, -1])

    H = prog.NewContinuousVariables(3, N, "H")
    Hdot = prog.NewContinuousVariables(3, N - 1, "Hdot")

    # Initialize angular momentum to zero (robot starts at rest)
    for n in range(N):
        prog.SetInitialGuess(H[:, n], np.zeros(3))
    for n in range(N-1):
    # Compute total torque from initial force guess
        total_torque = np.zeros(3)
    
        plant.SetPositions(context[n], q0)
    
        for i in range(2):  # feet
            for m in range(4):  # contact points per foot
                c_s = i * 4 + m
                
                if in_stance[i, n]:
                    p_WC = plant.CalcPointsPositions(
                        context[n],
                        foot_frame[i],
                        contact_points[m],
                        plant.world_frame()
                    ).flatten()
                    
                    r = p_WC - com0
                    f = prog.GetInitialGuess(contact_force[c_s][:, n])
                    total_torque += np.cross(r, f)
    
    # Initialize Hdot to match this torque
        prog.SetInitialGuess(Hdot[:, n], total_torque)
    
    
    # Dynamics Loop

    for n in range(N - 1):

        prog.AddConstraint(
            partial(velocity_dynamics_constraint, context_index=n, 
                    plant=plant, ad_plant=ad_plant, 
                    context=context_list, ad_context=ad_context_list),
            lb=-0.05 * np.ones(nv), ub=0.05 * np.ones(nv),
            vars=np.concatenate(([h[n]], q[:, n], v[:, n], q[:, n+1])),
            description=f"F_Kinematics_{n}"
        )

        # 2. CoM Integration (Approximate Euler)
        prog.AddConstraint(eq(com[:, n + 1], com[:, n] + h[n] * comdot[:, n]))
        prog.AddConstraint(eq(comdot[:, n + 1], comdot[:, n] + h[n] * comddot[:, n]))

        # 3. Angular Momentum Integration (Approximate Euler)
        prog.AddConstraint(eq(H[:, n+1], H[:, n] + h[n] * Hdot[:, n]))

        # 4. Newton's Law (Linear): F = ma
        total_force = sum(contact_force[i][:, n] for i in range(8))
        prog.AddConstraint(eq(total_mass * comddot[:, n], total_force + total_mass * gravity))
    
    """
    Angular momentum (about the center of mass)
    """
    

    def angular_momentum_constraint(vars, context_index):
        q, com, Hdot, contact_force = np.split(vars, [nq, nq+3, nq + 6])
        contact_force = contact_force.reshape(8,3).T # check again if I got this right
        if isinstance(vars[0], AutoDiffXd):
            dq = ExtractGradient(q)
            q = ExtractValue(q)
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            torque = com * 0.0
            for i in range(2):
                for m in range (4):
                    p_WF = plant.CalcPointsPositions(
                        context[context_index],
                        foot_frame[i],
                        contact_points[m],
                        plant.world_frame()
                    )   # have to check if this returns a autodiff or not (if not would probably cause an error)
                    Jq_WF = plant.CalcJacobianTranslationalVelocity(
                    context[context_index],
                    JacobianWrtVariable.kQDot,
                    foot_frame[i],
                    contact_points[m],
                    plant.world_frame(),
                    plant.world_frame(), 
                    )
                    ad_p_WF = InitializeAutoDiff(p_WF, Jq_WF @ dq)
                    r_com_to_contact = ad_p_WF.reshape(3) - com
                    torque = torque + np.cross(
                        r_com_to_contact, contact_force[:,(i*4) + m]
                    )
        else :
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            torque = np.zeros(3)
            for i in range(2):
                for m in range (4):
                    p_WF = plant.CalcPointsPositions(
                        context[context_index],
                        foot_frame[i],
                        contact_points[m],
                        plant.world_frame(),
                    )
                    torque += np.cross(p_WF.reshape(3) - com, contact_force[:, (i*4) + m])

        return Hdot-torque
    

    for n in range(N - 1):
        Fn = np.concatenate([contact_force[i][:, n] for i in range(8)])
        prog.AddConstraint(
            partial(angular_momentum_constraint, context_index=n),
            lb = -0.1 * np.ones(3),
            ub = 0.1 * np.ones(3),
            vars=np.concatenate((q[:, n ], com[:, n], Hdot[:, n], Fn)),
            description=f"H_torque_{n}"
        )

    """
    Making Sure the Centroidal dynamics is followed by the joints
    """
    # com == CenterOfMass(q), H = SpatialMomentumInWorldAboutPoint(q, v, com)
    # Make a new autodiff context for this constraint (to maximize cache hits)
    com_constraint_context = [ad_plant.CreateDefaultContext() for i in range(N)]
    

    def com_constraint (vars, context_index):
        qv, com, H = np.split(vars, [nq + nv, nq + nv + 3])
        if isinstance(vars[0], AutoDiffXd):
            if not autoDiffArrayEqual(
                qv,
                ad_plant.GetPositionsAndVelocities(
                    com_constraint_context[context_index])
                ):
                ad_plant.SetPositionsAndVelocities(
                    com_constraint_context[context_index], qv
                )
            com_q = ad_plant.CalcCenterOfMassPositionInWorld(
                com_constraint_context[context_index], [humanoid_g1]
            )
            H_qv = ad_plant.CalcSpatialMomentumInWorldAboutPoint(
                com_constraint_context[context_index], [humanoid_g1], com
            ).rotational()
        else:
            if not np.array_equal(
                qv, plant.GetPositionsAndVelocities(context[context_index])
            ):
                plant.SetPositionsAndVelocities(context[context_index], qv)
            com_q = plant.CalcCenterOfMassPositionInWorld(
                context[context_index], [humanoid_g1]
            )
            H_qv = plant.CalcSpatialMomentumInWorldAboutPoint(
                context[context_index], [humanoid_g1], com
            ).rotational()

        return np.concatenate((com_q - com, H_qv - H))
    for n in range (N):
        epsilon_com = 0.05 # 5cm tolerance
        epsilon_mom = 0.4   # Angular momentum tolerance

        prog.AddConstraint(
            partial(com_constraint, context_index = n),
            lb = np.concatenate((-epsilon_com * np.ones(3), -epsilon_mom * np.ones(3))),
            ub = np.concatenate(( epsilon_com * np.ones(3),  epsilon_mom * np.ones(3))),
            vars = np.concatenate((q[:,n], v[:,n], com[:,n], H[:,n])),
            description=f"spacial_momentum_{n}"
        )

    #Kinematic constraints
    def fixed_position_constraint(vars, context_index, frame, contact_point):
        q, qn = np.split(vars, [nq])
        if not np.array_equal(q, plant.GetPositions(context[context_index])):
            plant.SetPositions(context[context_index], q)
        if not np.array_equal (qn, plant.GetPositions(context[context_index+1])):
            plant.SetPositions(context[context_index+1], qn)
        p_WF = plant.CalcPointsPositions(context[context_index], frame,
                                        contact_point, plant.world_frame())
        p_WF_n = plant.CalcPointsPositions(context[context_index + 1], frame,
                                        contact_point, plant.world_frame())
        if isinstance(vars[0], AutoDiffXd):
            J_WF = plant.CalcJacobianTranslationalVelocity(
                context[context_index], JacobianWrtVariable.kQDot, frame,
                contact_point, plant.world_frame(), plant.world_frame())
            J_WF_n = plant.CalcJacobianTranslationalVelocity(
                context[context_index + 1], JacobianWrtVariable.kQDot, frame,
                contact_point, plant.world_frame(), plant.world_frame())
            return InitializeAutoDiff(
                p_WF_n - p_WF,
                J_WF_n @ ExtractGradient(qn) - J_WF @ ExtractGradient(q))
        else:
            return p_WF_n - p_WF

    for n in range(N):
        for i in range (2):
            for m in range (4):
                if in_stance[i,n]:
                    # foot shoube be on the ground (world position z = 0)
                    prog.AddConstraint(
                        PositionConstraint(plant, plant.world_frame(),
                        [-np.inf , -np.inf , -0.001],
                        [np.inf, np.inf, 0.005], foot_frame[i],
                        contact_points[m], context[n]), q[:,n]
                    )
                    
                    if n > 0 and in_stance [i,n-1]:
                        # feet should not move during stance
                        prog.AddConstraint(partial(
                            fixed_position_constraint,
                            context_index = n-1, 
                            frame = foot_frame[i],
                            contact_point = contact_points[m]),
                        lb = -0.005*np.ones(3),
                        ub = 0.005*np.ones(3),
                        vars = np.concatenate ((q[:, n - 1], q[:,n]
                        ))
                    
                        )
    snopt = SnoptSolver().solver_id()
    prog.SetSolverOption(snopt, 'Iterations Limits', 5e5 )
    prog.SetSolverOption(snopt, 'Major Iterations Limit', 20000 )
    prog.SetSolverOption(snopt, 'Major Feasibility Tolerance', 5e-4)
    prog.SetSolverOption(snopt, 'Major Optimality Tolerance', 1e-4)
    prog.SetSolverOption(snopt, 'Superbasics limit', 2000)
    prog.SetSolverOption(snopt, 'Linesearch tolerance', 0.9)

    # Check initial guess feasibility
    print("\nChecking initial guess...")
    x0 = prog.GetInitialGuess(prog.decision_variables())
    for binding in prog.GetAllConstraints():
        constraint = binding.evaluator()
        vars_idx = [prog.FindDecisionVariableIndex(v) for v in binding.variables()]
        x_constraint = x0[vars_idx]

        y = constraint.Eval(x_constraint)
        lb = constraint.lower_bound()
        ub = constraint.upper_bound()

        violation = np.maximum(lb - y, y - ub)
        max_violation = np.max(violation)

        if max_violation > 1e-3:
            print(f"Constraint '{constraint.get_description()}' violated by {max_violation:.4f}")

    result = Solve(prog)



    

    if result.is_success():
        print("\n Trajectory optimization converged!")
        
        # Extract solution
        t_sol = np.cumsum(np.concatenate(([0], result.GetSolution(h))))
        q_sol = PiecewisePolynomial.FirstOrderHold(t_sol, result.GetSolution(q))

        # Create a time grid across the trajectory
        times = np.linspace(q_sol.start_time(), q_sol.end_time(), 500)
        q_traj = np.array([q_sol.value(t).flatten() for t in times])

        # Plot joint trajectories
        plt.figure(figsize=(12, 6))
        for i in range(min(23, q_traj.shape[1])):  # Plot first 10 joints
            plt.plot(times, q_traj[:, i], label=f'q[{i}]', alpha=0.7)
        plt.xlabel("Time [s]")
        plt.ylabel("Joint positions [rad]") 
        plt.title("Joint Position Trajectory")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(15, 10))
    
    # Get force solutions for all contact points
        forces = []
        for c_s in range(8):
            force_sol = result.GetSolution(contact_force[c_s])
            forces.append(force_sol)
    
    # Plot 1: Force magnitudes over time
        plt.subplot(3, 1, 1)
        for c_s in range(8):
            foot_idx = c_s // 4
            contact_idx = c_s % 4
            foot_name = "Left" if foot_idx == 0 else "Right"
        
        # Compute force magnitude at each knot
            magnitudes = np.linalg.norm(forces[c_s], axis=0)
        
            plt.plot(t_sol[:-1], magnitudes, 
                label=f'{foot_name} foot, contact {contact_idx}',
                marker='o', markersize=3)
    
        plt.xlabel('Time [s]')
        plt.ylabel('Force Magnitude [N]')
        plt.title('Contact Force Magnitudes')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
    
    # Plot 2: Normal (vertical) forces
        plt.subplot(3, 1, 2)
        for c_s in range(8):
            foot_idx = c_s // 4
            foot_name = "Left" if foot_idx == 0 else "Right"
            contact_idx = c_s % 4
        
            plt.plot(t_sol[:-1], forces[c_s][2, :], 
                label=f'{foot_name} foot, contact {contact_idx}',
                marker='o', markersize=3)
    
        plt.xlabel('Time [s]')
        plt.ylabel('Normal Force Fz [N]')
        plt.title('Normal (Vertical) Contact Forces')
        expected_weight = total_mass * abs(gravity[2])
        plt.axhline(y=expected_weight/8, color='k', linestyle='--', 
                label='Expected per contact')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
    
        # Plot 3: Tangential (horizontal) forces
        plt.subplot(3, 1, 3)
        for c_s in range(8):
            foot_idx = c_s // 4
            foot_name = "Left" if foot_idx == 0 else "Right"
            contact_idx = c_s % 4
        
        # Compute horizontal force magnitude
            tangential = np.sqrt(forces[c_s][0, :]**2 + forces[c_s][1, :]**2)
        
            plt.plot(t_sol[:-1], tangential, 
                label=f'{foot_name} foot, contact {contact_idx}',
                marker='o', markersize=3)
    
        plt.xlabel('Time [s]')
        plt.ylabel('Tangential Force [N]')
        plt.title('Tangential (Horizontal) Contact Forces')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
    
        plt.tight_layout()
        # uncomment if you want to print
        plt.savefig("Tangential_force.png")

        print(f"\nTrajectory duration: {t_sol[-1]:.2f} seconds")
        print(f"Number of knot points: {N}")
        
        # --- Visualization-only playback in Meshcat ---
        builder2 = DiagramBuilder()
        plant2, scene_graph2 = AddMultibodyPlantSceneGraph(builder2, 0.0)

        # Add robot model
        parser2 = Parser(plant2)
        parser2.AddModels(str(project_root /"urdf/g1_description/g1_23dof.urdf"))
        plant2.Finalize()

        # Add Meshcat visualizer
        MeshcatVisualizer.AddToBuilder(builder2, scene_graph2, meshcat)
        diagram2 = builder2.Build()

        # Create contexts
        diagram_context = diagram2.CreateDefaultContext()
        plant_context = plant2.GetMyMutableContextFromRoot(diagram_context)

        print("\n Playing back trajectory via Meshcat...")

        # Loop through time samples and update plant positions
        for t in np.linspace(q_sol.start_time(), q_sol.end_time(), 500):
            q = q_sol.value(t).flatten()
            plant2.SetPositions(plant_context, q)
            diagram2.ForcedPublish(diagram_context)
            time.sleep(0.05)  # Adjust playback speed

        print("✓ Playback complete!")
        
        
    else:
        print("\n Solver failed.")
        print(f"Solver status: {result.get_solver_id().name()}")
        info_code = result.get_solver_details().info
        print(info_code)
        infeasible = result.GetInfeasibleConstraintNames(prog)
        if infeasible:
            print(f"\nNumber of infeasible constraints: {len(infeasible)}")
            print("First 100 infeasible constraints:")
            for constraint in infeasible[:100]:
                print(f"  - {constraint}")