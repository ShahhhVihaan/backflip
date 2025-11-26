import sys
import pickle
import numpy as np
import pinocchio as pin
import casadi
from pinocchio import casadi as cpin

MJCF_PATH = "urdf/humanoid/humanoid.xml"
PKL_PATH = "motion/humanoid_backflip.pkl"

DT = 0.02
T_TOTAL = 1.75
N_NODES = int(T_TOTAL / DT)
N_STANCE = int(0.3 / DT)

model = pin.buildModelFromMJCF(MJCF_PATH)
cmodel = cpin.Model(model)
cdata = cmodel.createData()

nq = model.nq
nv = model.nv
nu = nv - 6

r_foot_id = model.getFrameId("right_foot")
l_foot_id = model.getFrameId("left_foot")

print(f"Casadi Solver: Nodes={N_NODES}, Vars={N_NODES*(nq+nv+nu)}")

def get_warm_start():
    # Placeholder
    # initialize with a valid standing pose to prevent NaN errors.
    q_warm = np.zeros((N_NODES, nq))
    q_warm[:, 3] = 0 # qx
    q_warm[:, 4] = 0 # qy
    q_warm[:, 5] = 0 # qz
    q_warm[:, 6] = 1 # qw (Identity Quaternion)
    
    # Lift z slightly so we don't start in collision
    q_warm[:, 2] = 1.0 
    
    return q_warm

# symbolic functions
q_sym = casadi.SX.sym('q', nq)
v_sym = casadi.SX.sym('v', nv)
a_sym = casadi.SX.sym('a', nv)

# integration
q_int = cpin.integrate(cmodel, q_sym, v_sym)
integrate_func = casadi.Function('integrate', [q_sym, v_sym], [q_int])

# inverse dynamics (RNEA)
tau_sym = cpin.rnea(cmodel, cdata, q_sym, v_sym, a_sym)
rnea_func = casadi.Function('rnea', [q_sym, v_sym, a_sym], [tau_sym])

# contact kinematics
# update symbolic data
cpin.forwardKinematics(cmodel, cdata, q_sym, v_sym)
cpin.updateFramePlacements(cmodel, cdata)

# extract velocity (no arguments needed for getFrameVelocity in casadi bindings)
v_r_sym = cpin.getFrameVelocity(cmodel, cdata, r_foot_id, pin.LOCAL_WORLD_ALIGNED)
v_l_sym = cpin.getFrameVelocity(cmodel, cdata, l_foot_id, pin.LOCAL_WORLD_ALIGNED)

# create function
feet_vel_func = casadi.Function('feet_vel', [q_sym, v_sym], [v_r_sym.linear, v_l_sym.linear])

# build problem
opti = casadi.Opti()

Q = opti.variable(nq, N_NODES)
V = opti.variable(nv, N_NODES)
U = opti.variable(nu, N_NODES) 

total_cost = 0

for k in range(N_NODES - 1):
    # integration
    q_next_calc = integrate_func(Q[:, k], V[:, k] * DT)
    opti.subject_to(Q[:, k+1] == q_next_calc)
    
    # implicit acceleration
    acc_k = (V[:, k+1] - V[:, k]) / DT
    
    # dynamics
    tau_req = rnea_func(Q[:, k], V[:, k], acc_k)
    
    if k > N_STANCE:
        # flight phase
        # Root torques must be 0
        opti.subject_to(tau_req[0:6] == 0)
        # Actuators match U
        opti.subject_to(tau_req[6:] == U[:, k])
    else:
        # stance phase
        # feet fixed
        vr, vl = feet_vel_func(Q[:, k], V[:, k])
        opti.subject_to(vr == 0) 
        opti.subject_to(vl == 0)
        
        # actuators match U (ignoring root forces which are handled by ground)
        opti.subject_to(tau_req[6:] == U[:, k])

    # cost
    total_cost += casadi.sumsqr(U[:, k]) * 1e-4

# terminal constraints
q_end = Q[:, -1]
v_end = V[:, -1]

# complete flip (identity quaternion)
opti.subject_to(q_end[3] == 0)
opti.subject_to(q_end[4] == 0)
opti.subject_to(q_end[5] == 0)

# stop spinning
opti.subject_to(v_end[3:6] == 0)

opti.minimize(total_cost)

# q_guess = get_warm_start() 
# opti.set_initial(Q, q_guess.T)

p_opts = {"expand": True} 
s_opts = {"max_iter": 500, "print_level": 5}
opti.solver('ipopt', p_opts, s_opts)

print("Solving...")
try:
    sol = opti.solve()
    print("Optimization Success!")
    q_sol = sol.value(Q)
    
    # Visualize
    from pinocchio.visualize import MeshcatVisualizer
    import time
    
    viz = MeshcatVisualizer(model, pin.buildGeomFromMJCF(model, MJCF_PATH, pin.GeometryType.COLLISION), pin.buildGeomFromMJCF(model, MJCF_PATH, pin.GeometryType.VISUAL))
    viz.initViewer(open=False)
    viz.loadViewerModel()
    
    print("Replaying Solution...")
    while True:
        for k in range(N_NODES):
            viz.display(q_sol[:, k])
            time.sleep(DT)
        time.sleep(1)

except Exception as e:
    print(f"Failed: {e}")
    # print(opti.debug.value(Q))