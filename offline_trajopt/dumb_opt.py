import sys
import pickle
import numpy as np
import pinocchio as pin
import casadi
from pinocchio import casadi as cpin
from scipy.interpolate import interp1d

MJCF_PATH = "urdf/humanoid/humanoid.xml"
PKL_PATH = "motion/humanoid_backflip.pkl"

T_TOTAL = 1.75
T_TAKEOFF = 0.56
T_LANDING = 1.25
DT = 0.02

N_NODES = int(T_TOTAL / DT)
N_TAKEOFF = int(T_TAKEOFF / DT)
N_LANDING = int(T_LANDING / DT)

GROUND_HEIGHT = 0.0 


Q_START_TARGET = np.array([
 0.000e+00,  0.000e+00,  8.867e-01,  9.994e-01,  2.921e-02,  1.796e-02,
-5.250e-04,  8.637e-04, -1.100e-02,  1.144e-03,  3.331e-03,  2.821e-01,
-2.241e-02,  1.553e+00, -3.109e-01, -8.020e-02,  2.405e-01, -1.454e+00,
-1.229e-01, -1.837e-01, -1.489e-01, -4.892e-02,  7.776e-02, -1.678e-03,
 1.419e-02,  2.149e-02, -7.071e-02, -2.102e-01, -4.184e-02,  1.260e-02,
-2.585e-02,  2.786e-02, -2.149e-02,  1.852e-02,  2.102e-01
])

Q_END_TARGET = np.array([
-1.079e+00,  0.000e+00,  8.867e-01,  9.994e-01,  2.921e-02,  1.796e-02,
-5.250e-04,  8.637e-04, -1.100e-02,  1.144e-03,  4.347e-02,  2.821e-01,
-2.241e-02,  1.553e+00, -3.109e-01, -8.020e-02,  2.405e-01, -1.454e+00,
-1.229e-01, -1.837e-01, -1.489e-01, -4.836e-02,  7.688e-02, -1.678e-03,
 6.499e-02,  2.149e-02, -7.071e-02, -2.102e-01, -4.182e-02,  1.260e-02,
-8.986e-02,  2.786e-02, -5.204e-02, -8.193e-02,  2.105e-01
])

print(f"Total Nodes: {N_NODES} | DT: {DT}")
print(f"Takeoff: Node {N_TAKEOFF} ({T_TAKEOFF}s) | Landing: Node {N_LANDING} ({T_LANDING}s)")

model = pin.buildModelFromMJCF(MJCF_PATH)
cmodel = cpin.Model(model)
cdata = cmodel.createData()

nq = model.nq
nv = model.nv
nu = nv - 6

r_foot_id = model.getFrameId("right_foot")
l_foot_id = model.getFrameId("left_foot")

def get_warm_start():
    print(f"Loading {PKL_PATH} for Warm Start...")
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)
    
    raw_frames = data['frames']
    if isinstance(raw_frames[0], dict):
        raw_traj = np.array([f['qpos'] for f in raw_frames])
    else:
        raw_traj = np.array(raw_frames)

    # Convert Rotations (34 -> 35 dims)
    clean_traj = []
    for f in raw_traj:
        if f.shape[0] == 34:
            m = pin.exp3(f[3:6])
            q = pin.Quaternion(m)
            quat = np.array([q.x, q.y, q.z, q.w])
            clean_traj.append(np.concatenate([f[:3], quat, f[6:]]))
        else:
            clean_traj.append(f)
    clean_traj = np.array(clean_traj)

    t_file = np.linspace(0, T_TOTAL, len(clean_traj))
    t_opt = np.linspace(0, T_TOTAL, N_NODES)
    
    Q_warm = np.zeros((N_NODES, nq))
    for i in range(nq):
        interpolator = interp1d(t_file, clean_traj[:, i], kind='linear', fill_value="extrapolate")
        Q_warm[:, i] = interpolator(t_opt)
        if i == 6: # Normalize quaternion
             norms = np.linalg.norm(Q_warm[:, 3:7], axis=1)
             Q_warm[:, 3:7] /= norms[:, None]

    # Calculate Velocities
    # Using Pinocchio difference to handle quaternions correctly
    V_warm = np.zeros((N_NODES, nv))
    for k in range(N_NODES - 1):
        V_warm[k, :] = pin.difference(model, Q_warm[k], Q_warm[k+1]) / DT
    
    return Q_warm, V_warm

Q_WARM, V_WARM = get_warm_start()

q_sym = casadi.SX.sym('q', nq)
v_sym = casadi.SX.sym('v', nv)
a_sym = casadi.SX.sym('a', nv)

q_int = cpin.integrate(cmodel, q_sym, v_sym)
integrate_func = casadi.Function('integrate', [q_sym, v_sym], [q_int])

tau_id = cpin.rnea(cmodel, cdata, q_sym, v_sym, a_sym)
rnea_func = casadi.Function('rnea', [q_sym, v_sym, a_sym], [tau_id])

cpin.forwardKinematics(cmodel, cdata, q_sym, v_sym)
cpin.updateFramePlacements(cmodel, cdata)

v_r = cpin.getFrameVelocity(cmodel, cdata, r_foot_id, pin.LOCAL_WORLD_ALIGNED)
v_l = cpin.getFrameVelocity(cmodel, cdata, l_foot_id, pin.LOCAL_WORLD_ALIGNED)

p_r = cdata.oMf[r_foot_id]
p_l = cdata.oMf[l_foot_id]

feet_func = casadi.Function('feet', [q_sym, v_sym], 
                            [v_r.linear, v_l.linear, 
                             p_r.translation, p_l.translation])

opti = casadi.Opti()

Q = opti.variable(nq, N_NODES)
V = opti.variable(nv, N_NODES)
U = opti.variable(nu, N_NODES)

opti.subject_to(Q[7:, 0] == Q_START_TARGET[7:])
opti.subject_to(V[:, 0] == 0)

start_error = Q[0:7, 0] - Q_START_TARGET[0:7]
total_cost = casadi.sumsqr(start_error) * 10.0 

for k in range(N_NODES - 1):
    q_next = integrate_func(Q[:, k], V[:, k] * DT)
    opti.subject_to(Q[:, k+1] == q_next)
    
    acc = (V[:, k+1] - V[:, k]) / DT
    tau = rnea_func(Q[:, k], V[:, k], acc)
        
    # FLIGHT PHASE - Takeoff to Landing
    if k >= N_TAKEOFF and k < N_LANDING:
        opti.subject_to(tau[0:6] == 0)
        # Actuation
        opti.subject_to(tau[6:] == U[:, k])
        
    # STANCE PHASE - Start OR Recovery
    else:
        v_r_lin, v_l_lin, p_r_pos, p_l_pos = feet_func(Q[:, k], V[:, k])
        
        opti.subject_to(v_r_lin == 0)
        opti.subject_to(v_l_lin == 0)
        
        total_cost += casadi.sumsqr(p_r_pos[2] - GROUND_HEIGHT) * 1000.0
        total_cost += casadi.sumsqr(p_l_pos[2] - GROUND_HEIGHT) * 1000.0
        
        opti.subject_to(tau[6:] == U[:, k])

    total_cost += casadi.sumsqr(U[:, k]) * 1e-4
    
    total_cost += casadi.sumsqr(Q[:, k] - Q_WARM[k, :]) * 0.1 

opti.subject_to(Q[7:, -1] == Q_END_TARGET[7:])
opti.subject_to(V[:, -1] == 0)

end_error = Q[0:7, -1] - Q_END_TARGET[0:7]
total_cost += casadi.sumsqr(end_error) * 10.0

opti.minimize(total_cost)

opti.set_initial(Q, Q_WARM.T)
# opti.set_initial(V, V_WARM.T)

p_opts = {"expand": True}
s_opts = {"max_iter": 1000, "print_level": 5, "tol": 1e-3}
opti.solver('ipopt', p_opts, s_opts)

print("Solving with CasADi + IPOPT...")
try:
    sol = opti.solve()
    print("OPTIMAL SOLUTION FOUND!")
    q_sol = sol.value(Q)
    
    from pinocchio.visualize import MeshcatVisualizer
    import time
    
    viz = MeshcatVisualizer(model, pin.buildGeomFromMJCF(model, MJCF_PATH, pin.GeometryType.COLLISION), pin.buildGeomFromMJCF(model, MJCF_PATH, pin.GeometryType.VISUAL))
    viz.initViewer(open=False)
    viz.loadViewerModel()
    
    print("Replaying... (Check Meshcat)")
    while True:
        for k in range(N_NODES):
            viz.display(q_sol[:, k])
            time.sleep(DT)
        time.sleep(1.0)

except Exception as e:
    print(f"Solver Failed: {e}")