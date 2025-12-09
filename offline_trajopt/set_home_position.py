from offline_trajopt.imports_setup import *
def set_home(plant, context):
    PositionView = namedview("Positions", plant.GetPositionNames(
            plant.GetModelInstanceByName("g1_23dof")
        ),
    )
    q0 = PositionView(np.zeros(plant.num_positions()))
    q0.pelvis_qw = 1.0 # unit quaternion
    q0.pelvis_z = 0.783
     
    q0.left_hip_pitch_joint_q = -0.02
    q0.left_hip_roll_joint_q = 0
    q0.left_hip_yaw_joint_q = 0
    q0.left_knee_joint_q = 0.05
    q0.left_ankle_pitch_joint_q = 0.0
    q0.left_ankle_roll_joint_q = 0

    q0.right_hip_pitch_joint_q = -0.02
    q0.right_hip_roll_joint_q = 0
    q0.right_hip_yaw_joint_q = 0
    q0.right_knee_joint_q = 0.05
    q0.right_ankle_pitch_joint_q = 0.00
    q0.right_ankle_roll_joint_q = 0

    q0.waist_yaw_joint_q = 0
    q0.left_shoulder_pitch_joint_q = 0 
    q0.left_shoulder_roll_joint_q = 0 
    q0.left_shoulder_yaw_joint_q = 0
    q0.left_elbow_joint_q = 0.8 
    q0.left_wrist_roll_joint_q = 0 

    q0.right_shoulder_pitch_joint_q = 0 
    q0.right_shoulder_roll_joint_q = 0 
    q0.right_shoulder_yaw_joint_q = 0
    q0.right_elbow_joint_q = 0.8
    q0.right_wrist_roll_joint_q = 0 



    plant.SetPositions(context,q0[:])
    print ("Robot home position set.")