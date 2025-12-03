# From mjlab org
# Adapted from g1_spinkick example repository


"""Convert MimicKit pickle motion files to CSV format."""

import pickle

import numpy as np
import tyro
from scipy.spatial.transform import Rotation, Slerp


class FlexibleClass:
  """A class that accepts any arguments and stores them."""

  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs
    for k, v in kwargs.items():
      setattr(self, k, v)

  def __repr__(self):
    attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
    return f"{self.__class__.__name__}({list(attrs.keys())})"


class RobustUnpickler(pickle.Unpickler):
  def find_class(self, module, name):
    try:
      return super().find_class(module, name)
    except (ImportError, AttributeError):
      return type(name, (FlexibleClass,), {})


def ease_out_cubic(t):
  """Cubic ease-out curve for smooth deceleration."""
  return 1 - (1 - t) ** 3


def ease_in_cubic(t):
  """Cubic ease-in curve for smooth acceleration."""
  return t**3


def main(
  pkl_file: str = "humanoid_backflip.pkl",
  csv_file: str = "humanoid_backflip.csv",
  duration: float | None = None,
  pad_duration: float = 0.0,
  transition_duration: float = 0.5,
  add_start_transition: bool = True,
  add_end_transition: bool = True,
):
  """Convert pickle motion file to CSV format.

  Output format: [root_pos(3), quat_xyzw(4), joint_dof(N)]

  Args:
    pkl_file: Path to input .pkl file.
    csv_file: Path to output .csv file.
    duration: Desired duration in seconds. If None, use original duration. Motion will
      be cycled to reach this duration.
    pad_duration: Duration in seconds to hold the final pose at the end.
    transition_duration: Duration in seconds to blend to/from safe standing pose.
    add_start_transition: Whether to add transition from safe standing pose to motion start.
    add_end_transition: Whether to add transition from motion end to safe standing pose.
  """
  # ---------------------------------------------------------------------------
  # Safe pose taken from the "arms" keyframe in humanoid.xml
  # qpos order in MJCF:
  #   [root_x, root_y, root_z, root_qw, root_qx, root_qy, root_qz,
  #    abdomen_x, abdomen_y, abdomen_z,
  #    neck_x, neck_y, neck_z,
  #    right_shoulder_x, right_shoulder_y, right_shoulder_z, right_elbow,
  #    left_shoulder_x,  left_shoulder_y,  left_shoulder_z,  left_elbow,
  #    right_hip_x, right_hip_y, right_hip_z, right_knee,
  #    right_ankle_x, right_ankle_y, right_ankle_z,
  #    left_hip_x, left_hip_y, left_hip_z, left_knee,
  #    left_ankle_x, left_ankle_y, left_ankle_z]
  #
  # This must match the joint DOF ordering used to build the PKL frames.
  # ---------------------------------------------------------------------------

  SAFE_QPOS = np.array(
    [
      # free root joint: (x, y, z, qw, qx, qy, qz)
      0.0, 0.0, 0.880749, 1.0, 0.0, 0.0, 0.0,
      # abdomen_x, abdomen_y, abdomen_z
      0.0, 0.0, 0.0,
      # neck_x, neck_y, neck_z
      0.0, 0.0, 0.0,
      # right_shoulder_x, right_shoulder_y, right_shoulder_z, right_elbow
      1.55281651, -0.310888141, -0.080201156, 0.240463093,
      # left_shoulder_x, left_shoulder_y, left_shoulder_z, left_elbow
      -1.55281651, 0.310888141, 0.080201156, -0.240463093,
      # remaining leg joints (hips, knees, ankles) – all zero in this keyframe
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
  )

  # Safe standing height = pelvis z from the keyframe.
  SAFE_Z_HEIGHT = float(SAFE_QPOS[2])

  # Safe standing *joint* pose = all entries after the free root joint.
  SAFE_POSE_JOINTS = SAFE_QPOS[7:]


  print(f"Loading {pkl_file}...")
  with open(pkl_file, "rb") as f:
    unpickler = RobustUnpickler(f)
    data = unpickler.load()

  if isinstance(data, dict):
    frames = data["frames"]
    fps = data["fps"]
  else:
    frames = data.frames
    fps = data.fps

  frames = np.asarray(frames)
  original_duration = frames.shape[0] / fps

  print(f"Loaded motion with shape: {frames.shape}")
  print(f"FPS: {fps}")
  print(
    f"Original duration: {original_duration:.2f} seconds ({frames.shape[0]} frames)"
  )

  # Repeat motion if duration is specified.
  if duration is not None:
    if duration < original_duration:
      print(
        f"Warning: Requested duration ({duration}s) is shorter than original ({original_duration:.2f}s)"
      )
      print("         Truncating motion...")
      num_frames = int(duration * fps)
      frames = frames[:num_frames]
    else:
      num_cycles = int(np.ceil(duration / original_duration))
      print(f"Repeating motion {num_cycles} times to reach {duration}s...")

      # Calculate displacement per cycle (only XY, not Z).
      start_pos = frames[0, 0:3]
      end_pos = frames[-1, 0:3]
      displacement_per_cycle = end_pos - start_pos

      # Only accumulate XY displacement, zero out Z.
      displacement_per_cycle[2] = 0.0

      print(
        f"XY displacement per cycle: [{displacement_per_cycle[0]:.4f}, {displacement_per_cycle[1]:.4f}]"
      )
      if np.linalg.norm(displacement_per_cycle[:2]) < 1e-3:
        print("  (In-place motion detected)")
      else:
        print("  (Forward motion detected - accumulating XY displacement only)")

      # Repeat and accumulate XY displacement only.
      repeated_frames = []
      for cycle in range(num_cycles):
        cycle_frames = frames.copy()
        # Add cumulative displacement to this cycle (XY only).
        cycle_frames[:, 0:3] += displacement_per_cycle * cycle
        repeated_frames.append(cycle_frames)

      frames = np.vstack(repeated_frames)

      # Truncate to exact duration.
      num_frames = int(duration * fps)
      frames = frames[:num_frames]

    print(f"New motion shape: {frames.shape}")
    print(
      f"New duration: {frames.shape[0] / fps:.2f} seconds ({frames.shape[0]} frames)"
    )

  # Add transition FROM safe standing pose TO motion start.
  if add_start_transition:
    print("\nAdding start transition from safe standing pose to motion start...")

    # Get the first frame as target.
    target_frame = frames[0].copy()

    # Number of transition frames.
    transition_frames = int(transition_duration * fps)
    print(
      f"Creating {transition_duration}s start transition ({transition_frames} frames)..."
    )

    # Extract components from target (first) frame.
    target_pos = target_frame[0:3]
    target_rot = target_frame[3:6]  # axis-angle
    target_joints = target_frame[6:]

    # Convert target rotation to euler angles to extract yaw.
    target_rot_obj = Rotation.from_rotvec(target_rot)
    euler_zyx = target_rot_obj.as_euler("ZYX", degrees=False)
    yaw = euler_zyx[0]

    print(
      f"  First frame orientation: yaw={np.degrees(yaw):.1f}°, pitch={np.degrees(euler_zyx[1]):.1f}°, roll={np.degrees(euler_zyx[2]):.1f}°"
    )
    print(
      f"  Start orientation: yaw={np.degrees(yaw):.1f}° (matched), pitch=0°, roll=0°"
    )

    # Create start rotation: yaw from first frame, roll=0, pitch=0.
    start_rot_obj = Rotation.from_euler("ZYX", [yaw, 0, 0], degrees=False)
    start_rot = start_rot_obj.as_rotvec()

    # Start position: XY from first frame, Z at safe height.
    start_pos = target_pos.copy()
    start_pos[2] = SAFE_Z_HEIGHT

    # Start joints: safe standing pose.
    start_joints = SAFE_POSE_JOINTS

    # Create Slerp interpolator for smooth rotation transition.
    key_times = [0, 1]
    key_rots = Rotation.concatenate([start_rot_obj, target_rot_obj])
    slerp = Slerp(key_times, key_rots)

    # Create transition frames.
    start_transition = []
    for i in range(transition_frames):
      t = i / (transition_frames - 1) if transition_frames > 1 else 1.0
      t_eased = ease_in_cubic(t)  # Apply ease-in for acceleration.

      # LERP position.
      pos = start_pos * (1 - t_eased) + target_pos * t_eased

      # SLERP rotation.
      rot_obj = slerp(t_eased)
      rot = rot_obj.as_rotvec()

      # LERP joints.
      joints = start_joints * (1 - t_eased) + target_joints * t_eased

      # Combine.
      frame = np.concatenate([pos, rot, joints])
      start_transition.append(frame)

    start_transition = np.array(start_transition)
    frames = np.vstack([start_transition, frames])
    print(f"After start transition, motion shape: {frames.shape}")
    print(f"Total duration so far: {frames.shape[0] / fps:.2f} seconds")

  # Add transition TO safe standing pose at the end.
  if add_end_transition:
    print(
      "\nAdding end transition to safe standing pose (keeping yaw, resetting roll/pitch)..."
    )

    # Get the last frame as starting point for transition.
    start_frame = frames[-1].copy()

    # Number of transition frames.
    transition_frames = int(transition_duration * fps)
    print(
      f"Creating {transition_duration}s end transition ({transition_frames} frames)..."
    )

    # Extract components from start frame.
    start_pos = start_frame[0:3]
    start_rot = start_frame[3:6]  # axis-angle
    start_joints = start_frame[6:]

    # Convert start rotation to euler angles to extract yaw.
    start_rot_obj = Rotation.from_rotvec(start_rot)
    euler_zyx = start_rot_obj.as_euler("ZYX", degrees=False)
    yaw = euler_zyx[0]  # Extract yaw.

    print(
      f"  Final frame orientation: yaw={np.degrees(yaw):.1f}°, pitch={np.degrees(euler_zyx[1]):.1f}°, roll={np.degrees(euler_zyx[2]):.1f}°"
    )
    print(f"  Target orientation: yaw={np.degrees(yaw):.1f}° (kept), pitch=0°, roll=0°")

    # Create target rotation: yaw from final frame, roll=0, pitch=0.
    target_rot_obj = Rotation.from_euler("ZYX", [yaw, 0, 0], degrees=False)
    target_rot = target_rot_obj.as_rotvec()

    # Target: keep XY position, transition Z to safe height.
    target_pos = start_pos.copy()
    target_pos[2] = SAFE_Z_HEIGHT

    # Target joints from safe pose.
    target_joints = SAFE_POSE_JOINTS

    # Create Slerp interpolator for smooth rotation transition.
    key_times = [0, 1]
    key_rots = Rotation.concatenate([start_rot_obj, target_rot_obj])
    slerp = Slerp(key_times, key_rots)

    # Create transition frames.
    end_transition = []
    for i in range(transition_frames):
      t = i / (transition_frames - 1) if transition_frames > 1 else 1.0
      t_eased = ease_out_cubic(t)  # Apply ease-out for deceleration.

      # LERP position.
      pos = start_pos * (1 - t_eased) + target_pos * t_eased

      # SLERP rotation.
      rot_obj = slerp(t_eased)
      rot = rot_obj.as_rotvec()

      # LERP joints.
      joints = start_joints * (1 - t_eased) + target_joints * t_eased

      # Combine.
      frame = np.concatenate([pos, rot, joints])
      end_transition.append(frame)

    end_transition = np.array(end_transition)
    frames = np.vstack([frames, end_transition])
    print(f"After end transition, motion shape: {frames.shape}")
    print(f"Total duration: {frames.shape[0] / fps:.2f} seconds")

  # Add padding at the end (hold final pose).
  if pad_duration > 0:
    pad_frames = int(pad_duration * fps)
    print(
      f"\nAdding {pad_duration}s padding ({pad_frames} frames) - holding final pose..."
    )

    # Repeat the last frame.
    final_frame = frames[-1:].copy()
    padding = np.tile(final_frame, (pad_frames, 1))

    frames = np.vstack([frames, padding])
    print(f"Padded motion shape: {frames.shape}")
    print(
      f"Total duration: {frames.shape[0] / fps:.2f} seconds ({frames.shape[0]} frames)"
    )

  # Parse the data.
  root_pos = frames[:, 0:3]  # (N, 3)
  root_rot_3d = frames[:, 3:6]  # (N, 3) - axis-angle
  joint_dof = frames[:, 6:]  # (N, 29)

  assert SAFE_POSE_JOINTS.shape[0] == frames.shape[1] - 6, (
      f"SAFE_POSE_JOINTS has length {SAFE_POSE_JOINTS.shape[0]}, "
      f"but motion joint_dof has length {frames.shape[1] - 6}. "
      "Make sure the keyframe joint order and PKL joint order match."
  )

  print(f"\nRoot position: {root_pos.shape}")
  print(f"Root rotation (axis-angle): {root_rot_3d.shape}")
  print(f"Joint DOF: {joint_dof.shape}")

  # Convert axis-angle rotation to quaternion (XYZW format).
  print("\nConverting rotations from axis-angle to quaternions (XYZW)...")
  quats = []
  for rot_vec in root_rot_3d:
    angle = np.linalg.norm(rot_vec)
    if angle > 1e-6:
      rotation = Rotation.from_rotvec(rot_vec)
    else:
      rotation = Rotation.from_quat([0, 0, 0, 1])

    # Get quaternion in XYZW format.
    quat_xyzw = rotation.as_quat()  # Returns [x, y, z, w]
    quats.append(quat_xyzw)

  quats = np.array(quats)  # (N, 4) in XYZW format.
  print(f"Quaternions (XYZW): {quats.shape}")

  # Combine: [root_pos(3), quat_xyzw(4), joint_dof(29)].
  csv_data = np.concatenate([root_pos, quats, joint_dof], axis=1)
  print(
    f"Final CSV shape: {csv_data.shape} (columns: 3 pos + 4 quat_xyzw + {joint_dof.shape[1]} joints)"
  )

  # Save to CSV.
  print(f"\nSaving to {csv_file}...")
  np.savetxt(csv_file, csv_data, delimiter=",", fmt="%.8f")
  print("Done!")

  # Print some stats.
  print("\nMotion stats:")
  print(f"  Duration: {csv_data.shape[0] / fps:.2f} seconds")
  print(f"  Frames: {csv_data.shape[0]}")
  print(f"  FPS: {fps}")
  print("\nOutput format: [x, y, z, qx, qy, qz, qw, joint1, joint2, ...]")


if __name__ == "__main__":
  tyro.cli(
    main,
    default={
      "pkl_file": "humanoid_backflip.pkl",
      "csv_file": "humanoid_backflip.csv",
      "duration": None,
      "pad_duration": 0.0,
      "transition_duration": 0.5,
      "add_start_transition": True,
      "add_end_transition": True,
    },
  )
