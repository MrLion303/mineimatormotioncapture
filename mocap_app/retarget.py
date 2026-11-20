from __future__ import annotations

from math import acos, asin, atan2, degrees, sqrt
from typing import Any

import numpy as np


MIN_VISIBILITY = 0.35
_EPSILON = 1e-6
_MIN_TORSO_HEIGHT = 0.12
_MIN_TORSO_WIDTH = 0.045

# MediaPipe world coordinates: screen-right, vertical-up is negative Y,
# and the default front-facing depth axis points toward negative Z.
WORLD_REFERENCE_FRAME = (
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, -1.0),
)


def _point(pose: dict[str, Any], index: int):
    points = pose.get("world") or []
    if index >= len(points):
        return None
    value = points[index]
    if len(value) < 3:
        return None
    visibility = float(value[3]) if len(value) > 3 else 1.0
    presence = float(value[4]) if len(value) > 4 else visibility
    if min(visibility, presence) < MIN_VISIBILITY:
        return None
    return (float(value[0]), float(value[1]), float(value[2]))


def _midpoint(a, b):
    return tuple((a[i] + b[i]) * 0.5 for i in range(3))


def _subtract(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def _dot(a, b) -> float:
    return sum(a[i] * b[i] for i in range(3))


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(vector) -> float:
    return sqrt(_dot(vector, vector))


def _unit(vector):
    magnitude = _length(vector)
    if magnitude < _EPSILON:
        return None
    return tuple(component / magnitude for component in vector)


def _torso_frame(pose: dict[str, Any], face_flip: bool | None = None):
    """Build a stable orthonormal torso frame from shoulder and hip landmarks.

    Shoulder width is preferred because it captures torso twist. If that axis
    becomes almost parallel to the torso's up axis, its projected width is
    unreliable; in that case the pelvis width is used as a geometric fallback.
    The resulting forward axis can be anchored to the nose-to-ears direction.
    For a captured take this handedness is chosen once from the first complete
    torso frame and then held fixed, so head turns cannot flip the torso frame.
    Degenerate torso height or width is rejected instead of producing a frame
    whose axes can jump dramatically from one video frame to the next.
    """
    left_shoulder = _point(pose, 11)
    right_shoulder = _point(pose, 12)
    left_hip = _point(pose, 23)
    right_hip = _point(pose, 24)
    if any(point is None for point in (
        left_shoulder, right_shoulder, left_hip, right_hip
    )):
        return None

    shoulder_mid = _midpoint(left_shoulder, right_shoulder)
    hip_mid = _midpoint(left_hip, right_hip)
    torso_axis = _subtract(shoulder_mid, hip_mid)
    if _length(torso_axis) < _MIN_TORSO_HEIGHT:
        return None

    up = _unit(torso_axis)
    if up is None:
        return None

    def project_width(axis):
        along_up = _dot(axis, up)
        return tuple(axis[i] - along_up * up[i] for i in range(3))

    shoulder_width = project_width(_subtract(right_shoulder, left_shoulder))
    if _length(shoulder_width) >= _MIN_TORSO_WIDTH:
        right_hint = shoulder_width
    else:
        # A nearly vertical shoulder axis is a poor estimate of torso width.
        # The pelvis provides a more stable left/right reference in that frame.
        hip_width = project_width(_subtract(right_hip, left_hip))
        if _length(hip_width) < _MIN_TORSO_WIDTH:
            return None
        right_hint = hip_width

    right = _unit(right_hint)
    if right is None:
        return None
    forward = _unit(_cross(right, up))
    if forward is None:
        return None
    up = _unit(_cross(forward, right))
    if up is None:
        return None

    # Pose landmark left/right labels can make the shoulder-width vector point
    # opposite to the face direction (especially in front-facing footage).
    # Keep the torso frame aligned with the character's front so yaw, arms and
    # head are expressed in the same forward-facing coordinate system.
    if face_flip is None:
        nose = _point(pose, 0)
        left_ear = _point(pose, 7)
        right_ear = _point(pose, 8)
        if nose is not None and left_ear is not None and right_ear is not None:
            face_direction = _subtract(nose, _midpoint(left_ear, right_ear))
            face_flip = _dot(forward, face_direction) < 0.0

    if face_flip:
        right = tuple(-component for component in right)
        forward = tuple(-component for component in forward)
    return right, up, forward

def _relative_euler(reference_frame, current_frame) -> tuple[float, float, float]:
    """Return XYZ Euler angles for reference_frame.T * current_frame."""
    matrix = [
        [_dot(reference_frame[row], current_frame[column]) for column in range(3)]
        for row in range(3)
    ]
    sin_y = max(-1.0, min(1.0, -matrix[2][0]))
    angle_y = asin(sin_y)
    cos_y = sqrt(max(0.0, 1.0 - sin_y * sin_y))
    if cos_y > 1e-5:
        angle_x = atan2(matrix[2][1], matrix[2][2])
        angle_z = atan2(matrix[1][0], matrix[0][0])
    else:
        # Stable fallback near the Euler gimbal singularity.
        angle_x = atan2(-matrix[1][2], matrix[1][1])
        angle_z = 0.0
    return degrees(angle_x), degrees(angle_y), degrees(angle_z)


def _local_coordinates(vector, frame):
    right, up, forward = frame
    return _dot(vector, right), _dot(vector, up), _dot(vector, forward)


def _segment_rotation(start, end, frame) -> tuple[float, float]:
    """Map one 3D bone direction to a coherent two-axis Mine-imator rotation.

    ROT_X is the forward tilt and ROT_Z is the side tilt. They are derived as
    a composed rotation, not as two independent angles against the vertical;
    therefore a diagonal movement is represented by one consistent 3D vector.
    """
    direction = _subtract(end, start)
    local_direction = _unit(_local_coordinates(direction, frame))
    if local_direction is None:
        return 0.0, 0.0

    local_right, local_up, local_forward = local_direction
    rot_x = degrees(asin(max(-1.0, min(1.0, local_forward))))
    rot_z = degrees(atan2(local_right, -local_up))
    return rot_x, rot_z


def _bend_angle(start, joint, end) -> float:
    first = _subtract(start, joint)
    second = _subtract(end, joint)
    length_a = _length(first)
    length_b = _length(second)
    if length_a < _EPSILON or length_b < _EPSILON:
        return 0.0
    dot = _dot(first, second) / (length_a * length_b)
    inner_angle = degrees(acos(max(-1.0, min(1.0, dot))))
    return max(0.0, min(150.0, 180.0 - inner_angle))


def _matrix_from_frame(frame):
    """Return a local-to-world matrix whose columns are right/up/forward."""
    return np.column_stack(tuple(np.asarray(axis, dtype=float) for axis in frame))


def _matrix_to_yxz(
    matrix,
    previous_angles: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    """Extract continuous intrinsic Y-X-Z Euler angles from a rotation matrix.

    A matrix has two equivalent non-singular Y-X-Z Euler solutions. Selecting
    the principal branch independently each frame can flip Y/Z by 180 degrees
    near a straight limb and let the later 360-degree unwrap accumulate a false
    spin. Choose the equivalent solution closest to the previous frame. At
    +/-90 degrees, Y and Z are coupled, so preserve previous Z and solve Y from
    the observable combined angle.
    """
    sin_x = max(-1.0, min(1.0, -float(matrix[1, 2])))
    angle_x = asin(sin_x)
    cos_x = sqrt(max(0.0, 1.0 - sin_x * sin_x))

    if cos_x > 1e-6:
        angle_y = atan2(float(matrix[0, 2]), float(matrix[2, 2]))
        angle_z = atan2(float(matrix[1, 0]), float(matrix[1, 1]))
        principal = np.degrees((angle_y, angle_x, angle_z))
        if previous_angles is None:
            return tuple(float(value) for value in principal)

        previous = np.asarray(previous_angles, dtype=float)
        alternative = np.asarray((
            principal[0] + 180.0,
            180.0 - principal[1],
            principal[2] + 180.0,
        ))
        candidates = []
        for candidate in (principal, alternative):
            candidate = candidate + 360.0 * np.round((previous - candidate) / 360.0)
            candidates.append(candidate)
        best = min(
            candidates,
            key=lambda candidate: float(np.sum((candidate - previous) ** 2)),
        )
        return tuple(float(value) for value in best)

    angle_x_degrees = degrees(angle_x)
    if previous_angles is None:
        angle_y = atan2(-float(matrix[2, 0]), float(matrix[0, 0]))
        return degrees(angle_y), angle_x_degrees, 0.0

    previous_y, _, previous_z = (float(value) for value in previous_angles)
    if sin_x >= 0.0:
        # At +90 degrees the observable combination is Y - Z.
        combined = atan2(float(matrix[0, 1]), float(matrix[0, 0]))
        angle_z_degrees = previous_z
        angle_y_degrees = degrees(combined) + angle_z_degrees
    else:
        # At -90 degrees the observable combination is Y + Z.
        combined = atan2(-float(matrix[0, 1]), float(matrix[0, 0]))
        angle_z_degrees = previous_z
        angle_y_degrees = degrees(combined) - angle_z_degrees

    angle_y_degrees += 360.0 * round((previous_y - angle_y_degrees) / 360.0)
    angle_z_degrees += 360.0 * round((previous_z - angle_z_degrees) / 360.0)
    return angle_y_degrees, angle_x_degrees, angle_z_degrees


def _rotation_y(angle_degrees: float):
    angle = np.radians(float(angle_degrees))
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.asarray(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
        dtype=float,
    )


def _project_axis(axis, direction):
    vector = np.asarray(axis, dtype=float)
    unit_direction = np.asarray(direction, dtype=float)
    return vector - float(np.dot(vector, unit_direction)) * unit_direction


def _bone_frame(
    pose,
    start_index: int,
    joint_index: int,
    end_index: int,
    x_direction: float,
    fallback_right,
    previous_frame=None,
):
    """Build a full 3D bone frame and keep its bend-plane sign continuous."""
    start = _point(pose, start_index)
    joint = _point(pose, joint_index)
    end = _point(pose, end_index)
    if start is None or joint is None or end is None:
        return None

    upper = np.asarray(_subtract(start, joint), dtype=float)
    lower = np.asarray(_subtract(end, joint), dtype=float)
    upper_length = float(np.linalg.norm(upper))
    lower_length = float(np.linalg.norm(lower))
    if upper_length < _EPSILON or lower_length < _EPSILON:
        return None
    local_y = upper / upper_length

    bend_plane = np.cross(float(x_direction) * lower, local_y)
    plane_size = float(np.linalg.norm(bend_plane))
    if plane_size < 0.035:
        preferred = previous_frame[:, 0] if previous_frame is not None else fallback_right
        local_x = _project_axis(preferred, local_y)
        x_length = float(np.linalg.norm(local_x))
        if x_length < _EPSILON:
            return None
        local_x /= x_length
    else:
        local_x = bend_plane / plane_size

    if previous_frame is not None and float(np.dot(local_x, previous_frame[:, 0])) < 0.0:
        local_x = -local_x

    local_z = np.cross(local_x, local_y)
    z_length = float(np.linalg.norm(local_z))
    if z_length < _EPSILON:
        return None
    local_z /= z_length
    local_x = np.cross(local_y, local_z)
    local_x /= max(float(np.linalg.norm(local_x)), _EPSILON)
    return np.column_stack((local_x, local_y, local_z)), local_y, lower / lower_length


def _limb_controls(
    pose,
    start_index,
    joint_index,
    end_index,
    reference_matrix,
    x_direction,
    previous_frame=None,
    previous_angles=None,
):
    """Retarget one limb with three local Euler axes plus native X bending."""
    bone = _bone_frame(
        pose,
        start_index,
        joint_index,
        end_index,
        x_direction,
        reference_matrix[:, 0],
        previous_frame,
    )
    if bone is None:
        return None
    bone_matrix, upper_direction, lower_direction = bone
    relative = reference_matrix.T @ bone_matrix
    angle_y, angle_x, angle_z = _matrix_to_yxz(
        relative,
        previous_angles=previous_angles,
    )
    bend = degrees(acos(max(-1.0, min(1.0, -float(np.dot(
        upper_direction, lower_direction
    ))))))
    return {
        "ROT_X": angle_x,
        "ROT_Y": angle_y,
        "ROT_Z": angle_z,
        "BEND_ANGLE_X": max(0.0, min(150.0, bend)),
    }, bone_matrix, (angle_y, angle_x, angle_z)


def _head_frame(pose, body_matrix, previous_frame=None):
    nose = _point(pose, 0)
    left_ear = _point(pose, 7)
    right_ear = _point(pose, 8)
    if nose is None or left_ear is None or right_ear is None:
        return None

    face_forward = np.asarray(_subtract(
        nose, _midpoint(left_ear, right_ear)
    ), dtype=float)
    forward_length = float(np.linalg.norm(face_forward))
    if forward_length < _EPSILON:
        return None
    forward = face_forward / forward_length

    right_hint = np.asarray(_subtract(right_ear, left_ear), dtype=float)
    right_hint = _project_axis(right_hint, forward)
    right_length = float(np.linalg.norm(right_hint))
    if right_length < _EPSILON:
        return None
    right = right_hint / right_length

    sign_reference = (
        previous_frame[:, 0]
        if previous_frame is not None
        else body_matrix[:, 0]
    )
    if float(np.dot(right, sign_reference)) < 0.0:
        right = -right

    up = np.cross(forward, right)
    up_length = float(np.linalg.norm(up))
    if up_length < _EPSILON:
        return None
    up /= up_length
    right = np.cross(up, forward)
    right /= max(float(np.linalg.norm(right)), _EPSILON)
    return np.column_stack((right, up, forward))


def _head_controls(pose, body_matrix, previous_frame=None, previous_angles=None):
    head_matrix = _head_frame(pose, body_matrix, previous_frame)
    if head_matrix is None:
        return None
    relative = body_matrix.T @ head_matrix
    angle_y, angle_x, angle_z = _matrix_to_yxz(
        relative,
        previous_angles=previous_angles,
    )
    return {
        "ROT_X": angle_x,
        "ROT_Y": angle_y,
        "ROT_Z": angle_z,
    }, head_matrix, (angle_y, angle_x, angle_z)


def _reference_setup(reference_pose, fallback_pose=None):
    if reference_pose is not None:
        unflipped = _torso_frame(reference_pose, face_flip=False)
        aligned = _torso_frame(reference_pose, face_flip=None)
        face_flip = (
            unflipped is not None
            and aligned is not None
            and _dot(unflipped[0], aligned[0]) < 0.0
        )
        return _torso_frame(reference_pose, face_flip=face_flip), face_flip

    if fallback_pose is not None:
        return _torso_frame(fallback_pose, face_flip=None), None
    return None, None


def _controls_for_pose(
    pose, reference_frame, face_flip, previous_frames, previous_angles
):
    current_frame = _torso_frame(pose, face_flip=face_flip)
    controls: dict[str, dict[str, float]] = {}
    reference_frame = reference_frame or WORLD_REFERENCE_FRAME
    reference_matrix = _matrix_from_frame(reference_frame)

    current_matrix = (
        _matrix_from_frame(current_frame)
        if current_frame is not None
        else None
    )
    if current_matrix is not None:
        tilt_x, yaw_y, roll_z = _relative_euler(reference_frame, current_frame)
        controls["body"] = {
            "BEND_ANGLE_X": tilt_x,
            "BEND_ANGLE_Y": roll_z,
            "BEND_ANGLE_Z": yaw_y,
        }

        for part, indices in (
            ("left_arm", (11, 13, 15)),
            ("right_arm", (12, 14, 16)),
        ):
            limb = _limb_controls(
                pose, *indices, current_matrix, -1.0,
                previous_frames.get(part),
                previous_angles.get(part),
            )
            if limb is not None:
                limb_controls, bone_matrix, angles = limb
                controls[part] = limb_controls
                previous_frames[part] = bone_matrix
                previous_angles[part] = angles

        head = _head_controls(
            pose,
            current_matrix,
            previous_frames.get("head"),
            previous_angles.get("head"),
        )
        if head is not None:
            head_controls, head_matrix, angles = head
            controls["head"] = head_controls
            previous_frames["head"] = head_matrix
            previous_angles["head"] = angles

        body_yaw, _, _ = _matrix_to_yxz(reference_matrix.T @ current_matrix)
        leg_matrix = reference_matrix @ _rotation_y(body_yaw)
    else:
        leg_matrix = reference_matrix

    for part, indices in (
        ("left_leg", (23, 25, 27)),
        ("right_leg", (24, 26, 28)),
    ):
        limb = _limb_controls(
            pose, *indices, leg_matrix, 1.0,
            previous_frames.get(part),
            previous_angles.get(part),
        )
        if limb is not None:
            limb_controls, bone_matrix, angles = limb
            controls[part] = limb_controls
            previous_frames[part] = bone_matrix
            previous_angles[part] = angles

    return controls


def calculate_pose_controls(
    pose: dict[str, Any],
    reference_pose: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """Convert one MediaPipe pose to full 3D Mine-imator human controls."""
    reference_frame, face_flip = _reference_setup(reference_pose, pose)
    return _controls_for_pose(pose, reference_frame, face_flip, {}, {})


def calculate_pose_sequence(
    samples,
    reference_pose: dict[str, Any] | None = None,
) -> list[tuple[float, dict[str, dict[str, float]]]]:
    """Convert a take in chronological order, preserving bone-frame continuity."""
    samples = list(samples)
    if not samples:
        return []
    if reference_pose is None:
        reference_pose = next(
            (
                pose for _, pose in samples
                if _torso_frame(pose) is not None
            ),
            samples[0][1],
        )

    reference_frame, face_flip = _reference_setup(reference_pose)
    previous_frames: dict[str, np.ndarray] = {}
    previous_angles: dict[str, tuple[float, float, float]] = {}
    converted = []
    for timestamp, pose in samples:
        controls = _controls_for_pose(
            pose, reference_frame, face_flip, previous_frames, previous_angles
        )
        converted.append((max(0.0, float(timestamp)), controls))
    return converted


