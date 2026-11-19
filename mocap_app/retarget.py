from __future__ import annotations

from math import acos, asin, atan2, degrees, sqrt
from typing import Any


MIN_VISIBILITY = 0.35
_EPSILON = 1e-6
_MIN_TORSO_HEIGHT = 0.12
_MIN_TORSO_WIDTH = 0.045
# Below this sagittal share, the knee bend direction is geometrically ambiguous.
_BEND_DIRECTION_AMBIGUITY_RATIO = 0.12

# The mapped 3D landmark convention is screen-right, vertical-up is negative Y,
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


def _torso_frame(pose: dict[str, Any], fallback_frame=None):
    """Build an orthonormal torso frame, preserving heading through width collapse.

    Shoulder width is preferred because it captures torso twist. Pelvis width
    is the geometric fallback. When both width estimates become degenerate
    (common in profile views), the previous reliable right axis is projected
    onto the current torso plane. This preserves heading while still following
    the measured torso lean and tilt.
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
    hip_width = project_width(_subtract(right_hip, left_hip))
    if _length(shoulder_width) >= _MIN_TORSO_WIDTH:
        right_hint = shoulder_width
    elif _length(hip_width) >= _MIN_TORSO_WIDTH:
        right_hint = hip_width
    elif fallback_frame is not None:
        # In a side-on view both measured width axes may collapse. Reuse the
        # previous heading, but reproject it against this frame's measured up.
        right_hint = project_width(fallback_frame[0])
        if _length(right_hint) < _EPSILON:
            previous_forward = project_width(fallback_frame[2])
            right_hint = _cross(up, previous_forward)
        if _length(right_hint) < _EPSILON:
            return None
    else:
        return None

    right = _unit(right_hint)
    if right is None:
        return None
    # Avoid a numerical 180-degree flip when a noisy width vector changes sign.
    if fallback_frame is not None and _dot(right, fallback_frame[0]) < 0.0:
        right = tuple(-component for component in right)
    forward = _unit(_cross(right, up))
    if forward is None:
        return None
    up = _unit(_cross(forward, right))
    if up is None:
        return None
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


def _bend_angle(start, joint, end, frame=None, previous_value=None) -> float:
    first = _subtract(start, joint)
    second = _subtract(end, joint)
    length_a = _length(first)
    length_b = _length(second)
    if length_a < _EPSILON or length_b < _EPSILON:
        return 0.0
    dot = _dot(first, second) / (length_a * length_b)
    inner_angle = degrees(acos(max(-1.0, min(1.0, dot))))
    magnitude = max(0.0, min(150.0, 180.0 - inner_angle))
    if frame is None or magnitude < _EPSILON:
        return magnitude

    # For the legs, preserve which side of the hip-to-ankle line the knee
    # moved toward. In this frame, forward is +local Z and positive Mine-imator
    # BEND_ANGLE_X folds the lower segment toward that forward direction.
    first_local = _local_coordinates(first, frame)
    second_local = _local_coordinates(second, frame)
    bend_axis = _cross(first_local, second_local)
    forward_bend = -bend_axis[0]
    axis_magnitude = _length(bend_axis)
    direction_ratio = (
        abs(forward_bend) / axis_magnitude
        if axis_magnitude > _EPSILON else 0.0
    )
    if (
        direction_ratio < _BEND_DIRECTION_AMBIGUITY_RATIO
        and previous_value is not None
        and abs(float(previous_value)) > _EPSILON
    ):
        # Near a profile/straight-on view, depth noise can change the sign while
        # the measured knee geometry barely moves. Keep the last reliable side.
        return magnitude if float(previous_value) > 0.0 else -magnitude
    if abs(forward_bend) < _EPSILON:
        return magnitude if previous_value is None or float(previous_value) >= 0.0 else -magnitude
    return magnitude if forward_bend > 0.0 else -magnitude


def _limb_controls(
    pose,
    start_index,
    joint_index,
    end_index,
    frame,
    signed_bend: bool = False,
    previous_controls: dict[str, float] | None = None,
):
    start = _point(pose, start_index)
    joint = _point(pose, joint_index)
    end = _point(pose, end_index)
    if start is None or joint is None or end is None or frame is None:
        return None
    rot_x, rot_z = _segment_rotation(start, joint, frame)
    return {
        "ROT_X": rot_x,
        "ROT_Y": 0.0,
        "ROT_Z": rot_z,
        "BEND_ANGLE_X": _bend_angle(
            start,
            joint,
            end,
            frame if signed_bend else None,
            (previous_controls or {}).get("BEND_ANGLE_X") if signed_bend else None,
        ),
    }


def calculate_pose_controls(
    pose: dict[str, Any],
    reference_pose: dict[str, Any] | None = None,
    previous_frame=None,
    previous_controls: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """Convert 3D body landmarks to native Mine-imator human channels.

    Torso orientation is calculated as a complete 3D frame and compared with
    the first reliable frame. Arms are expressed in the *current* torso frame,
    so a torso turn does not get added a second time to the arms. Legs use the
    fixed reference frame because their Mine-imator parent is the character
    root, which this application intentionally keeps stationary.
    """
    reference_frame = (
        _torso_frame(reference_pose)
        if reference_pose is not None
        else WORLD_REFERENCE_FRAME
    )
    if reference_frame is None:
        reference_frame = previous_frame or WORLD_REFERENCE_FRAME

    # The explicit previous frame is authoritative during a sequence. For
    # isolated conversions with a calibration pose, that pose supplies the
    # heading fallback. Never invent a temporal fallback for a missing pose.
    fallback_frame = previous_frame
    if fallback_frame is None and reference_pose is not None:
        fallback_frame = reference_frame
    current_frame = _torso_frame(pose, fallback_frame=fallback_frame)

    controls: dict[str, dict[str, float]] = {}

    if current_frame is not None:
        bend_x, bend_y, bend_z = _relative_euler(reference_frame, current_frame)
        controls["body"] = {
            "ROT_X": bend_x,
            "ROT_Y": bend_y,
            "ROT_Z": bend_z,
        }

        left_arm = _limb_controls(pose, 11, 13, 15, current_frame)
        right_arm = _limb_controls(pose, 12, 14, 16, current_frame)
        if left_arm is not None:
            controls["left_arm"] = left_arm
        if right_arm is not None:
            controls["right_arm"] = right_arm

        # MotionBERT's Human3.6M skeleton has neck and head-center joints, but
        # no nose or ears. Prefer its explicit neck-to-head axis instead of
        # fabricating facial landmarks from the shoulders (which caused large
        # false head rotations). True facial landmarks remain supported for
        # pose sources that actually provide them.
        head_axis = pose.get("head_axis")
        head_rotation = None
        if isinstance(head_axis, dict):
            neck_value = head_axis.get("neck")
            head_value = head_axis.get("head")
            confidence = float(head_axis.get("confidence", 1.0))
            if (
                isinstance(neck_value, (list, tuple))
                and isinstance(head_value, (list, tuple))
                and len(neck_value) >= 3
                and len(head_value) >= 3
                and confidence >= MIN_VISIBILITY
            ):
                neck = tuple(float(v) for v in neck_value[:3])
                head = tuple(float(v) for v in head_value[:3])
                if _length(_subtract(head, neck)) >= _EPSILON:
                    head_rotation = _segment_rotation(head, neck, current_frame)
        if head_rotation is not None:
            controls["head"] = {
                "ROT_X": head_rotation[0],
                "ROT_Z": head_rotation[1],
            }
        else:
            nose = _point(pose, 0)
            left_ear = _point(pose, 7)
            right_ear = _point(pose, 8)
            if nose is not None and left_ear is not None and right_ear is not None:
                ear_mid = _midpoint(left_ear, right_ear)
                ear_axis = _subtract(right_ear, left_ear)
                face_direction = _subtract(nose, ear_mid)
                face_local = _local_coordinates(face_direction, current_frame)
                ear_local = _local_coordinates(ear_axis, current_frame)
                controls["head"] = {
                    "ROT_X": degrees(atan2(face_local[1], max(abs(face_local[2]), 1e-5))),
                    "ROT_Z": degrees(atan2(ear_local[1], max(abs(ear_local[0]), 1e-5))),
                }

    previous_controls = previous_controls or {}
    left_leg = _limb_controls(
        pose, 23, 25, 27, reference_frame, signed_bend=True,
        previous_controls=previous_controls.get("left_leg"),
    )
    right_leg = _limb_controls(
        pose, 24, 26, 28, reference_frame, signed_bend=True,
        previous_controls=previous_controls.get("right_leg"),
    )
    if left_leg is not None:
        controls["left_leg"] = left_leg
    if right_leg is not None:
        controls["right_leg"] = right_leg

    return controls
