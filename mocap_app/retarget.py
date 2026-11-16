from __future__ import annotations

from math import acos, atan2, degrees, sqrt
from typing import Any


MIN_VISIBILITY = 0.20


def _point(pose: dict[str, Any], index: int):
    points = pose.get("world") or []
    if index >= len(points):
        return None
    value = points[index]
    if len(value) < 3:
        return None
    visibility = float(value[3]) if len(value) > 3 else 1.0
    if visibility < MIN_VISIBILITY:
        return None
    return (float(value[0]), float(value[1]), float(value[2]))


def _midpoint(a, b):
    return tuple((a[i] + b[i]) / 2.0 for i in range(3))


def _subtract(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def _segment_rotation(start, end, down_sign: float) -> tuple[float, float]:
    direction = _subtract(end, start)
    vertical = direction[1] * down_sign
    rot_x = degrees(atan2(direction[2], vertical))
    rot_z = degrees(atan2(direction[0], vertical))
    return rot_x, rot_z


def _bend_angle(start, joint, end) -> float:
    first = _subtract(start, joint)
    second = _subtract(end, joint)
    length_a = sqrt(sum(component * component for component in first))
    length_b = sqrt(sum(component * component for component in second))
    if length_a < 1e-6 or length_b < 1e-6:
        return 0.0
    dot = sum(first[i] * second[i] for i in range(3)) / (length_a * length_b)
    inner_angle = degrees(acos(max(-1.0, min(1.0, dot))))
    return max(0.0, min(150.0, 180.0 - inner_angle))


def _limb_controls(pose, start_index, joint_index, end_index, down_sign):
    start = _point(pose, start_index)
    joint = _point(pose, joint_index)
    end = _point(pose, end_index)
    if start is None or joint is None or end is None:
        return None
    rot_x, rot_z = _segment_rotation(start, joint, down_sign)
    return {
        "ROT_X": rot_x,
        "ROT_Y": 0.0,
        "ROT_Z": rot_z,
        "BEND_ANGLE_X": _bend_angle(start, joint, end),
    }


def calculate_pose_controls(pose: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Map MediaPipe world landmarks to Mine-imator's native human channels.

    The values are absolute measurements in the camera coordinate frame. The
    exporter calibrates each part against its first valid sample before writing
    keyframes, so a neutral starting stance becomes the character's rest pose.
    """
    left_knee = _point(pose, 25)
    left_ankle = _point(pose, 27)
    right_knee = _point(pose, 26)
    right_ankle = _point(pose, 28)

    vertical_directions = []
    if left_knee is not None and left_ankle is not None:
        vertical_directions.append(left_ankle[1] - left_knee[1])
    if right_knee is not None and right_ankle is not None:
        vertical_directions.append(right_ankle[1] - right_knee[1])
    down_sign = 1.0 if not vertical_directions or sum(vertical_directions) >= 0 else -1.0

    controls: dict[str, dict[str, float]] = {}

    left_shoulder = _point(pose, 11)
    right_shoulder = _point(pose, 12)
    left_hip = _point(pose, 23)
    right_hip = _point(pose, 24)

    if all(point is not None for point in (
        left_shoulder, right_shoulder, left_hip, right_hip
    )):
        shoulder_mid = _midpoint(left_shoulder, right_shoulder)
        hip_mid = _midpoint(left_hip, right_hip)
        shoulder_axis = _subtract(right_shoulder, left_shoulder)
        torso_axis = _subtract(shoulder_mid, hip_mid)

        # Mine-imator's human torso is animated with the three native bend
        # channels. Feeding this orientation into ROT_X/Y/Z makes a standing
        # user export values close to 180 degrees and twists the whole torso.
        torso_vertical = -torso_axis[1] * down_sign
        controls["body"] = {
            "BEND_ANGLE_X": degrees(atan2(torso_axis[2], torso_vertical)),
            "BEND_ANGLE_Y": degrees(atan2(-shoulder_axis[2], shoulder_axis[0])),
            "BEND_ANGLE_Z": degrees(atan2(torso_axis[0], torso_vertical)),
        }

    left_arm = _limb_controls(pose, 11, 13, 15, down_sign)
    right_arm = _limb_controls(pose, 12, 14, 16, down_sign)
    left_leg = _limb_controls(pose, 23, 25, 27, down_sign)
    right_leg = _limb_controls(pose, 24, 26, 28, down_sign)

    if left_arm is not None:
        controls["left_arm"] = left_arm
    if right_arm is not None:
        controls["right_arm"] = right_arm
    if left_leg is not None:
        controls["left_leg"] = left_leg
    if right_leg is not None:
        controls["right_leg"] = right_leg

    nose = _point(pose, 0)
    left_ear = _point(pose, 7)
    right_ear = _point(pose, 8)
    if nose is not None and left_ear is not None and right_ear is not None:
        ear_mid = _midpoint(left_ear, right_ear)
        ear_axis = _subtract(right_ear, left_ear)
        head_direction = _subtract(nose, ear_mid)
        # The reference human animates the head on X and Z. A separate Y
        # rotation is not available in that demonstrated head rig.
        controls["head"] = {
            "ROT_X": degrees(
                atan2(head_direction[1] * down_sign, max(abs(head_direction[2]), 1e-5))
            ),
            "ROT_Z": degrees(
                atan2(ear_axis[1] * down_sign, max(abs(ear_axis[0]), 1e-5))
            ),
        }

    return controls
