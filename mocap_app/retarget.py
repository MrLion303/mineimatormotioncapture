from __future__ import annotations

from math import acos, asin, atan2, cos, degrees, pi, sin, sqrt
from typing import Any, Iterable


MIN_VISIBILITY = 0.20
Vector = tuple[float, float, float]
Basis = tuple[Vector, Vector, Vector]
Quaternion = tuple[float, float, float, float]


def _point(pose: dict[str, Any], index: int) -> Vector | None:
    points = pose.get("world") or []
    if index >= len(points):
        return None
    value = points[index]
    if value is None or len(value) < 3:
        return None

    visibility = float(value[3]) if len(value) > 3 else 1.0
    presence = float(value[4]) if len(value) > 4 else 1.0
    if min(visibility, presence) < MIN_VISIBILITY:
        return None

    point = (float(value[0]), float(value[1]), float(value[2]))
    if not all(abs(component) < 1000 for component in point):
        return None
    return point


def _add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: Vector, factor: float) -> Vector:
    return (a[0] * factor, a[1] * factor, a[2] * factor)


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(value: Vector) -> float:
    return sqrt(max(0.0, _dot(value, value)))


def _normalize(value: Vector) -> Vector | None:
    magnitude = _length(value)
    if magnitude < 1e-6:
        return None
    return _scale(value, 1.0 / magnitude)


def _midpoint(a: Vector, b: Vector) -> Vector:
    return _scale(_add(a, b), 0.5)


def _body_basis(pose: dict[str, Any]) -> Basis | None:
    left_shoulder = _point(pose, 11)
    right_shoulder = _point(pose, 12)
    left_hip = _point(pose, 23)
    right_hip = _point(pose, 24)
    if any(point is None for point in (left_shoulder, right_shoulder, left_hip, right_hip)):
        return None

    shoulder_mid = _midpoint(left_shoulder, right_shoulder)
    hip_mid = _midpoint(left_hip, right_hip)
    shoulder_width = _sub(right_shoulder, left_shoulder)
    hip_width = _sub(right_hip, left_hip)

    x_axis = _normalize(_add(shoulder_width, hip_width)) or _normalize(shoulder_width)
    if x_axis is None:
        return None

    up_raw = _sub(shoulder_mid, hip_mid)
    up_orthogonal = _sub(up_raw, _scale(x_axis, _dot(up_raw, x_axis)))
    y_axis = _normalize(up_orthogonal)
    if y_axis is None:
        return None

    z_axis = _normalize(_cross(x_axis, y_axis))
    if z_axis is None:
        return None
    return (x_axis, y_axis, z_axis)



def _root_basis(pose: dict[str, Any]) -> Basis | None:
    """Estimate pelvis/root orientation independently from the upper torso."""
    left_hip = _point(pose, 23)
    right_hip = _point(pose, 24)
    left_knee = _point(pose, 25) or _point(pose, 27)
    right_knee = _point(pose, 26) or _point(pose, 28)
    if any(point is None for point in (left_hip, right_hip, left_knee, right_knee)):
        return None

    hip_mid = _midpoint(left_hip, right_hip)
    knee_mid = _midpoint(left_knee, right_knee)
    hip_width = _sub(right_hip, left_hip)
    knee_width = _sub(right_knee, left_knee)
    x_axis = _normalize(_add(hip_width, knee_width)) or _normalize(hip_width)
    if x_axis is None:
        return None

    up_raw = _sub(hip_mid, knee_mid)
    up_orthogonal = _sub(up_raw, _scale(x_axis, _dot(up_raw, x_axis)))
    y_axis = _normalize(up_orthogonal)
    if y_axis is None:
        return None

    z_axis = _normalize(_cross(x_axis, y_axis))
    if z_axis is None:
        return None
    return (x_axis, y_axis, z_axis)

def _to_local(vector: Vector, basis: Basis) -> Vector:
    return (
        _dot(vector, basis[0]),
        _dot(vector, basis[1]),
        _dot(vector, basis[2]),
    )


def _basis_in_parent(child: Basis, parent: Basis) -> Basis:
    return tuple(
        (
            _dot(axis, parent[0]),
            _dot(axis, parent[1]),
            _dot(axis, parent[2]),
        )
        for axis in child
    )  # type: ignore[return-value]


def _head_basis(pose: dict[str, Any]) -> Basis | None:
    nose = _point(pose, 0)
    left_side = _point(pose, 7) or _point(pose, 2)
    right_side = _point(pose, 8) or _point(pose, 5)
    if nose is None or left_side is None or right_side is None:
        return None

    x_axis = _normalize(_sub(right_side, left_side))
    if x_axis is None:
        return None
    side_mid = _midpoint(left_side, right_side)
    forward_raw = _sub(nose, side_mid)
    forward_orthogonal = _sub(forward_raw, _scale(x_axis, _dot(forward_raw, x_axis)))
    z_axis = _normalize(forward_orthogonal)
    if z_axis is None:
        return None
    y_axis = _normalize(_cross(z_axis, x_axis))
    if y_axis is None:
        return None
    z_axis = _normalize(_cross(x_axis, y_axis))
    if z_axis is None:
        return None
    return (x_axis, y_axis, z_axis)


def _quaternion_normalize(q: Quaternion) -> Quaternion:
    magnitude = sqrt(sum(value * value for value in q))
    if magnitude < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(value / magnitude for value in q)  # type: ignore[return-value]


def _quaternion_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return _quaternion_normalize((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ))


def _quaternion_conjugate(q: Quaternion) -> Quaternion:
    return (q[0], -q[1], -q[2], -q[3])


def _quaternion_from_euler_xyz(angles: tuple[float, float, float]) -> Quaternion:
    x, y, z = (value * pi / 360.0 for value in angles)
    cx, sx = cos(x), sin(x)
    cy, sy = cos(y), sin(y)
    cz, sz = cos(z), sin(z)
    return _quaternion_normalize((
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ))


def _quaternion_to_euler_xyz(q: Quaternion) -> tuple[float, float, float]:
    w, x, y, z = _quaternion_normalize(q)

    angle_x = atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    sin_y = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    angle_y = asin(sin_y)
    angle_z = atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return (degrees(angle_x), degrees(angle_y), degrees(angle_z))


def _quaternion_from_basis(basis: Basis) -> Quaternion:
    x_axis, y_axis, z_axis = basis
    m00, m01, m02 = x_axis[0], y_axis[0], z_axis[0]
    m10, m11, m12 = x_axis[1], y_axis[1], z_axis[1]
    m20, m21, m22 = x_axis[2], y_axis[2], z_axis[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        scale = sqrt(trace + 1.0) * 2.0
        q = (
            0.25 * scale,
            (m21 - m12) / scale,
            (m02 - m20) / scale,
            (m10 - m01) / scale,
        )
    elif m00 > m11 and m00 > m22:
        scale = sqrt(max(1e-12, 1.0 + m00 - m11 - m22)) * 2.0
        q = (
            (m21 - m12) / scale,
            0.25 * scale,
            (m01 + m10) / scale,
            (m02 + m20) / scale,
        )
    elif m11 > m22:
        scale = sqrt(max(1e-12, 1.0 + m11 - m00 - m22)) * 2.0
        q = (
            (m02 - m20) / scale,
            (m01 + m10) / scale,
            0.25 * scale,
            (m12 + m21) / scale,
        )
    else:
        scale = sqrt(max(1e-12, 1.0 + m22 - m00 - m11)) * 2.0
        q = (
            (m10 - m01) / scale,
            (m02 + m20) / scale,
            (m12 + m21) / scale,
            0.25 * scale,
        )
    return _quaternion_normalize(q)


def _quaternion_between_vectors(start: Vector, end: Vector) -> Quaternion:
    a = _normalize(start)
    b = _normalize(end)
    if a is None or b is None:
        return (1.0, 0.0, 0.0, 0.0)

    dot = max(-1.0, min(1.0, _dot(a, b)))
    cross = _cross(a, b)
    if dot < -0.999999:
        fallback = (1.0, 0.0, 0.0) if abs(a[0]) < 0.8 else (0.0, 1.0, 0.0)
        axis = _normalize(_cross(a, fallback)) or (0.0, 0.0, 1.0)
        return (0.0, axis[0], axis[1], axis[2])
    return _quaternion_normalize((1.0 + dot, cross[0], cross[1], cross[2]))


def _relative_basis_rotation(reference: Basis, current: Basis) -> Quaternion:
    reference_q = _quaternion_from_basis(reference)
    current_q = _quaternion_from_basis(current)
    return _quaternion_multiply(_quaternion_conjugate(reference_q), current_q)


def _base_rotation(
    base_values: dict[str, dict[str, float]] | None,
    part: str,
) -> tuple[float, float, float]:
    values = (base_values or {}).get(part, {})
    return (
        float(values.get("ROT_X", 0.0)),
        float(values.get("ROT_Y", 0.0)),
        float(values.get("ROT_Z", 0.0)),
    )


def _apply_rotation_delta(
    base_values: dict[str, dict[str, float]] | None,
    part: str,
    delta: Quaternion,
) -> dict[str, float]:
    # Mine-imator's human limbs carry baked-in rest angles (for example,
    # left arm ROT_Y around +120 degrees and right arm around -60 degrees).
    # Add local motion offsets to those values instead of composing a world
    # quaternion with the baked Euler pose, which can create Euler flips.
    base_x, base_y, base_z = _base_rotation(base_values, part)
    delta_x, delta_y, delta_z = _quaternion_to_euler_xyz(delta)
    return {
        "ROT_X": base_x + delta_x,
        "ROT_Y": base_y + delta_y,
        "ROT_Z": base_z + delta_z,
    }


def _bend_angle(start: Vector, joint: Vector, end: Vector) -> float:
    upper = _normalize(_sub(start, joint))
    lower = _normalize(_sub(end, joint))
    if upper is None or lower is None:
        return 0.0
    dot = max(-1.0, min(1.0, _dot(upper, lower)))
    interior = degrees(acos(dot))
    return max(0.0, min(150.0, 180.0 - interior))


def _limb_points(pose: dict[str, Any], indices: tuple[int, int, int]):
    points = tuple(_point(pose, index) for index in indices)
    if any(point is None for point in points):
        return None
    return points


def _limb_controls(
    pose: dict[str, Any],
    reference_pose: dict[str, Any],
    current_body: Basis,
    reference_body: Basis,
    current_root: Basis,
    reference_root: Basis,
    root_delta: Quaternion,
    part: str,
    indices: tuple[int, int, int],
    base_values: dict[str, dict[str, float]] | None,
    part_parents: dict[str, str] | None = None,
) -> dict[str, float] | None:
    current = _limb_points(pose, indices)
    reference = _limb_points(reference_pose, indices)
    if current is None or reference is None:
        return None

    default_parent = "body" if part in ("left_arm", "right_arm") else "root"
    parent_part = (part_parents or {}).get(part, default_parent)
    if parent_part == "body":
        current_parent = current_body
        reference_parent = reference_body
        inherits_root = False
    else:
        current_parent = current_root
        reference_parent = reference_root
        inherits_root = True

    current_vector = _to_local(_sub(current[1], current[0]), current_parent)
    reference_vector = _to_local(_sub(reference[1], reference[0]), reference_parent)
    if _length(current_vector) < 1e-4 or _length(reference_vector) < 1e-4:
        return None

    local_delta = _quaternion_between_vectors(reference_vector, current_vector)
    # Legs are direct children of the character root in the supplied rigs.
    # They inherit pelvis yaw, but do not compensate for a torso-only twist.
    delta_q = (
        _quaternion_multiply(root_delta, local_delta)
        if inherits_root
        else local_delta
    )
    result = _apply_rotation_delta(base_values, part, delta_q)

    current_bend = _bend_angle(current[0], current[1], current[2])
    reference_bend = _bend_angle(reference[0], reference[1], reference[2])
    base_bend = float((base_values or {}).get(part, {}).get("BEND_ANGLE_X", 0.0))
    result["BEND_ANGLE_X"] = max(
        0.0,
        min(180.0, base_bend + current_bend - reference_bend),
    )
    return result


def calculate_pose_controls(
    pose: dict[str, Any],
    reference_pose: dict[str, Any] | None = None,
    base_values: dict[str, dict[str, float]] | None = None,
    part_parents: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Calculate local rotations using the actual Mine-imator part hierarchy."""
    reference_pose = reference_pose or pose
    current_body = _body_basis(pose)
    reference_body = _body_basis(reference_pose)
    if current_body is None or reference_body is None:
        return {}

    current_root = _root_basis(pose) or current_body
    reference_root = _root_basis(reference_pose) or reference_body
    body_delta = _relative_basis_rotation(reference_body, current_body)
    root_delta = _relative_basis_rotation(reference_root, current_root)

    controls: dict[str, dict[str, float]] = {}
    controls["body"] = _apply_rotation_delta(base_values, "body", body_delta)

    current_head = _head_basis(pose)
    reference_head = _head_basis(reference_pose)
    if current_head is not None and reference_head is not None:
        head_parent = (part_parents or {}).get("head", "body")
        if head_parent == "body":
            current_head_local = _basis_in_parent(current_head, current_body)
            reference_head_local = _basis_in_parent(reference_head, reference_body)
        else:
            current_head_local = _basis_in_parent(current_head, current_root)
            reference_head_local = _basis_in_parent(reference_head, reference_root)
        head_delta = _relative_basis_rotation(reference_head_local, current_head_local)
        if head_parent != "body":
            head_delta = _quaternion_multiply(root_delta, head_delta)
        controls["head"] = _apply_rotation_delta(base_values, "head", head_delta)

    limb_specs = (
        ("left_arm", (11, 13, 15)),
        ("right_arm", (12, 14, 16)),
        ("left_leg", (23, 25, 27)),
        ("right_leg", (24, 26, 28)),
    )
    for part, indices in limb_specs:
        limb = _limb_controls(
            pose,
            reference_pose,
            current_body,
            reference_body,
            current_root,
            reference_root,
            root_delta,
            part,
            indices,
            base_values,
            part_parents=part_parents,
        )
        if limb is not None:
            controls[part] = limb

    return controls


class _OneEuroFilter:
    def __init__(
        self,
        min_cutoff: float = 1.8,
        beta: float = 0.12,
        derivative_cutoff: float = 1.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.previous_value: float | None = None
        self.previous_derivative = 0.0
        self.previous_time: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * pi * max(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / max(dt, 1e-4))

    def filter(self, value: float, timestamp: float) -> float:
        if self.previous_value is None or self.previous_time is None:
            self.previous_value = value
            self.previous_time = timestamp
            return value

        dt = max(1.0 / 120.0, min(0.25, timestamp - self.previous_time))
        derivative = (value - self.previous_value) / dt
        derivative_alpha = self._alpha(self.derivative_cutoff, dt)
        filtered_derivative = (
            derivative_alpha * derivative
            + (1.0 - derivative_alpha) * self.previous_derivative
        )
        cutoff = self.min_cutoff + self.beta * abs(filtered_derivative)
        value_alpha = self._alpha(cutoff, dt)
        filtered_value = (
            value_alpha * value
            + (1.0 - value_alpha) * self.previous_value
        )

        self.previous_value = filtered_value
        self.previous_derivative = filtered_derivative
        self.previous_time = timestamp
        return filtered_value


def stabilize_pose_samples(
    samples: Iterable[tuple[float, dict[str, Any]]],
    enabled: bool = True,
) -> list[tuple[float, dict[str, Any]]]:
    """Apply time-aware One Euro filtering to 3D landmarks before retargeting."""
    ordered = sorted(
        ((float(timestamp), pose) for timestamp, pose in samples),
        key=lambda item: item[0],
    )
    if not enabled or not ordered:
        return ordered

    filters: dict[int, tuple[_OneEuroFilter, _OneEuroFilter, _OneEuroFilter]] = {}
    stabilized: list[tuple[float, dict[str, Any]]] = []

    for timestamp, pose in ordered:
        world = pose.get("world") or []
        filtered_world = []
        for index, point in enumerate(world):
            if point is None or len(point) < 3:
                filtered_world.append(point)
                continue

            visibility = float(point[3]) if len(point) > 3 else 1.0
            presence = float(point[4]) if len(point) > 4 else 1.0
            if min(visibility, presence) < MIN_VISIBILITY:
                filtered_world.append(list(point))
                continue

            point_filters = filters.setdefault(
                index,
                (_OneEuroFilter(), _OneEuroFilter(), _OneEuroFilter()),
            )
            filtered = [
                point_filters[axis].filter(float(point[axis]), timestamp)
                for axis in range(3)
            ]
            filtered.extend(float(value) for value in point[3:])
            filtered_world.append(filtered)

        filtered_pose = dict(pose)
        filtered_pose["world"] = filtered_world
        stabilized.append((timestamp, filtered_pose))

    return stabilized
